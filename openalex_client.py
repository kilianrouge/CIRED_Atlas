"""
openalex_client.py – versatile OpenAlex search layer for ATLAS.

Supported condition types
─────────────────────────
  keywords_title_abstract   full-text search in title + abstract
  keywords_title            full-text search in title only
  author                    single OA author ID (A…)
  journal                   single OA source ID (S…)
  topic                     OA topic ID
  field                     OA field ID
  domain                    OA domain ID
  citing_library            papers that cite items in the Zotero collection
  author_in_library         recent papers by prolific authors in the collection
  zotero_collection         papers whose DOI/OA-ID matches a specific Zotero sub-collection

Each condition producing OpenAlex Work dicts; results are then merged,
deduplicated and returned with a ``_reasons`` list for UI display.

Conditions within a profile are combined as independent OR (union) searches
(the standard literature-alert paradigm); the ``combinator`` field is exposed
for UI labelling but true AND-filtering across heterogeneous OpenAlex queries
would require client-side intersection and is not yet implemented.
"""
from __future__ import annotations

import math
import time
import warnings
from collections import Counter
from datetime import date, timedelta
from typing import Any

import pyalex
from pyalex import Works, Authors, Sources

import database as db

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── pyalex config ─────────────────────────────────────────────────────────────

def _configure_pyalex() -> None:
    email = db.get_setting("openalex_email", "")
    if email:
        # httpx encodes headers as ASCII; strip/replace any non-ASCII chars
        email = email.encode("ascii", errors="ignore").decode("ascii").strip()
    if email:
        pyalex.config.email = email
    pyalex.config.max_retries = 0   # we implement our own backoff


_SELECT_FIELDS = [
    "id", "doi", "title", "abstract_inverted_index",
    "authorships", "publication_year", "primary_location",
    "topics", "primary_topic", "cited_by_count",
]

# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _since_date(lookback_days: int) -> str:
    # Add a 2-day grace buffer so papers at the exact edge of the window aren't missed
    # (e.g. a paper published exactly `lookback_days` ago would otherwise be excluded
    # since OA uses strict >= on from_publication_date).
    return (date.today() - timedelta(days=lookback_days + 2)).isoformat()


def _lang_type_filter(cond: dict,
                      default_lang: str = "en",
                      default_type: str = "article") -> dict:
    """Return a partial OA filter dict for language and type from a condition."""
    f: dict = {}
    lang = cond.get("language", default_lang)
    if lang and lang != "any":
        f["language"] = lang
    doc_type = cond.get("doc_type", default_type)
    if doc_type and doc_type != "any":
        f["type"] = doc_type
    return f


def _build_scope_filter(scope_conds: list[dict]) -> dict:
    """
    Build an OpenAlex extra_filter dict from conditions marked ``scope: true``.
    These filters are AND-injected into every non-scope condition's query.
    Supported scope types: field, domain, journal, language, doc_type.
    """
    f: dict = {}
    for sc in scope_conds:
        ctype = sc.get("type", "")
        val   = (sc.get("value") or "").strip()
        if not val:
            continue
        if ctype == "field":
            f["topics.field.id"] = val
        elif ctype == "domain":
            f["topics.domain.id"] = val
        elif ctype == "journal":
            f["primary_location.source.id"] = val
        elif ctype == "keywords_title_abstract":
            f["title_and_abstract.search"] = val
        elif ctype == "keywords_title":
            f["title.search"] = val
        elif ctype == "language":
            if val != "any":
                f["language"] = val
        elif ctype == "doc_type":
            if val != "any":
                f["type"] = val
    return f


def _paginate(query, max_results: int) -> list[dict]:
    results: list[dict] = []
    per_page = min(200, max(1, max_results))
    pager = query.paginate(per_page=per_page, n_max=max_results)
    for attempt in range(6):
        try:
            for page in pager:
                results.extend(page)
                if len(results) >= max_results:
                    break
                time.sleep(0.25)
            break
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "too many" in msg.lower():
                wait = 8 * (2 ** attempt)
                time.sleep(wait)
                pager = query.paginate(per_page=per_page, n_max=max_results)
                results = []
            else:
                raise
    return results[:max_results]


def _get_with_backoff(query, per_page: int = 200) -> list[dict]:
    for attempt in range(5):
        try:
            return query.get(per_page=per_page) or []
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "too many" in msg.lower():
                time.sleep(10 * (2 ** attempt))
            else:
                raise
    raise RuntimeError("OpenAlex: too many 429 retries")


def _deduplicate(papers: list[dict], seen: set[str],
                 existing_dois: set[str], existing_oa_ids: set[str]) -> list[dict]:
    out = []
    for p in papers:
        oa_id = (p.get("id") or "").replace("https://openalex.org/", "")
        doi = (p.get("doi") or "").replace("https://doi.org/", "").lower().strip()
        if oa_id in existing_oa_ids or oa_id in seen:
            continue
        if doi and doi in existing_dois:
            continue
        seen.add(oa_id)
        out.append(p)
    return out


def reconstruct_abstract(paper: dict) -> str:
    """Rebuild human-readable abstract from OpenAlex inverted index."""
    inv: dict[str, list[int]] | None = paper.get("abstract_inverted_index")
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, pos_list in inv.items():
        for pos in pos_list:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def paper_text(paper: dict) -> str:
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or reconstruct_abstract(paper)
    return f"{title}. {abstract}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Individual condition handlers
# ─────────────────────────────────────────────────────────────────────────────

def _search_keywords_title_abstract(cond: dict, lookback_days: int,
                                     seen: set[str], existing_dois: set[str],
                                     existing_oa_ids: set[str],
                                     max_results: int = 150) -> list[dict]:
    """Search title+abstract with the given search string (supports AND/OR/wildcards * and "phrases")."""
    value: str = cond.get("value", "")
    extra_filter: dict = cond.get("extra_filter") or {}
    exclude_words: list[str] = [w.lower() for w in (cond.get("exclude_words") or [])]

    oa_filter: dict = {
        "title_and_abstract.search": value,
        **_lang_type_filter(cond),
        "from_publication_date": _since_date(lookback_days),
        **extra_filter,
    }
    query = (
        Works()
        .filter(**oa_filter)
        .select(_SELECT_FIELDS)
        .sort(cited_by_count="desc")
    )
    raw = _paginate(query, max_results * 3)
    if exclude_words:
        raw = [p for p in raw if not any(w in (p.get("title") or "").lower() for w in exclude_words)]
    return _deduplicate(raw, seen, existing_dois, existing_oa_ids)


def _search_keywords_title(cond: dict, lookback_days: int,
                            seen: set[str], existing_dois: set[str],
                            existing_oa_ids: set[str],
                            max_results: int = 150) -> list[dict]:
    value: str = cond.get("value", "")
    extra_filter: dict = cond.get("extra_filter") or {}
    oa_filter: dict = {
        "title.search": value,
        **_lang_type_filter(cond),
        "from_publication_date": _since_date(lookback_days),
        **extra_filter,
    }
    query = (
        Works()
        .filter(**oa_filter)
        .select(_SELECT_FIELDS)
        .sort(cited_by_count="desc")
    )
    raw = _paginate(query, max_results * 3)
    return _deduplicate(raw, seen, existing_dois, existing_oa_ids)


def _search_by_author(cond: dict, lookback_days: int,
                      seen: set[str], existing_dois: set[str],
                      existing_oa_ids: set[str],
                      max_results: int = 30) -> list[dict]:
    author_id: str = cond.get("value", "")
    if not author_id:
        return []
    extra_filter: dict = cond.get("extra_filter") or {}
    oa_filter = {
        "author.id": author_id,
        "from_publication_date": _since_date(lookback_days),
        **_lang_type_filter(cond),
        **extra_filter,
    }
    query = Works().filter(**oa_filter).select(_SELECT_FIELDS).sort(publication_date="desc")
    raw = _paginate(query, max_results)
    return _deduplicate(raw, seen, existing_dois, existing_oa_ids)


def _search_by_journal(cond: dict, lookback_days: int,
                        seen: set[str], existing_dois: set[str],
                        existing_oa_ids: set[str],
                        max_results: int = 100) -> list[dict]:
    source_id: str = cond.get("value", "")
    if not source_id:
        return []
    # Support pipe-separated multi-journal
    source_ids = [s.strip() for s in source_id.split("|") if s.strip()]
    pipe = "|".join(source_ids)
    extra_filter: dict = cond.get("extra_filter") or {}
    oa_filter = {
        "primary_location.source.id": pipe,
        **_lang_type_filter(cond),
        "from_publication_date": _since_date(lookback_days),
        **extra_filter,
    }
    query = Works().filter(**oa_filter).select(_SELECT_FIELDS).sort(publication_date="desc")
    raw = _paginate(query, max_results)
    return _deduplicate(raw, seen, existing_dois, existing_oa_ids)


def _search_by_topic(cond: dict, lookback_days: int,
                     seen: set[str], existing_dois: set[str],
                     existing_oa_ids: set[str],
                     max_results: int = 100) -> list[dict]:
    topic_id: str = cond.get("value", "")
    if not topic_id:
        return []
    extra_filter: dict = cond.get("extra_filter") or {}
    oa_filter = {
        "topics.id": topic_id,
        **_lang_type_filter(cond),
        "from_publication_date": _since_date(lookback_days),
        **extra_filter,
    }
    query = Works().filter(**oa_filter).select(_SELECT_FIELDS).sort(cited_by_count="desc")
    raw = _paginate(query, max_results)
    return _deduplicate(raw, seen, existing_dois, existing_oa_ids)


def _search_by_field(cond: dict, lookback_days: int,
                     seen: set[str], existing_dois: set[str],
                     existing_oa_ids: set[str],
                     max_results: int = 100) -> list[dict]:
    field_id: str = cond.get("value", "")
    if not field_id:
        return []
    extra_filter: dict = cond.get("extra_filter") or {}
    oa_filter = {
        "topics.field.id": field_id,
        **_lang_type_filter(cond),
        "from_publication_date": _since_date(lookback_days),
        **extra_filter,
    }
    query = Works().filter(**oa_filter).select(_SELECT_FIELDS).sort(cited_by_count="desc")
    raw = _paginate(query, max_results)
    return _deduplicate(raw, seen, existing_dois, existing_oa_ids)


def _search_by_domain(cond: dict, lookback_days: int,
                      seen: set[str], existing_dois: set[str],
                      existing_oa_ids: set[str],
                      max_results: int = 100) -> list[dict]:
    domain_id: str = cond.get("value", "")
    if not domain_id:
        return []
    extra_filter: dict = cond.get("extra_filter") or {}
    oa_filter = {
        "topics.domain.id": domain_id,
        **_lang_type_filter(cond),
        "from_publication_date": _since_date(lookback_days),
        **extra_filter,
    }
    query = Works().filter(**oa_filter).select(_SELECT_FIELDS).sort(cited_by_count="desc")
    raw = _paginate(query, max_results)
    return _deduplicate(raw, seen, existing_dois, existing_oa_ids)


def _search_citing_library(library_oa_ids: list[str], lookback_days: int,
                            seen: set[str], existing_dois: set[str],
                            existing_oa_ids: set[str],
                            max_per_batch: int = 200,
                            max_results: int = 200,
                            field_id: str | None = None,
                            domain_id: str | None = None,
                            language: str = "en",
                            doc_type: str = "article") -> list[dict]:
    """Papers that cite any work currently in the Zotero collection."""
    if not library_oa_ids:
        return []
    all_results: list[dict] = []
    batch_size = 50
    n_batches = 0
    extra: dict = {}
    if field_id:
        extra["topics.field.id"] = field_id
    elif domain_id:
        extra["topics.domain.id"] = domain_id
    for i in range(0, len(library_oa_ids), batch_size):
        if len(all_results) >= max_results * 2:   # buffer for dedup losses
            break
        batch = library_oa_ids[i: i + batch_size]
        pipe = "|".join(batch)
        for attempt in range(4):
            try:
                _lt = {}
                if language and language != "any":
                    _lt["language"] = language
                if doc_type and doc_type != "any":
                    _lt["type"] = doc_type
                query = Works().filter(**{
                    "cites": pipe,
                    "from_publication_date": _since_date(lookback_days),
                    **_lt,
                    **extra,
                }).select(_SELECT_FIELDS).sort(publication_date="desc")
                results = _paginate(query, min(max_per_batch, max_results))
                all_results.extend(results)
                n_batches += 1
                time.sleep(0.5)
                break
            except Exception as exc:
                if "429" in str(exc) or "too many" in str(exc).lower():
                    time.sleep(15 * (2 ** attempt))
                else:
                    break
    out = _deduplicate(all_results, seen, existing_dois, existing_oa_ids)[:max_results]
    return out


def _search_prolific_authors(library_oa_ids: list[str],
                              library_meta: list[dict],
                              lookback_days: int,
                              seen: set[str],
                              existing_dois: set[str],
                              existing_oa_ids: set[str],
                              min_papers: int = 3,
                              max_per_author: int = 15,
                              max_results: int = 200,
                              field_id: str | None = None,
                              domain_id: str | None = None,
                              language: str = "en",
                              doc_type: str = "article") -> list[dict]:
    """Recent papers by authors who have ≥ min_papers in the Zotero collection."""
    # Count author IDs via bulk work fetch
    author_counts: Counter = Counter()
    batch_size = 100
    for i in range(0, len(library_oa_ids), batch_size):
        batch = library_oa_ids[i: i + batch_size]
        pipe = "|".join(batch)
        try:
            pager = Works().filter(**{"ids.openalex": pipe}).select(["id", "authorships"]).paginate(per_page=200, n_max=None)
            for page in pager:
                for work in page:
                    for auth in (work.get("authorships") or []):
                        aid = (auth.get("author") or {}).get("id", "").replace("https://openalex.org/", "")
                        if aid:
                            author_counts[aid] += 1
            time.sleep(0.2)
        except Exception:
            pass

    prolific = [aid for aid, cnt in author_counts.items() if cnt >= min_papers]
    if not prolific:
        return []

    # Cap per-author results so total stays close to max_results
    capped_per_author = max(1, min(max_per_author, math.ceil(max_results / max(len(prolific), 1))))

    extra: dict = {}
    if field_id:
        extra["topics.field.id"] = field_id
    elif domain_id:
        extra["topics.domain.id"] = domain_id

    all_results: list[dict] = []
    for author_id in prolific:
        if len(all_results) >= max_results * 2:
            break
        try:
            _lt = {}
            if language and language != "any":
                _lt["language"] = language
            if doc_type and doc_type != "any":
                _lt["type"] = doc_type
            query = Works().filter(**{
                "author.id": author_id,
                "from_publication_date": _since_date(lookback_days),
                **_lt,
                **extra,
            }).select(_SELECT_FIELDS).sort(publication_date="desc")
            raw = _paginate(query, capped_per_author)
            all_results.extend(raw)
            time.sleep(0.15)
        except Exception:
            pass
    return _deduplicate(all_results, seen, existing_dois, existing_oa_ids)[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# Resolve DOIs to OpenAlex IDs (for citing / authors searches)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_library_oa_ids(library_meta: list[dict], _tag: str = "[ATLAS]") -> list[str]:
    """Return all OpenAlex Work IDs for the library items."""
    known = {m["openalex_id"] for m in library_meta if m.get("openalex_id")}
    resolved: set[str] = set(known)

    dois = [m["doi"] for m in library_meta if m.get("doi") and not m.get("openalex_id")]
    for i in range(0, len(dois), 50):
        batch = dois[i: i + 50]
        doi_filter = "|".join(batch)
        try:
            pager = Works().filter(doi=doi_filter).select(["id", "doi"]).paginate(per_page=200, n_max=None)
            for page in pager:
                for r in page:
                    oa_id = (r.get("id") or "").replace("https://openalex.org/", "")
                    if oa_id:
                        resolved.add(oa_id)
            time.sleep(0.2)
        except Exception as exc:
            print(f"{_tag} resolve OA IDs error: {exc}")

    print(f"{_tag} Library: {len(resolved)} OA IDs resolved ({len(known)} from Zotero, {len(resolved)-len(known)} via DOI lookup)")
    return list(resolved)


# ─────────────────────────────────────────────────────────────────────────────
# Autocomplete helpers (called from Flask routes)
# ─────────────────────────────────────────────────────────────────────────────

def autocomplete_authors(q: str, limit: int = 10) -> list[dict]:
    """Return OpenAlex author suggestions for the query string."""
    if not q or len(q) < 2:
        return []
    import requests
    params: dict = {"q": q}
    email = db.get_setting("openalex_email", "")
    if email:
        params["mailto"] = email
    r = requests.get(
        "https://api.openalex.org/autocomplete/authors",
        params=params,
        timeout=8,
    )
    r.raise_for_status()
    return [
        {
            "id": x.get("id", "").replace("https://openalex.org/", ""),
            "display_name": x.get("display_name", ""),
            "hint": x.get("hint", ""),
            "cited_by_count": x.get("cited_by_count", 0),
        }
        for x in r.json().get("results", [])
    ]


def autocomplete_sources(q: str, limit: int = 10) -> list[dict]:
    """Return OpenAlex journal/source suggestions for the query string."""
    if not q or len(q) < 2:
        return []
    import requests
    params: dict = {"q": q}
    email = db.get_setting("openalex_email", "")
    if email:
        params["mailto"] = email
    r = requests.get(
        "https://api.openalex.org/autocomplete/sources",
        params=params,
        timeout=8,
    )
    r.raise_for_status()
    return [
        {
            "id": x.get("id", "").replace("https://openalex.org/", ""),
            "display_name": x.get("display_name", ""),
            "hint": x.get("hint", ""),
            "issn": x.get("issn", []),
        }
        for x in r.json().get("results", [])
    ]


def autocomplete_topics(q: str, limit: int = 10) -> list[dict]:
    """Return OpenAlex topic suggestions for the query string."""
    if not q or len(q) < 2:
        return []
    import requests
    params: dict = {"q": q}
    email = db.get_setting("openalex_email", "")
    if email:
        params["mailto"] = email
    r = requests.get(
        "https://api.openalex.org/autocomplete/topics",
        params=params,
        timeout=8,
    )
    r.raise_for_status()
    return [
        {
            "id": x.get("id", "").replace("https://openalex.org/", ""),
            "display_name": x.get("display_name", ""),
            "hint": x.get("hint", ""),
        }
        for x in r.json().get("results", [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Library abstract enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_library_abstracts(meta: list[dict], batch_size: int = 50) -> list[dict]:
    """
    For library items that have no abstract, fetch it from OpenAlex using
    the item's DOI or OpenAlex ID.  Items are mutated in-place and returned.
    Works in batches of up to `batch_size` items per OpenAlex request.
    """
    _configure_pyalex()

    # Track how many already had abstracts before we start
    had_abstract_before = sum(1 for m in meta if m.get("abstract"))

    # Separate items that need enrichment, keyed for fast lookup
    by_oa_id: dict[str, dict] = {}
    by_doi:   dict[str, dict] = {}
    for m in meta:
        if m.get("abstract"):
            continue
        if m.get("openalex_id"):
            by_oa_id[m["openalex_id"]] = m
        elif m.get("doi"):
            by_doi[m["doi"]] = m

    def _fill(works: list[dict]) -> None:
        for w in works:
            oa_id = (w.get("id") or "").replace("https://openalex.org/", "")
            doi   = (w.get("doi") or "").replace("https://doi.org/", "").lower().strip()
            abstract = reconstruct_abstract(w)
            if not abstract:
                continue
            if oa_id in by_oa_id:
                by_oa_id[oa_id]["abstract"] = abstract
                # Also set openalex_id in case item was matched by DOI
                by_oa_id[oa_id].setdefault("openalex_id", oa_id)
            if doi in by_doi:
                by_doi[doi]["abstract"] = abstract
                by_doi[doi].setdefault("openalex_id", oa_id)

    # Fetch by OA ID in batches
    oa_ids = list(by_oa_id.keys())
    for i in range(0, len(oa_ids), batch_size):
        batch = oa_ids[i: i + batch_size]
        try:
            works = _get_with_backoff(
                Works().filter(openalex_id="|".join(batch))
                       .select(["id", "doi", "abstract_inverted_index"]),
                per_page=batch_size,
            )
            _fill(works)
        except Exception as exc:
            print(f"[ATLAS] enrich_library_abstracts (oa_id batch): {exc}")

    # Fetch by DOI in batches (only items not already resolved above)
    remaining_dois = [doi for doi, m in by_doi.items() if not m.get("abstract")]
    for i in range(0, len(remaining_dois), batch_size):
        batch = remaining_dois[i: i + batch_size]
        try:
            works = _get_with_backoff(
                Works().filter(doi="|".join(batch))
                       .select(["id", "doi", "abstract_inverted_index"]),
                per_page=batch_size,
            )
            _fill(works)
        except Exception as exc:
            print(f"[ATLAS] enrich_library_abstracts (doi batch): {exc}")

    filled   = sum(1 for m in meta if m.get("abstract"))
    total    = len(meta)
    enriched = filled - had_abstract_before
    print(f"[ATLAS] Abstracts: {filled}/{total} library items have an abstract ({enriched} fetched from OpenAlex)")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: run all conditions for a profile
# ─────────────────────────────────────────────────────────────────────────────

def run_profile_search(
    conditions: list[dict],
    lookback_days: int,
    library_meta: list[dict],
    max_per_condition: int = 150,
    progress_cb=None,
    run_id: int | None = None,
) -> list[dict]:
    """
    Execute all conditions in a profile and return a deduplicated list of
    OpenAlex Work dicts, each annotated with a ``_reasons`` list.

    Args:
        conditions      list of condition dicts from the profile JSON
        lookback_days   how far back to search
        library_meta    slim list from zotero_client.extract_item_metadata()
        max_per_condition  cap per individual condition
        progress_cb     optional callable(pct: int, step: str) for progress reporting
    """
    _configure_pyalex()

    _tag = f"[run {run_id}]" if run_id is not None else "[ATLAS]"

    existing_dois = {m["doi"] for m in library_meta if m.get("doi")}
    existing_oa_ids = {m["openalex_id"] for m in library_meta if m.get("openalex_id")}
    seen: set[str] = set()

    # Pre-resolve library OA IDs once (used by citing / authors handlers)
    library_oa_ids: list[str] | None = None

    # ── Scope: conditions marked scope:true are AND-injected into all others ──
    scope_conds  = [c for c in conditions if c.get("scope")]
    run_conds    = [c for c in conditions if not c.get("scope")]
    scope_ef     = _build_scope_filter(scope_conds)
    # For citing/authors handlers that accept field_id/domain_id directly
    scope_field  = next((sc["value"] for sc in scope_conds if sc.get("type") == "field"), None)
    scope_domain = next((sc["value"] for sc in scope_conds if sc.get("type") == "domain"), None)

    scope_summary = ""
    if scope_conds:
        labels = ", ".join(sc.get("label") or sc.get("type","?") for sc in scope_conds)
        scope_summary = f" | scope: {labels}"

    def _with_scope(cond: dict) -> dict:
        """Return a shallow copy of cond with scope extra_filter merged in."""
        if not scope_ef:
            return cond
        merged = dict(cond)
        merged["extra_filter"] = {**scope_ef, **(cond.get("extra_filter") or {})}
        return merged

    print(f"{_tag} Starting search: {len(run_conds)} condition(s), {lookback_days}d lookback{scope_summary}")

    # We collect (paper, reason_label) tuples; then merge by oa_id
    collected: dict[str, dict] = {}   # oa_id → paper dict with _reasons list
    n_total = max(len(run_conds), 1)
    _cb = progress_cb if callable(progress_cb) else lambda *_: None

    def _annotate_and_store(papers: list[dict], reason_label: str) -> None:
        for p in papers:
            oa_id = (p.get("id") or "").replace("https://openalex.org/", "")
            if not oa_id:
                continue
            p["abstract"] = p.get("abstract") or reconstruct_abstract(p)
            if oa_id in collected:
                if reason_label not in collected[oa_id].get("_reasons", []):
                    collected[oa_id].setdefault("_reasons", []).append(reason_label)
            else:
                p["_reasons"] = [reason_label]
                collected[oa_id] = p

    for cond_idx, cond in enumerate(run_conds):
        ctype: str = cond.get("type", "")
        label: str = cond.get("label") or ctype
        _cb(20 + round(cond_idx / n_total * 50), f"Cond {cond_idx+1}/{n_total}\n{label}")

        papers: list[dict] = []
        try:
            if ctype == "keywords_title_abstract":
                papers = _search_keywords_title_abstract(
                    _with_scope(cond), lookback_days, seen, existing_dois, existing_oa_ids, max_per_condition
                )
                _annotate_and_store(papers, label)

            elif ctype == "keywords_title":
                papers = _search_keywords_title(
                    _with_scope(cond), lookback_days, seen, existing_dois, existing_oa_ids, max_per_condition
                )
                _annotate_and_store(papers, label)

            elif ctype == "author":
                papers = _search_by_author(
                    _with_scope(cond), lookback_days, seen, existing_dois, existing_oa_ids, max_per_condition
                )
                _annotate_and_store(papers, label)

            elif ctype == "journal":
                papers = _search_by_journal(
                    _with_scope(cond), lookback_days, seen, existing_dois, existing_oa_ids, max_per_condition
                )
                _annotate_and_store(papers, label)

            elif ctype == "topic":
                papers = _search_by_topic(
                    _with_scope(cond), lookback_days, seen, existing_dois, existing_oa_ids, max_per_condition
                )
                _annotate_and_store(papers, label)

            elif ctype == "field":
                papers = _search_by_field(
                    _with_scope(cond), lookback_days, seen, existing_dois, existing_oa_ids, max_per_condition
                )
                _annotate_and_store(papers, label)

            elif ctype == "domain":
                papers = _search_by_domain(
                    _with_scope(cond), lookback_days, seen, existing_dois, existing_oa_ids, max_per_condition
                )
                _annotate_and_store(papers, label)

            elif ctype == "citing_library":
                if library_oa_ids is None:
                    library_oa_ids = resolve_library_oa_ids(library_meta, _tag)
                papers = _search_citing_library(
                    library_oa_ids, lookback_days, seen, existing_dois, existing_oa_ids,
                    max_results=max_per_condition,
                    field_id=cond.get("field") or scope_field,
                    domain_id=cond.get("domain") or scope_domain,
                    language=cond.get("language", "en"),
                    doc_type=cond.get("doc_type", "article"),
                )
                _annotate_and_store(papers, label or "cites my library")

            elif ctype == "author_in_library":
                if library_oa_ids is None:
                    library_oa_ids = resolve_library_oa_ids(library_meta, _tag)
                min_papers = int(cond.get("min_papers") or 3)
                papers = _search_prolific_authors(
                    library_oa_ids, library_meta, lookback_days, seen,
                    existing_dois, existing_oa_ids, min_papers=min_papers,
                    max_results=max_per_condition,
                    field_id=cond.get("field") or scope_field,
                    domain_id=cond.get("domain") or scope_domain,
                    language=cond.get("language", "en"),
                    doc_type=cond.get("doc_type", "article"),
                )
                _annotate_and_store(papers, label or f"prolific author (≥{min_papers})")

            elif ctype == "zotero_collection":
                pass

            print(f"{_tag} Cond {cond_idx+1}/{n_total} '{label}': {len(papers)} papers")

        except Exception as exc:
            import traceback
            print(f"{_tag} Cond '{label}' FAILED: {exc}\n{traceback.format_exc()}")
            papers = []

    print(f"{_tag} Search done: {len(collected)} unique candidates")
    _cb(72, f"OpenAlex search complete — {len(collected)} candidates")
    return list(collected.values())
