"""
zotero_client.py – Zotero library integration for ATLAS.

Provides:
  • get_collections()          – full nested tree of collections
  • get_library_items()        – slim metadata list for a collection key
  • extract_item_metadata()    – normalise Zotero item dicts
  • add_papers_to_collection() – push accepted OpenAlex papers to Zotero
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from pyzotero import zotero

import database as db

# ── Connection ────────────────────────────────────────────────────────────────

def _ascii(s: str) -> str:
    """Strip non-ASCII characters (e.g. from rich-text copy-paste) so httpx headers don't crash."""
    return s.encode("ascii", errors="ignore").decode("ascii").strip()


def _zot() -> zotero.Zotero:
    api_key = _ascii(db.get_setting("zotero_api_key", ""))
    library_id = _ascii(db.get_setting("zotero_library_id", ""))
    library_type = db.get_setting("zotero_library_type", "user")
    if not api_key or not library_id:
        raise EnvironmentError("Zotero credentials not configured. Go to Settings.")
    zot = zotero.Zotero(str(library_id), library_type, api_key)
    return zot


# ── Collections tree ──────────────────────────────────────────────────────────

def _build_tree(collections: list[dict]) -> list[dict]:
    """Convert flat list of Zotero collection objects into a nested tree."""
    by_key: dict[str, dict] = {}
    for col in collections:
        key = col["key"]
        by_key[key] = {
            "key": key,
            "name": col["data"]["name"],
            "parent": col["data"].get("parentCollection") or None,
            "children": [],
        }

    roots: list[dict] = []
    for node in by_key.values():
        parent_key = node["parent"]
        if parent_key and parent_key in by_key:
            by_key[parent_key]["children"].append(node)
        else:
            roots.append(node)

    def _sort(nodes: list[dict]) -> None:
        nodes.sort(key=lambda n: n["name"].lower())
        for n in nodes:
            _sort(n["children"])

    _sort(roots)
    return roots


def get_collections() -> list[dict]:
    """Return the full nested collection tree for the configured library."""
    zot = _zot()
    all_cols = zot.collections()
    return _build_tree(all_cols)


def _flatten_tree(tree: list[dict]) -> list[dict]:
    """Flatten nested tree into a list {key, name, path}."""
    out: list[dict] = []

    def _walk(nodes: list[dict], prefix: str = "") -> None:
        for n in nodes:
            path = f"{prefix}{n['name']}"
            out.append({"key": n["key"], "name": n["name"], "path": path})
            _walk(n["children"], f"{path} / ")

    _walk(tree)
    return out


def get_flat_collections() -> list[dict]:
    return _flatten_tree(get_collections())


# ── Library items ─────────────────────────────────────────────────────────────

def get_library_items(collection_key: str, use_cache: bool = True) -> list[dict]:
    """
    Fetch all journalArticle items from the given collection key.
    Caches to disk for 6 hours.
    """
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"library_{collection_key}.json"

    if use_cache and cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < 6:
            return json.loads(cache_path.read_text())

    zot = _zot()
    items: list[dict] = []
    start = 0
    while True:
        chunk = zot.collection_items(
            collection_key, itemType="journalArticle", start=start, limit=100
        )
        if not chunk:
            break
        items.extend(chunk)
        start += len(chunk)
        if len(chunk) < 100:
            break
        time.sleep(0.3)

    cache_path.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    return items


def extract_item_metadata(items: list[dict]) -> list[dict]:
    """
    Return slim dicts from raw Zotero items:
      title, abstract, doi, openalex_id, authors, year, zotero_key
    """
    out = []
    for item in items:
        d = item.get("data", {})
        doi = re.sub(r"^https?://doi\.org/", "", (d.get("DOI") or "").strip(), flags=re.IGNORECASE).lower()
        extra: str = d.get("extra", "")
        oa_match = re.search(r"OpenAlex:\s*(W\d+)", extra, re.IGNORECASE)
        oa_id = oa_match.group(1) if oa_match else None

        authors = []
        for c in d.get("creators", []):
            if c.get("creatorType") != "author":
                continue
            first = c.get("firstName", "").strip()
            last = c.get("lastName", "").strip()
            name = c.get("name", "").strip()
            if first and last:
                authors.append(f"{first} {last}")
            elif last:
                authors.append(last)
            elif name:
                authors.append(name)

        out.append({
            "title": d.get("title", ""),
            "abstract": d.get("abstractNote", ""),
            "doi": doi,
            "openalex_id": oa_id,
            "authors": authors,
            "year": (d.get("date") or "")[:4],
            "zotero_key": item.get("key"),
        })
    return out


# ── Write operations ──────────────────────────────────────────────────────────

def _paper_to_zotero_item(paper: dict, collection_key: str,
                           extra_tag: str | None = None,
                           inbox_tag: str = "atlas-inbox") -> dict:
    """Convert an OpenAlex Work dict to a Zotero journalArticle payload."""
    title: str = paper.get("title") or ""
    doi: str = (paper.get("doi") or "").replace("https://doi.org/", "")
    year = paper.get("publication_year") or paper.get("year")
    venue: str = ""
    if paper.get("primary_location"):
        src = (paper["primary_location"] or {}).get("source") or {}
        venue = src.get("display_name", "")

    abstract: str = paper.get("abstract") or ""

    authors_zotero = []
    for authorship in (paper.get("authorships") or []):
        author = authorship.get("author", {})
        raw_name: str = (author or {}).get("display_name", "")
        parts = raw_name.rsplit(" ", 1)
        if len(parts) == 2:
            first, last = parts
        else:
            first, last = "", raw_name
        authors_zotero.append(
            {"creatorType": "author", "firstName": first, "lastName": last}
        )

    oa_id: str = (paper.get("id") or "").replace("https://openalex.org/", "")
    score: float | None = paper.get("_combined_score")
    reasons: list[str] = paper.get("_reasons", [])

    extra_lines = []
    if oa_id:
        extra_lines.append(f"OpenAlex: {oa_id}")
    if score is not None:
        extra_lines.append(f"ATLAS score: {score:.3f}")
    if reasons:
        extra_lines.append(f"ATLAS reason: {'; '.join(reasons)}")
    extra = "\n".join(extra_lines)

    tags = [
        {"tag": "atlas-import"},
        {"tag": inbox_tag},
    ]
    if extra_tag:
        tags.append({"tag": extra_tag})
    for reason in reasons:
        short = reason[:50].strip()
        if short:
            tags.append({"tag": f"atlas:{short}"})

    return {
        "itemType": "journalArticle",
        "title": title,
        "abstractNote": abstract,
        "publicationTitle": venue,
        "date": str(year) if year else "",
        "DOI": doi,
        "url": paper.get("doi") or (f"https://openalex.org/{oa_id}" if oa_id else ""),
        "creators": authors_zotero,
        "extra": extra,
        "collections": [collection_key],
        "tags": tags,
    }


def _find_or_create_subcollection(zot: zotero.Zotero, parent_key: str, child_name: str) -> str:
    for sub in zot.collections_sub(parent_key):
        if sub["data"]["name"].lower() == child_name.lower():
            return sub["key"]
    for attempt in range(3):
        try:
            result = zot.create_collections([{"name": child_name, "parentCollection": parent_key}])
            return list(result.get("success", {}).values())[0]
        except Exception as exc:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not create sub-collection '{child_name}'")


def add_papers_to_collection(
    papers: list[dict],
    target_collection_key: str,
    inbox_subcollection: str | None = None,
    extra_tag: str | None = None,
) -> list[str]:
    """
    Push a list of OpenAlex papers to Zotero.

    Returns list of created Zotero item keys.
    """
    if not papers:
        return []

    zot = _zot()

    # Resolve target collection (optionally nested)
    dest_key = target_collection_key
    if inbox_subcollection:
        dest_key = _find_or_create_subcollection(zot, target_collection_key, inbox_subcollection)

    created_keys: list[str] = []
    for i in range(0, len(papers), 50):
        batch = papers[i: i + 50]
        payload = [_paper_to_zotero_item(p, dest_key, extra_tag=extra_tag) for p in batch]
        resp = zot.create_items(payload)
        for key in resp.get("success", {}).values():
            created_keys.append(key)
        time.sleep(0.5)

    return created_keys


# ── Cache invalidation ────────────────────────────────────────────────────────

def invalidate_cache(collection_key: str | None = None) -> None:
    cache_dir = Path(__file__).parent / "cache"
    if collection_key:
        p = cache_dir / f"library_{collection_key}.json"
        if p.exists():
            p.unlink()
    else:
        for p in cache_dir.glob("library_*.json"):
            p.unlink()
