"""
app.py – ATLAS Flask application.

Run with:
    conda run -n Shiny_app1_screening_sobriety python app.py
    (or via run.sh)

Listens on http://localhost:5050 by default.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime, date, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template, request, abort, send_from_directory

import database as db
import openalex_client as oac
import zotero_client as zc
import embeddings as emb

# ── App init ──────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False

db.init_db()

# Suppress per-request access logs ("GET /api/... 200" lines)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Bind the DB user to the Unix account running the process.
# Each user on the server runs their own `python app.py`, so $USER is unique.
_SERVER_USER = re.sub(r"[^a-zA-Z0-9_-]", "_", os.getenv("USER", "default"))[:32] or "default"
db.set_current_user(_SERVER_USER)
print(f"[ATLAS] running as user: {_SERVER_USER}")

# In-progress background runs: run_id → thread
_running: dict[int, threading.Thread] = {}

# Cross-run deduplication (within a simultaneous multi-profile launch group)
_group_seen:     dict[str, set[str]]       = {}
_group_locks:    dict[str, threading.Lock] = {}
_group_lock_meta = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# User context (per-request and per-thread)
# ─────────────────────────────────────────────────────────────────────────────

@app.before_request
def _set_user_context():
    """Re-bind DB user on every request thread (threads don't inherit _local)."""
    db.set_current_user(_SERVER_USER)


@app.route("/api/whoami", methods=["GET"])
def api_whoami():
    return _ok(user=_SERVER_USER)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(data=None, **kwargs):
    """Return a JSON response.  Lists are returned as a bare JSON array;
    dicts are spread into {ok:true, ...}; kwargs are merged in."""
    if isinstance(data, list):
        if kwargs:
            # rare case: list + extra fields → wrap under 'results'
            return jsonify({"ok": True, "results": data, **kwargs})
        return jsonify(data)
    payload = {"ok": True}
    if isinstance(data, dict):
        payload.update(data)
    payload.update(kwargs)
    return jsonify(payload)


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def _resolve_lookback(timeframe: str, profile_id: int | None) -> int:
    """Convert a timeframe string to a concrete number of days."""
    MAP = {
        "last_week": 7,
        "last_2_weeks": 14,
        "last_month": 30,
        "last_2_months": 60,
        "last_year": 365,
    }
    if timeframe in MAP:
        return MAP[timeframe]
    if timeframe == "since_last_run" and profile_id:
        last = db.get_last_run_date_for_profile(profile_id)
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                delta = (datetime.utcnow() - last_dt).days + 1
                return max(1, delta)
            except ValueError:
                pass
    return 30  # default: one month


# ─────────────────────────────────────────────────────────────────────────────
# Group deduplication helper
# ─────────────────────────────────────────────────────────────────────────────

def _claim_for_group(group_id: str, ranked: list) -> list:
    """Atomically claim paper IDs for this group; drops papers already claimed
    by another worker in the same simultaneous run batch."""
    with _group_lock_meta:
        if group_id not in _group_locks:
            _group_locks[group_id] = threading.Lock()
            _group_seen[group_id]  = set()
    with _group_locks[group_id]:
        seen   = _group_seen[group_id]
        unique = []
        for item in ranked:
            _, _, _, paper = item
            oa_id = (paper.get("id") or "").replace("https://openalex.org/", "")
            if oa_id and oa_id not in seen:
                seen.add(oa_id)
                unique.append(item)
        return unique


# ─────────────────────────────────────────────────────────────────────────────
# Background search worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_search_worker(run_id: int, profile: dict, lookback_days: int,
                        collection_key: str, weight_library: float, weight_feedback: float,
                        user_id: str = "default",
                        group_id: str | None = None) -> None:
    """Runs in a background thread; writes results to DB."""
    db.set_current_user(user_id)   # bind DB to the requesting user

    def _progress(pct: int, step: str) -> None:
        db.update_run(run_id, progress_pct=pct, progress_step=step)

    try:
        _progress(3, "1/5\nLoading Zotero library")
        db.update_run(run_id, lookback_days=lookback_days)

        # ── 1. Load Zotero library ────────────────────────────────────────────
        raw_items = zc.get_library_items(collection_key)
        library_meta = zc.extract_item_metadata(raw_items)
        library_texts = [
            f"{m['title']}. {m['abstract']}".strip()
            for m in library_meta
            if m.get("title") or m.get("abstract")
        ]

        _progress(15, "2/5\nSearching OpenAlex")
        # ── 2. Fetch candidates via OpenAlex ──────────────────────────────────
        conditions: list[dict] = profile.get("conditions", [])
        max_per = db.get_setting("max_per_condition", 30)
        candidates = oac.run_profile_search(
            conditions, lookback_days, library_meta,
            max_per_condition=max_per,
            progress_cb=_progress,
        )

        db.update_run(run_id, n_candidates=len(candidates))

        if not candidates:
            db.update_run(run_id, status="done", n_results=0,
                          progress_pct=100, progress_step="Done — no new papers found",
                          completed_at=datetime.utcnow().isoformat())
            return

        _progress(75, f"3/5\nCaching {len(candidates)} papers")
        # ── 3. Cache paper metadata ───────────────────────────────────────────
        for p in candidates:
            oa_id = (p.get("id") or "").replace("https://openalex.org/", "")
            if not oa_id:
                continue
            doi = (p.get("doi") or "").replace("https://doi.org/", "").lower().strip()
            title = p.get("title") or ""
            abstract = p.get("abstract") or oac.reconstruct_abstract(p)
            authors = [
                (a.get("author") or {}).get("display_name", "")
                for a in (p.get("authorships") or [])
            ]
            year = p.get("publication_year")
            venue = ""
            if p.get("primary_location"):
                src = (p["primary_location"] or {}).get("source") or {}
                venue = src.get("display_name", "")
            cited = p.get("cited_by_count", 0) or 0
            topics = [
                (t.get("display_name") or "") for t in (p.get("topics") or [])
            ]
            db.upsert_paper(oa_id, doi, title, abstract, authors, year, venue, cited, topics, p)

        _progress(85, "4/5\nComputing embeddings")
        # ── 4. Rank by similarity ──────────────────────────────────────────────
        threshold = float(db.get_setting("similarity_threshold", 0.0))
        n_cand    = len(candidates)

        def _emb_progress(cur_batch: int, tot_batches: int) -> None:
            pct = 85 + round(cur_batch / max(tot_batches, 1) * 9)
            _progress(pct, f"4/5\nEncoding {cur_batch}/{tot_batches} batches ({n_cand} papers)")

        ranked = emb.rank_candidates(
            library_texts, candidates,
            weight_library=weight_library,
            weight_feedback=weight_feedback,
            threshold=threshold,
            collection_key=collection_key,
            embed_progress_cb=_emb_progress,
        )

        # ── 4b. Cross-run deduplication (within this launch group) ─────────────────
        if group_id and ranked:
            before = len(ranked)
            ranked = _claim_for_group(group_id, ranked)
            dropped = before - len(ranked)
            if dropped:
                print(f"[ATLAS] run {run_id}: dropped {dropped} papers already in group")

        _progress(95, f"5/5\nSaving {len(ranked)} results")
        # ── 5. Write results ──────────────────────────────────────────────────
        for combined, sl, sf, paper in ranked:
            oa_id = (paper.get("id") or "").replace("https://openalex.org/", "")
            reasons = paper.get("_reasons", [])
            db.add_run_result(run_id, oa_id, sl, sf, combined, reasons)

        db.update_run(run_id, status="done", n_results=len(ranked),
                      progress_pct=100, progress_step=f"Done\n{len(ranked)} papers found",
                      completed_at=datetime.utcnow().isoformat())

    except Exception as exc:
        db.update_run(run_id, status="error",
                      error_msg=str(exc)[:500],
                      progress_step="Error: " + str(exc)[:80],
                      completed_at=datetime.utcnow().isoformat())
        traceback.print_exc()
    finally:
        _running.pop(run_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Routes – SPA entry
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "star.ico", mimetype="image/x-icon")


@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    settings = db.get_all_settings()
    # Mask secrets for display
    masked = dict(settings)
    if masked.get("zotero_api_key"):
        k: str = masked["zotero_api_key"]
        if "•" in k:
            # Stored value is a previously-masked placeholder — clear it so user re-enters the real key
            masked["zotero_api_key"] = ""
        else:
            masked["zotero_api_key"] = k[:4] + "••••••" + k[-4:] if len(k) > 8 else "••••••"
    return _ok(masked)


@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    data: dict = request.get_json(force=True) or {}
    allowed = {
        "zotero_api_key", "zotero_library_id", "zotero_library_type",
        "openalex_email", "similarity_threshold", "max_per_condition",
        "default_collection_key", "default_inbox_subcollection",
        "default_tag", "weight_arbitration",
        # legacy keys kept for backward compatibility
        "weight_library", "weight_feedback",
    }
    saved = []
    for k, v in data.items():
        if k in allowed:
            # Never save a masked API key (contains the • placeholder character)
            if k == "zotero_api_key" and "•" in str(v):
                continue
            db.set_setting(k, v)
            saved.append(k)
    # Invalidate centroid caches when library settings change
    if "default_collection_key" in saved:
        emb.invalidate_library_centroid(data.get("default_collection_key", "default"))
    return _ok(saved=saved)


# ─────────────────────────────────────────────────────────────────────────────
# Zotero collections
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/zotero/collections", methods=["GET"])
def api_zotero_collections():
    try:
        tree = zc.get_collections()
        return _ok(tree)
    except EnvironmentError as exc:
        return _err(str(exc), 400)
    except Exception as exc:
        return _err(f"Zotero error: {exc}", 500)


@app.route("/api/zotero/collections/flat", methods=["GET"])
def api_zotero_collections_flat():
    try:
        flat = zc.get_flat_collections()
        return _ok(flat)
    except EnvironmentError as exc:
        return _err(str(exc), 400)
    except Exception as exc:
        return _err(f"Zotero error: {exc}", 500)


# ─────────────────────────────────────────────────────────────────────────────
# Search profiles
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/profiles", methods=["GET"])
def api_list_profiles():
    return _ok(db.list_profiles())


@app.route("/api/profiles", methods=["POST"])
def api_create_profile():
    data: dict = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return _err("name is required")
    pid = db.create_profile(
        name=name,
        description=data.get("description", ""),
        conditions=data.get("conditions", []),
        combinator=data.get("combinator", "OR"),
    )
    return _ok(db.get_profile(pid))


@app.route("/api/profiles/<int:pid>", methods=["GET"])
def api_get_profile(pid: int):
    profile = db.get_profile(pid)
    if not profile:
        return _err("Profile not found", 404)
    return _ok(profile)


@app.route("/api/profiles/<int:pid>", methods=["PUT"])
def api_update_profile(pid: int):
    data: dict = request.get_json(force=True) or {}
    if not db.get_profile(pid):
        return _err("Profile not found", 404)
    db.update_profile(pid, **data)
    return _ok(db.get_profile(pid))


@app.route("/api/profiles/<int:pid>", methods=["DELETE"])
def api_delete_profile(pid: int):
    db.delete_profile(pid)
    return _ok()


# ─────────────────────────────────────────────────────────────────────────────
# Search runs
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/search/run", methods=["POST"])
def api_start_run():
    """Start one or more profile searches simultaneously.
    Accepts either profile_id (int) or profile_ids (list[int]).
    Uses weight_arbitration (0=library, 1=feedback) to split the scoring weight.
    Returns a JSON array of {run_id, profile_id, profile_name, lookback_days}.
    """
    data: dict = request.get_json(force=True) or {}

    # Support both single profile_id and list of profile_ids
    raw_ids = data.get("profile_ids") or ()
    if not raw_ids and data.get("profile_id"):
        raw_ids = [data["profile_id"]]
    profile_ids = [int(x) for x in raw_ids]

    if not profile_ids:
        return _err("profile_ids is required", 400)

    timeframe: str = data.get("timeframe", "since_last_run")
    collection_key = data.get("collection_key") or db.get_setting("default_collection_key", "")
    if not collection_key:
        return _err("No collection configured – set a working collection in Settings", 400)

    # weight_arbitration: 0.0 = all library, 1.0 = all feedback
    raw_arb = data.get("weight_arbitration")
    if raw_arb is None:
        raw_arb = db.get_setting("weight_arbitration", 0.3)
    weight_arbitration = max(0.0, min(1.0, float(raw_arb)))
    weight_library  = 1.0 - weight_arbitration
    weight_feedback = weight_arbitration

    group_id     = str(uuid.uuid4())   # shared ID for all runs in this batch
    current_user = db.get_current_user()
    started: list[dict] = []
    for pid in profile_ids:
        profile = db.get_profile(pid)
        if not profile:
            continue
        lookback_days = _resolve_lookback(timeframe, pid)
        run_id = db.create_run(
            profile_id=pid,
            profile_name=profile["name"],
            timeframe=timeframe,
            weight_library=weight_library,
            weight_feedback=weight_feedback,
            group_id=group_id,
        )
        t = threading.Thread(
            target=_run_search_worker,
            args=(run_id, profile, lookback_days, collection_key, weight_library, weight_feedback,
                  current_user, group_id),
            daemon=True,
        )
        _running[run_id] = t
        t.start()
        started.append({"run_id": run_id, "profile_id": pid,
                         "profile_name": profile["name"], "lookback_days": lookback_days,
                         "group_id": group_id})

    if not started:
        return _err("No valid profiles found", 404)

    return _ok(started)  # list → returned as bare JSON array


@app.route("/api/search/run/<int:run_id>/status", methods=["GET"])
def api_run_status(run_id: int):
    run = db.get_run(run_id)
    if not run:
        return _err("Run not found", 404)
    run["is_running"] = run_id in _running
    run.setdefault("progress_pct", 0)
    run.setdefault("progress_step", "")
    return _ok(run)


@app.route("/api/search/run/<int:run_id>/results", methods=["GET"])
def api_run_results(run_id: int):
    run = db.get_run(run_id)
    if not run:
        return _err("Run not found", 404)
    results = db.get_run_results(run_id)
    return _ok(results)  # list → returned as bare JSON array


@app.route("/api/search/run/<int:run_id>", methods=["DELETE"])
def api_delete_run(run_id: int):
    """Remove a run from history, clear its paper_decisions (training memory),
    and allow its papers to resurface in future searches.
    """
    if not db.get_run(run_id):
        return _err("Run not found", 404)
    db.delete_run(run_id)
    return _ok(deleted=run_id)


@app.route("/api/history", methods=["GET"])
def api_history():
    limit = int(request.args.get("limit", 50))
    runs = db.list_runs(limit)
    return _ok(runs)


# ─────────────────────────────────────────────────────────────────────────────
# Paper decisions
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/results/<int:result_id>/decide", methods=["POST"])
def api_decide(result_id: int):
    data: dict = request.get_json(force=True) or {}
    decision: str = data.get("decision", "").lower()
    if decision not in ("accepted", "rejected", "skipped"):
        return _err("decision must be accepted, rejected or skipped")

    db.decide_run_result(result_id, decision)

    zotero_keys: list[str] = []
    if decision in ("accepted", "rejected"):
        conn = db.get_connection()
        row = conn.execute(
            "SELECT run_id, paper_oa_id FROM run_results WHERE id=?", (result_id,)
        ).fetchone()
        if row:
            oa_id = row["paper_oa_id"]
            run_id = row["run_id"]
            paper = db.get_paper(oa_id)
            title = paper.get("title") if paper else None
            abstract = paper.get("abstract") if paper else None
            db.record_decision(oa_id, decision, run_id, title, abstract)
            _cache = Path(__file__).parent / "cache"
            for _f in ("centroid_accepted.npy", "centroid_rejected.npy"):
                (_cache / _f).unlink(missing_ok=True)

            # Optionally push to Zotero
            if data.get("add_to_zotero") and decision == "accepted" and paper:
                try:
                    raw = dict(paper.get("raw_data") or {})
                    rr = conn.execute(
                        "SELECT combined_score, match_reasons FROM run_results WHERE id=?",
                        (result_id,)
                    ).fetchone()
                    if rr:
                        raw["_combined_score"] = rr["combined_score"]
                        try:
                            raw["_reasons"] = json.loads(rr["match_reasons"] or "[]")
                        except Exception:
                            raw["_reasons"] = []
                    col_key = data.get("collection_key") or db.get_setting("default_collection_key", "")
                    inbox = data.get("inbox_subcollection") or db.get_setting("default_inbox_subcollection", "") or None
                    extra_tag = data.get("tag") or db.get_setting("default_tag") or None
                    zotero_keys = zc.add_papers_to_collection([raw], col_key, inbox, extra_tag)
                except Exception as exc:
                    return _ok(zotero_added=False, zotero_error=str(exc))

    return _ok(zotero_added=bool(zotero_keys), zotero_keys=zotero_keys)


@app.route("/api/results/batch_decide", methods=["POST"])
def api_batch_decide():
    """Accept/reject/skip a list of result IDs at once + optionally push to Zotero."""
    data: dict = request.get_json(force=True) or {}
    result_ids: list[int] = data.get("result_ids", [])
    decision: str = data.get("decision", "").lower()
    push_to_zotero: bool = bool(data.get("push_to_zotero") or data.get("add_to_zotero", False))

    if decision not in ("accepted", "rejected", "skipped"):
        return _err("decision must be accepted, rejected or skipped")

    conn = db.get_connection()
    papers_to_push: list[dict] = []

    for rid in result_ids:
        db.decide_run_result(rid, decision)
        if decision in ("accepted", "rejected"):
            row = conn.execute(
                "SELECT run_id, paper_oa_id FROM run_results WHERE id=?", (rid,)
            ).fetchone()
            if row:
                oa_id = row["paper_oa_id"]
                run_id = row["run_id"]
                paper = db.get_paper(oa_id)
                title = paper.get("title") if paper else None
                abstract = paper.get("abstract") if paper else None
                db.record_decision(oa_id, decision, run_id, title, abstract)
                if push_to_zotero and decision == "accepted" and paper:
                    raw = dict(paper.get("raw_data") or {})
                    rr = conn.execute(
                        "SELECT combined_score, match_reasons FROM run_results WHERE id=?", (rid,)
                    ).fetchone()
                    if rr:
                        raw["_combined_score"] = rr["combined_score"]
                        try:
                            raw["_reasons"] = json.loads(rr["match_reasons"] or "[]")
                        except Exception:
                            raw["_reasons"] = []
                    papers_to_push.append(raw)

    _cache = Path(__file__).parent / "cache"
    for _f in ("centroid_accepted.npy", "centroid_rejected.npy"):
        (_cache / _f).unlink(missing_ok=True)

    zotero_keys: list[str] = []
    if papers_to_push:
        try:
            col_key = data.get("collection_key") or db.get_setting("default_collection_key", "")
            inbox = data.get("inbox_subcollection") or db.get_setting("default_inbox_subcollection", "") or None
            extra_tag = data.get("extra_tag") or data.get("tag") or db.get_setting("default_tag") or None
            zotero_keys = zc.add_papers_to_collection(papers_to_push, col_key, inbox, extra_tag)
        except Exception as exc:
            return _ok(decided=len(result_ids), zotero_keys=[], zotero_error=str(exc))

    return _ok(decided=len(result_ids), zotero_keys=zotero_keys)


# Add single paper to Zotero a posteriori (from History panel)
@app.route("/api/papers/<string:oa_id>/add_to_zotero", methods=["POST"])
def api_add_to_zotero(oa_id: str):
    data: dict = request.get_json(force=True) or {}
    paper = db.get_paper(oa_id)
    if not paper:
        return _err("Paper not found in local cache", 404)

    raw = paper.get("raw_data") or {}
    raw["_combined_score"] = data.get("score")
    raw["_reasons"] = data.get("reasons", [])

    col_key = data.get("collection_key") or db.get_setting("default_collection_key", "")
    inbox = data.get("inbox_subcollection") or db.get_setting("default_inbox_subcollection", "") or None
    extra_tag = data.get("extra_tag") or db.get_setting("default_tag") or None

    try:
        keys = zc.add_papers_to_collection([raw], col_key, inbox, extra_tag)
        # Persist as accepted decision
        db.record_decision(oa_id, "accepted", None, paper.get("title"), paper.get("abstract"), 
                           zotero_key=keys[0] if keys else None)
        return _ok(zotero_keys=keys)
    except Exception as exc:
        return _err(str(exc), 500)


# ─────────────────────────────────────────────────────────────────────────────
# Autocomplete
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/autocomplete/authors")
def api_autocomplete_authors():
    q = request.args.get("q", "")
    try:
        return jsonify({"results": oac.autocomplete_authors(q)})
    except Exception as exc:
        return jsonify({"results": [], "error": str(exc)})


@app.route("/api/autocomplete/sources")
def api_autocomplete_sources():
    q = request.args.get("q", "")
    try:
        return jsonify({"results": oac.autocomplete_sources(q)})
    except Exception as exc:
        return jsonify({"results": [], "error": str(exc)})


@app.route("/api/autocomplete/topics")
def api_autocomplete_topics():
    q = request.args.get("q", "")
    try:
        return jsonify({"results": oac.autocomplete_topics(q)})
    except Exception as exc:
        return jsonify({"results": [], "error": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/cache/invalidate", methods=["POST"])
def api_invalidate_cache():
    data: dict = request.get_json(force=True) or {}
    col_key = data.get("collection_key")
    zc.invalidate_cache(col_key)
    emb.invalidate_library_centroid(col_key or "default")
    return _ok()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = int(os.getenv("ATLAS_PORT", "5050"))


def _open_browser():
    import time as _t
    _t.sleep(1.2)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║   ATLAS - Literature Discovery App   ║")
    print(f"  ║   http://{HOST}:{PORT}              ║")
    print(f"  ╚══════════════════════════════════════╝\n")
    t = threading.Thread(target=_open_browser, daemon=True)
    t.start()
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)
