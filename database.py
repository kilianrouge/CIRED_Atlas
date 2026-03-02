"""
database.py – SQLite schema and all persistence helpers for ATLAS.

Tables
------
settings            key/value store (Zotero creds, OpenAlex email, UI prefs)
search_profiles     saved search configurations (conditions as JSON)
search_runs         historical search executions
papers              cached OpenAlex paper metadata
run_results         papers surfaced in a run + their scores / decision
paper_decisions     persistent per-paper user decisions (accept/reject) used
                    for feedback-learning across runs
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Per-user database paths ───────────────────────────────────────────────────
_LEGACY_DB  = Path(__file__).parent / "atlas.db"
_DATA_DIR   = Path(__file__).parent / "data" / "users"
_MIGRATIONS = [
    ("progress_pct",  "INTEGER DEFAULT 0"),
    ("progress_step", "TEXT    DEFAULT ''"),
    ("group_id",      "TEXT"),
]

_local = threading.local()
_default_user: str = "default"  # set at startup; used by threads with no explicit bind


def get_db_path(user_id: str) -> Path:
    """Return the SQLite file path for a given user (name sanitised)."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:32] or "default"
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR / f"{safe}.db"


def set_current_user(user_id: str) -> None:
    """Bind the current thread (and the module-level fallback) to a user's database."""
    global _default_user
    _default_user = user_id
    _local.user_id = user_id


def get_current_user() -> str:
    return getattr(_local, "user_id", _default_user)


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and apply migrations on a freshly opened connection."""
    conn.executescript(SCHEMA)
    for col, defn in _MIGRATIONS:
        try:
            conn.execute(f"ALTER TABLE search_runs ADD COLUMN {col} {defn}")
            conn.commit()
        except Exception:
            pass  # column already exists


# ─────────────────────────────────────────────────────────────────────────────
# Connection management
# ─────────────────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a per-thread, per-user SQLite connection (created on first access)."""
    user_id = get_current_user()
    attr = f"conn_{user_id}"
    conn = getattr(_local, attr, None)
    if conn is None:
        db_path = get_db_path(user_id)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(conn)
        setattr(_local, attr, conn)
    return conn


@contextmanager
def transaction():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Schema creation
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS search_profiles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    description TEXT,
    conditions  TEXT NOT NULL DEFAULT '[]',   -- JSON array of condition objects
    combinator  TEXT NOT NULL DEFAULT 'OR',   -- 'AND' | 'OR'
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id      INTEGER REFERENCES search_profiles(id) ON DELETE SET NULL,
    profile_name    TEXT,                      -- snapshot at run time
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    timeframe       TEXT NOT NULL DEFAULT 'since_last_run',
    lookback_days   INTEGER,                   -- resolved value used
    status          TEXT NOT NULL DEFAULT 'running',  -- running | done | error
    error_msg       TEXT,
    n_candidates    INTEGER DEFAULT 0,
    n_results       INTEGER DEFAULT 0,
    weight_library  REAL DEFAULT 0.7,
    weight_feedback REAL DEFAULT 0.3
);

CREATE TABLE IF NOT EXISTS papers (
    oa_id       TEXT PRIMARY KEY,              -- e.g. W2123456789
    doi         TEXT,
    title       TEXT,
    abstract    TEXT,
    authors     TEXT,                          -- JSON list of strings
    year        INTEGER,
    venue       TEXT,
    cited_by_count INTEGER DEFAULT 0,
    topics      TEXT,                          -- JSON list
    raw_data    TEXT,                          -- full OpenAlex JSON
    fetched_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS papers_doi ON papers(doi);

CREATE TABLE IF NOT EXISTS run_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES search_runs(id) ON DELETE CASCADE,
    paper_oa_id     TEXT NOT NULL,
    score_library   REAL,
    score_feedback  REAL,
    combined_score  REAL,
    match_reasons   TEXT,                      -- JSON list of strings
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | rejected | skipped
    decided_at      TEXT,
    UNIQUE(run_id, paper_oa_id)
);

CREATE INDEX IF NOT EXISTS run_results_run ON run_results(run_id);
CREATE INDEX IF NOT EXISTS run_results_paper ON run_results(paper_oa_id);
CREATE INDEX IF NOT EXISTS run_results_status ON run_results(status);

CREATE TABLE IF NOT EXISTS paper_decisions (
    paper_oa_id  TEXT PRIMARY KEY,
    decision     TEXT NOT NULL,               -- 'accepted' | 'rejected'
    decided_at   TEXT NOT NULL,
    run_id       INTEGER,
    zotero_key   TEXT,
    notes        TEXT,
    title        TEXT,                         -- snapshot for display
    abstract     TEXT                          -- snapshot for embedding
);
"""


def init_db() -> None:
    """Initialise the DB for the current user.
    On first run, migrates legacy atlas.db → data/users/default.db if present.
    """
    if _LEGACY_DB.exists() and not get_db_path("default").exists():
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(_LEGACY_DB), str(get_db_path("default")))
    get_connection()  # opens (or creates) the user's DB and runs _init_schema


# ─────────────────────────────────────────────────────────────────────────────
# Settings helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_setting(key: str, default: Any = None) -> Any:
    row = get_connection().execute(
        "SELECT value FROM settings WHERE key=?", (key,)
    ).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return row["value"]


def set_setting(key: str, value: Any) -> None:
    with transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)",
            (key, json.dumps(value)),
        )


def get_all_settings() -> dict:
    rows = get_connection().execute("SELECT key, value FROM settings").fetchall()
    out: dict = {}
    for row in rows:
        try:
            out[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            out[row["key"]] = row["value"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Search profiles
# ─────────────────────────────────────────────────────────────────────────────

def list_profiles() -> list[dict]:
    rows = get_connection().execute(
        "SELECT * FROM search_profiles ORDER BY updated_at DESC"
    ).fetchall()
    return [_profile_row_to_dict(r) for r in rows]


def get_profile(profile_id: int) -> dict | None:
    row = get_connection().execute(
        "SELECT * FROM search_profiles WHERE id=?", (profile_id,)
    ).fetchone()
    return _profile_row_to_dict(row) if row else None


def create_profile(name: str, description: str, conditions: list, combinator: str = "OR") -> int:
    now = datetime.utcnow().isoformat()
    with transaction() as conn:
        cur = conn.execute(
            """INSERT INTO search_profiles(name,description,conditions,combinator,created_at,updated_at)
               VALUES(?,?,?,?,?,?)""",
            (name, description, json.dumps(conditions), combinator, now, now),
        )
    return cur.lastrowid


def update_profile(profile_id: int, **kwargs) -> None:
    allowed = {"name", "description", "conditions", "combinator"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    updates["updated_at"] = datetime.utcnow().isoformat()
    if "conditions" in updates:
        updates["conditions"] = json.dumps(updates["conditions"])
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [profile_id]
    with transaction() as conn:
        conn.execute(f"UPDATE search_profiles SET {sets} WHERE id=?", vals)


def delete_profile(profile_id: int) -> None:
    with transaction() as conn:
        conn.execute("DELETE FROM search_profiles WHERE id=?", (profile_id,))


def _profile_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["conditions"] = json.loads(d.get("conditions") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["conditions"] = []
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Search runs
# ─────────────────────────────────────────────────────────────────────────────

def create_run(profile_id: int | None, profile_name: str, timeframe: str,
               weight_library: float = 0.7, weight_feedback: float = 0.3,
               group_id: str | None = None) -> int:
    now = datetime.utcnow().isoformat()
    with transaction() as conn:
        cur = conn.execute(
            """INSERT INTO search_runs
               (profile_id,profile_name,started_at,timeframe,status,weight_library,weight_feedback,group_id)
               VALUES(?,?,?,?,?,?,?,?)""",
            (profile_id, profile_name, now, timeframe, "running", weight_library, weight_feedback, group_id),
        )
    return cur.lastrowid


def update_run(run_id: int, **kwargs) -> None:
    allowed = {"completed_at", "status", "error_msg", "n_candidates", "n_results",
               "lookback_days", "progress_pct", "progress_step"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [run_id]
    with transaction() as conn:
        conn.execute(f"UPDATE search_runs SET {sets} WHERE id=?", vals)


def get_run(run_id: int) -> dict | None:
    row = get_connection().execute(
        "SELECT * FROM search_runs WHERE id=?", (run_id,)
    ).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 50) -> list[dict]:
    rows = get_connection().execute(
        "SELECT * FROM search_runs ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_run(run_id: int) -> None:
    """Delete a run, its results (CASCADE), and paper_decisions from this run.
    Removes the run from training memory so its papers can reappear in future
    searches and won't bias the feedback centroid.
    """
    with transaction() as conn:
        conn.execute("DELETE FROM paper_decisions WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM search_runs WHERE id=?", (run_id,))
    # Invalidate embedding caches
    cache = Path(__file__).parent / "cache"
    for f in ("centroid_accepted.npy", "centroid_rejected.npy"):
        (cache / f).unlink(missing_ok=True)


def get_last_run_date_for_profile(profile_id: int) -> dict | None:
    """Return {completed_at, lookback_days} of the last completed run for this profile, or None."""
    row = get_connection().execute(
        """SELECT completed_at, lookback_days FROM search_runs
           WHERE profile_id=? AND status='done'
           ORDER BY started_at DESC LIMIT 1""",
        (profile_id,),
    ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Papers
# ─────────────────────────────────────────────────────────────────────────────

def upsert_paper(oa_id: str, doi: str | None, title: str | None, abstract: str | None,
                 authors: list[str], year: int | None, venue: str | None,
                 cited_by_count: int, topics: list, raw_data: dict) -> None:
    now = datetime.utcnow().isoformat()
    with transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO papers
               (oa_id,doi,title,abstract,authors,year,venue,cited_by_count,topics,raw_data,fetched_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (oa_id, doi, title, abstract,
             json.dumps(authors), year, venue, cited_by_count,
             json.dumps(topics), json.dumps(raw_data, ensure_ascii=False), now),
        )


def get_paper(oa_id: str) -> dict | None:
    row = get_connection().execute(
        "SELECT * FROM papers WHERE oa_id=?", (oa_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    for f in ("authors", "topics", "raw_data"):
        try:
            d[f] = json.loads(d.get(f) or "[]")
        except (json.JSONDecodeError, TypeError):
            d[f] = []
    return d


def get_all_accepted_paper_texts() -> list[str]:
    """Return title+abstract of all historically accepted papers (for feedback embedding)."""
    rows = get_connection().execute(
        "SELECT title, abstract FROM paper_decisions WHERE decision='accepted'"
    ).fetchall()
    return [f"{r['title'] or ''}. {r['abstract'] or ''}".strip() for r in rows if r["title"]]


def get_all_rejected_paper_texts() -> list[str]:
    """Return title+abstract of all historically rejected papers (for feedback embedding)."""
    rows = get_connection().execute(
        "SELECT title, abstract FROM paper_decisions WHERE decision='rejected'"
    ).fetchall()
    return [f"{r['title'] or ''}. {r['abstract'] or ''}".strip() for r in rows if r["title"]]


# ─────────────────────────────────────────────────────────────────────────────
# Run results
# ─────────────────────────────────────────────────────────────────────────────

def add_run_result(run_id: int, paper_oa_id: str, score_library: float,
                   score_feedback: float, combined_score: float,
                   match_reasons: list[str]) -> int:
    with transaction() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO run_results
               (run_id,paper_oa_id,score_library,score_feedback,combined_score,match_reasons,status)
               VALUES(?,?,?,?,?,?,?)""",
            (run_id, paper_oa_id, score_library, score_feedback, combined_score,
             json.dumps(match_reasons), "pending"),
        )
    return cur.lastrowid


def get_run_results(run_id: int) -> list[dict]:
    rows = get_connection().execute(
        """SELECT rr.*, p.title, p.abstract, p.authors, p.year, p.venue,
                  p.doi, p.cited_by_count, p.topics
           FROM run_results rr
           LEFT JOIN papers p ON p.oa_id = rr.paper_oa_id
           WHERE rr.run_id=?
           ORDER BY rr.combined_score DESC""",
        (run_id,),
    ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        for f in ("match_reasons", "authors", "topics"):
            try:
                d[f] = json.loads(d.get(f) or "[]")
            except (json.JSONDecodeError, TypeError):
                d[f] = []
        out.append(d)
    return out


def decide_run_result(result_id: int, decision: str) -> None:
    """Update status on a single run_result row (pending→accepted/rejected/skipped)."""
    now = datetime.utcnow().isoformat()
    with transaction() as conn:
        conn.execute(
            "UPDATE run_results SET status=?, decided_at=? WHERE id=?",
            (decision, now, result_id),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Paper decisions  (persistent feedback learning store)
# ─────────────────────────────────────────────────────────────────────────────

def record_decision(paper_oa_id: str, decision: str, run_id: int | None,
                    title: str | None, abstract: str | None,
                    zotero_key: str | None = None) -> None:
    now = datetime.utcnow().isoformat()
    with transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO paper_decisions
               (paper_oa_id,decision,decided_at,run_id,zotero_key,title,abstract)
               VALUES(?,?,?,?,?,?,?)""",
            (paper_oa_id, decision, now, run_id, zotero_key, title, abstract),
        )


def get_decision(paper_oa_id: str) -> dict | None:
    row = get_connection().execute(
        "SELECT * FROM paper_decisions WHERE paper_oa_id=?", (paper_oa_id,)
    ).fetchone()
    return dict(row) if row else None
