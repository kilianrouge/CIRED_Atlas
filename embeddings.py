"""
embeddings.py – mxbai-embed-large-v1 based ranking for ATLAS.

Scoring (nearest-neighbour, replaces the old single-centroid approach):
  (A) score_library   – mean cosine similarity to the K_LIBRARY most similar papers
                        of the working-folder library. A library covering several themes
                        is no longer blurred into one average vector.
  (B) score_feedback  – mean cosine similarity to the K_ACCEPTED most similar papers the
                        current user has accepted. Rejected papers are NOT used: they are
                        on-topic-but-not-interesting and sit in the same region as accepted
                        ones, so penalising similarity to them did not help (see
                        tools/eval_ranking.py).

Combined score = weight_library * score_library + weight_feedback * score_feedback
(weights normalised; library-only while the user has no accepted papers).

Paper vectors are cached on disk keyed by a hash of their text, so only new or edited
papers are ever re-embedded and caches can never go stale.

Falls back to TF-IDF if the neural model cannot be loaded – loudly: the run is flagged
in the UI (see `info` in rank_candidates).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np

import database as db

_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
_MODEL_TAG = "mxbai"  # part of cache file names: changing the model invalidates caches
# Complete local copy of the model (tools/download_model.py). Loading from here never
# touches the Hugging Face Hub, whose lookups broke model loading from March 2026 on.
_MODEL_DIR = Path(__file__).parent / "models" / "mxbai-embed-large-v1"
_CACHE_DIR = Path(__file__).parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

K_LIBRARY = 10   # neighbours in the library used for score_library
K_ACCEPTED = 3   # neighbours among accepted papers used for score_feedback

# ── Model loading ─────────────────────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()
_encode_lock = threading.Lock()  # serialise encodes: parallel runs would just thrash the CPU

# Public, read by app.py: {"method": "neural" | "tfidf" | None (not loaded yet), "error": str | None}
model_status: dict = {"method": None, "error": None, "source": None}


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            try:
                if (_MODEL_DIR / "model.safetensors").exists():
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    source = str(_MODEL_DIR)
                else:
                    source = _MODEL_NAME
                from sentence_transformers import SentenceTransformer
                # low_cpu_mem_usage=True (default in ST 4.x) causes "meta tensor" errors on CPU-only
                _model = SentenceTransformer(source, model_kwargs={"low_cpu_mem_usage": False})
                model_status.update(method="neural", error=None, source=source)
                print(f"[ATLAS embeddings] Loaded {_MODEL_NAME} from {source}")
            except Exception as exc:
                model_status.update(method="tfidf", error=f"{type(exc).__name__}: {exc}"[:300])
                print("\n" + "!" * 78 +
                      f"\n[ATLAS embeddings] NEURAL MODEL UNAVAILABLE – ranking falls back to TF-IDF."
                      f"\n  {type(exc).__name__}: {exc}"
                      f"\n  Fix: python tools/download_model.py   (installs it into {_MODEL_DIR})\n"
                      + "!" * 78 + "\n")
                _model = False
    return _model if _model is not False else None


def _embed(texts: list[str], batch_size: int = 32, progress_cb=None) -> "np.ndarray | None":
    """Embed texts. Returns float32 unit-normalised array (N, dim) or None."""
    model = _get_model()
    if not model:
        return None
    n_batches = math.ceil(len(texts) / batch_size)
    all_vecs: list = []
    try:
        with _encode_lock:
            for i, start in enumerate(range(0, len(texts), batch_size)):
                batch = texts[start: start + batch_size]
                vecs = model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)
                all_vecs.append(vecs)
                if progress_cb:
                    progress_cb(i + 1, n_batches)
        return np.vstack(all_vecs) if all_vecs else None
    except Exception as exc:
        print(f"[ATLAS embeddings] Encode error: {exc}")
        return None


# ── Content-addressed vector cache ────────────────────────────────────────────

_cache_locks: dict[str, threading.Lock] = {}
_cache_lock_meta = threading.Lock()


def _safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "default"


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _embed_cached(texts: list[str], cache_path: Path, progress_cb=None,
                  log_label: str | None = None) -> "np.ndarray | None":
    """Vectors for `texts` (same order), embedding only texts not already in cache_path.

    The cache file holds exactly the current set of texts afterwards (obsolete ones dropped).
    Writes are atomic (tmp file + rename) because several ATLAS processes share cache/.
    """
    if not texts:
        return None
    with _cache_lock_meta:
        lock = _cache_locks.setdefault(str(cache_path), threading.Lock())
    keys = [_text_key(t) for t in texts]
    with lock:
        cached: dict[str, np.ndarray] = {}
        if cache_path.exists():
            try:
                with np.load(cache_path) as d:
                    cached = dict(zip(d["keys"].tolist(), d["vecs"]))
            except Exception as exc:
                print(f"[ATLAS embeddings] Ignoring unreadable cache {cache_path.name}: {exc}")
        todo: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in cached and k not in todo:
                todo[k] = t
        if todo:
            if log_label:
                print(f"[ATLAS embeddings] {log_label}: embedding {len(todo)} new paper(s), "
                      f"{len(set(keys)) - len(todo)} cached")
            new_vecs = _embed(list(todo.values()), progress_cb=progress_cb)
            if new_vecs is None:
                return None
            cached.update(zip(todo.keys(), new_vecs))
        out = np.vstack([cached[k] for k in keys]).astype(np.float32)
        current = list(dict.fromkeys(keys))
        if todo or len(cached) != len(current):
            _save_npz(cache_path, keys=np.array(current),
                      vecs=np.vstack([cached[k] for k in current]))
        return out


def _save_npz(path: Path, **arrays) -> None:
    """Best-effort cache write; never fails a run.

    cache/ is shared by every user's ATLAS process and has the sticky bit, so a rename over
    a file owned by another user is refused: then write in place (files are made 0666).
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.chmod(tmp, 0o666)
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            with open(tmp, "rb") as src, open(path, "wb") as dst:
                dst.write(src.read())
    except OSError as exc:
        print(f"[ATLAS embeddings] Could not write cache {path.name}: {exc}")
    finally:
        tmp.unlink(missing_ok=True)


def library_vectors(library_texts: list[str], collection_key: str = "default",
                    progress_cb=None) -> "np.ndarray | None":
    path = _CACHE_DIR / f"vecs_{_MODEL_TAG}_lib_{_safe(collection_key)}.npz"
    return _embed_cached(library_texts, path, progress_cb=progress_cb, log_label="Library")


def accepted_vectors() -> "np.ndarray | None":
    """Vectors of the current user's accepted papers (per-user cache file)."""
    texts = db.get_all_accepted_paper_texts()
    path = _CACHE_DIR / f"vecs_{_MODEL_TAG}_accepted_{_safe(db.get_current_user())}.npz"
    return _embed_cached(texts, path, log_label="Accepted feedback") if texts else None


# ── Scoring ───────────────────────────────────────────────────────────────────

def topk_mean_sim(X: "np.ndarray", R: "np.ndarray | None", k: int) -> "np.ndarray | None":
    """For each row of X: mean cosine similarity to its k nearest rows of R (unit vectors)."""
    if R is None or len(R) == 0:
        return None
    S = X @ R.T
    k = max(1, min(k, R.shape[0]))
    return np.partition(S, S.shape[1] - k, axis=1)[:, -k:].mean(axis=1)


def score_vectors(cand: "np.ndarray", lib: "np.ndarray | None", acc: "np.ndarray | None",
                  weight_library: float = 0.65, weight_feedback: float = 0.35,
                  k_library: int = K_LIBRARY, k_accepted: int = K_ACCEPTED):
    """Pure scoring function (shared with tools/eval_ranking.py).

    Returns (combined, score_library, score_feedback, wl, wf) as arrays / floats.
    """
    n = len(cand)
    sl = topk_mean_sim(cand, lib, k_library)
    sf = topk_mean_sim(cand, acc, k_accepted)
    if sl is None:
        sl = np.full(n, 0.5, dtype=np.float32)
    if sf is None:
        # No accepted papers yet: library only, so scores stay in the natural cosine range
        return sl.copy(), sl, np.zeros(n, dtype=np.float32), 1.0, 0.0
    total = weight_library + weight_feedback
    wl = weight_library / total if total else 1.0
    wf = weight_feedback / total if total else 0.0
    return wl * sl + wf * sf, sl, sf, wl, wf


# ── TF-IDF fallback ───────────────────────────────────────────────────────────

def _tfidf_cosine(lib_texts: list[str], candidate_texts: list[str]) -> "np.ndarray":
    def ngrams(text: str, n: int = 3) -> Counter:
        t = text.lower()
        return Counter(t[i: i + n] for i in range(len(t) - n + 1))

    doc_freq: Counter = Counter()
    lib_vecs = [ngrams(t) for t in lib_texts]
    for v in lib_vecs:
        for k in v:
            doc_freq[k] += 1

    N = len(lib_texts)
    idf = {k: math.log(N / df) for k, df in doc_freq.items()}

    def tfidf_vec(counter: Counter) -> dict[str, float]:
        total = sum(counter.values()) or 1
        return {k: (cnt / total) * idf.get(k, 0) for k, cnt in counter.items()}

    def cosine(a: dict, b: dict) -> float:
        keys = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v ** 2 for v in a.values())) or 1e-9
        nb = math.sqrt(sum(v ** 2 for v in b.values())) or 1e-9
        return dot / (na * nb)

    tfidf_lib = [tfidf_vec(v) for v in lib_vecs]
    centroid: Counter = Counter()
    for tv in tfidf_lib:
        for k, v in tv.items():
            centroid[k] += v / len(tfidf_lib)

    scores = []
    for ct in candidate_texts:
        cv = tfidf_vec(ngrams(ct))
        scores.append(cosine(dict(centroid), cv))
    return np.array(scores, dtype=np.float32)


# ── Main ranking function ─────────────────────────────────────────────────────

def rank_candidates(
    library_texts: list[str],
    candidates: list[dict],
    weight_library: float = 0.7,
    weight_feedback: float = 0.3,
    threshold: float = 0.0,
    collection_key: str = "default",
    progress_cb=None,  # callable(pct: int, text: str) — same signature as _progress in app.py
    run_id: int | None = None,
    info: dict | None = None,  # filled with {"method": "neural"|"tfidf", "error": ...}
) -> list[tuple[float, float, float, dict]]:
    """
    Score and rank candidate papers.

    Returns list of (combined_score, score_library, score_feedback, paper_dict)
    sorted descending by combined_score, filtered by threshold.
    """
    info = info if info is not None else {}
    if not candidates:
        return []
    _tag = f"[run {run_id}]" if run_id is not None else "[ATLAS]"

    candidate_texts = [
        f"{p.get('title') or ''}. {p.get('abstract') or ''}".strip()
        for p in candidates
    ]

    n_lib = len(library_texts)
    n_cand = len(candidates)
    _batch = 32  # must match batch_size used in _embed

    def _lib_progress(cur: int, tot: int) -> None:
        if progress_cb:
            progress_cb(85 + round(cur / max(tot, 1) * 4),
                        f"4/5\nEmbedding new library papers: {min(cur * _batch, n_lib)}")

    def _cand_progress(cur: int, tot: int) -> None:
        if progress_cb:
            progress_cb(89 + round(cur / max(tot, 1) * 5),
                        f"4/5\nEmbedding papers: {min(cur * _batch, n_cand)}/{n_cand} papers")

    lib = cand = acc = None
    if _get_model() is not None:
        lib = library_vectors(library_texts, collection_key, progress_cb=_lib_progress) if library_texts else None
        cand = _embed(candidate_texts, progress_cb=_cand_progress)
        acc = accepted_vectors()

    if cand is not None:
        info.update(method="neural", error=None)
        combined, sl, sf, wl, wf = score_vectors(cand, lib, acc, weight_library, weight_feedback)
        n_acc = 0 if acc is None else len(acc)
        print(f"{_tag} Scoring (neural kNN): library k={K_LIBRARY} over {0 if lib is None else len(lib)} papers"
              f" | feedback k={K_ACCEPTED} over {n_acc} accepted | weights lib={wl:.2f} fb={wf:.2f}")
    else:
        # Loud fallback: flagged on the run so the user can see ranking quality is degraded
        info.update(method="tfidf", error=model_status.get("error") or "neural embedding failed")
        print(f"{_tag} WARNING: neural ranking unavailable ({info['error']}) – using TF-IDF, feedback ignored")
        sl = (_tfidf_cosine(library_texts, candidate_texts) if library_texts
              else np.full(n_cand, 0.5, dtype=np.float32))
        sf = np.zeros(n_cand, dtype=np.float32)
        combined, wl, wf = sl, 1.0, 0.0

    ranked = [(float(c), float(l), float(f), paper)
              for c, l, f, paper in zip(combined, sl, sf, candidates) if c >= threshold]
    ranked.sort(key=lambda x: x[0], reverse=True)

    # ── Summary log ──────────────────────────────────────────────────────────
    if ranked:
        scores = [r[0] for r in ranked]
        print(f"{_tag} Ranking done ({info['method']}): {len(ranked)}/{n_cand} above threshold "
              f"| top={scores[0]:.3f} median={scores[len(scores)//2]:.3f} min={scores[-1]:.3f}")
        print(f"{_tag}   top papers (combined / library / feedback):")
        for c, l, f, paper in ranked[:5]:
            title = (paper.get("title") or "?")[:60]
            print(f'{_tag}     {c:.3f} = {wl:.2f}×{l:.3f}lib + {wf:.2f}×{f:.3f}fb  “{title}”')
    else:
        print(f"{_tag} Ranking done ({info['method']}): 0 candidates above threshold={threshold}")

    return ranked


def warm_up() -> dict:
    """Load the model (call from a background thread at startup). Returns model_status."""
    _get_model()
    return model_status


def invalidate_library_centroid(collection_key: str = "default") -> None:
    """Kept for app.py compatibility. The vector cache is content-addressed and never stale,
    so this only removes files left by the old centroid-based ranking."""
    for p in _CACHE_DIR.glob("centroid_*"):
        p.unlink(missing_ok=True)
