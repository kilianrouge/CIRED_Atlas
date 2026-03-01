"""
embeddings.py – mxbai-embed-large-v1 based ranking for ATLAS.

Two scoring axes:
  (A) score_library   – cosine similarity to centroid of working-folder library
  (B) score_feedback  – composite: +boost if similar to accepted, -penalty if similar to rejected

Combined score = weight_library * score_library + weight_feedback * score_feedback

Falls back to TF-IDF if sentence-transformers is unavailable.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

import database as db

_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
_CACHE_DIR = Path(__file__).parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

# ── Model loading ─────────────────────────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
        except Exception as exc:
            print(f"[ATLAS embeddings] Model unavailable: {exc}. Using TF-IDF fallback.")
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


# ── Centroid helpers ──────────────────────────────────────────────────────────

def _mean_centroid(vecs: "np.ndarray") -> "np.ndarray":
    c = vecs.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm > 1e-9:
        c /= norm
    return c


def _build_centroid_cached(texts: list[str], cache_path: Path, max_age_h: float = 12.0) -> "np.ndarray | None":
    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < max_age_h:
            return np.load(str(cache_path))
    if not texts:
        return None
    vecs = _embed(texts)
    if vecs is None:
        return None
    centroid = _mean_centroid(vecs)
    np.save(str(cache_path), centroid)
    return centroid


def build_library_centroid(library_texts: list[str], collection_key: str = "default",
                            force: bool = False) -> "np.ndarray | None":
    cache_path = _CACHE_DIR / f"centroid_lib_{collection_key}.npy"
    if force and cache_path.exists():
        cache_path.unlink()
    return _build_centroid_cached(library_texts, cache_path, max_age_h=12.0)


def build_feedback_centroids(force: bool = False) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """Build / load accepted and rejected centroids from persistent DB decisions."""
    accepted_texts = db.get_all_accepted_paper_texts()
    rejected_texts = db.get_all_rejected_paper_texts()

    acc_path = _CACHE_DIR / "centroid_accepted.npy"
    rej_path = _CACHE_DIR / "centroid_rejected.npy"

    if force:
        for p in (acc_path, rej_path):
            if p.exists():
                p.unlink()

    acc_centroid = _build_centroid_cached(accepted_texts, acc_path, max_age_h=2.0) if accepted_texts else None
    rej_centroid = _build_centroid_cached(rejected_texts, rej_path, max_age_h=2.0) if rejected_texts else None
    return acc_centroid, rej_centroid


# ── Main ranking function ─────────────────────────────────────────────────────

def rank_candidates(
    library_texts: list[str],
    candidates: list[dict],
    weight_library: float = 0.7,
    weight_feedback: float = 0.3,
    threshold: float = 0.0,
    collection_key: str = "default",
    embed_progress_cb=None,
) -> list[tuple[float, float, float, dict]]:
    """
    Score and rank candidate papers.

    Returns list of (combined_score, score_library, score_feedback, paper_dict)
    sorted descending by combined_score, filtered by threshold.
    """
    if not candidates:
        return []

    candidate_texts = [
        f"{p.get('title') or ''}. {p.get('abstract') or ''}".strip()
        for p in candidates
    ]

    # ── Library similarity ────────────────────────────────────────────────────
    use_neural = _get_model() is not None

    if use_neural and library_texts:
        lib_centroid = build_library_centroid(library_texts, collection_key)
    else:
        lib_centroid = None

    if lib_centroid is not None:
        cand_vecs = _embed(candidate_texts, progress_cb=embed_progress_cb)
        if cand_vecs is not None:
            scores_lib = (cand_vecs @ lib_centroid).tolist()
        else:
            # Neural embed failed for candidates; fall back
            scores_lib = _tfidf_cosine(library_texts, candidate_texts).tolist()
    elif library_texts:
        scores_lib = _tfidf_cosine(library_texts, candidate_texts).tolist()
    else:
        scores_lib = [0.5] * len(candidates)

    # ── Feedback similarity ───────────────────────────────────────────────────
    acc_centroid, rej_centroid = build_feedback_centroids()

    if use_neural and (acc_centroid is not None or rej_centroid is not None):
        cand_vecs = cand_vecs if "cand_vecs" in dir() else _embed(candidate_texts)
        if cand_vecs is not None:
            scores_acc = (cand_vecs @ acc_centroid).tolist() if acc_centroid is not None else [0.0] * len(candidates)
            scores_rej = (cand_vecs @ rej_centroid).tolist() if rej_centroid is not None else [0.0] * len(candidates)
            scores_feedback = [
                max(0.0, a - r)
                for a, r in zip(scores_acc, scores_rej)
            ]
        else:
            scores_feedback = [0.0] * len(candidates)
    else:
        scores_feedback = [0.0] * len(candidates)

    # ── Combine ───────────────────────────────────────────────────────────────
    total_w = weight_library + weight_feedback
    wl = weight_library / total_w if total_w else 1.0
    wf = weight_feedback / total_w if total_w else 0.0

    ranked: list[tuple[float, float, float, dict]] = []
    for i, paper in enumerate(candidates):
        sl = float(scores_lib[i]) if i < len(scores_lib) else 0.0
        sf = float(scores_feedback[i]) if i < len(scores_feedback) else 0.0
        combined = wl * sl + wf * sf
        if combined >= threshold:
            ranked.append((combined, sl, sf, paper))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def invalidate_library_centroid(collection_key: str = "default") -> None:
    p = _CACHE_DIR / f"centroid_lib_{collection_key}.npy"
    if p.exists():
        p.unlink()
