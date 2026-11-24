"""Offline benchmark of ATLAS ranking on a user's accept/reject history.

Measures how well a scoring method puts the papers the user accepted above the ones they
rejected. Run it before changing anything in embeddings.py.

Usage (from the ATLAS folder, with the app's Python):
    python tools/eval_ranking.py rouge AWQUBV2G
        <user>            : data/users/<user>.db (opened read-only)
        <collection key>  : library cached in cache/library_<key>.json (written by the app)
    options: --split-date 2026-09-01   temporal split (train before, test from that date)

Protocols
  cv       : stratified 5-fold CV x4 on decided papers
  temporal : learn from decisions before --split-date, test on later ones (realistic)
Metrics: ROC-AUC, average precision (AP; random = share of accepted), and the share of
accepted papers that land in the top 20% of the ranking.

Caveats: accepted papers pushed to Zotero are in the library too, so library items that
duplicate a decided paper are removed (else nearest-neighbour scores leak the label).
Labels only exist for papers ATLAS showed, so the decided set is biased by the ranker in use.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import embeddings as emb  # noqa: E402
import zotero_client as zc  # noqa: E402


def load(user: str, collection_key: str):
    c = sqlite3.connect(f"file:{ROOT / 'data' / 'users' / (user + '.db')}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute("""
      SELECT d.decision, d.decided_at, COALESCE(d.title, p.title) title,
             COALESCE(NULLIF(d.abstract,''), p.abstract) abstract
      FROM paper_decisions d LEFT JOIN papers p ON p.oa_id=d.paper_oa_id
      WHERE d.decision IN ('accepted','rejected')""").fetchall()
    dec = [dict(r) for r in rows if r["title"]]
    texts = [f"{d['title'] or ''}. {d['abstract'] or ''}".strip() for d in dec]
    meta = zc.extract_item_metadata(json.load(open(ROOT / "cache" / f"library_{collection_key}.json")))
    lib_texts = [f"{m.get('title') or ''}. {m.get('abstract') or ''}".strip()
                 for m in meta if m.get("title") or m.get("abstract")]
    y = np.array([d["decision"] == "accepted" for d in dec], dtype=int)
    dates = np.array([d["decided_at"] for d in dec])
    return texts, lib_texts, y, dates


def _centroid(v):
    c = v.mean(0)
    return c / (np.linalg.norm(c) + 1e-9)


def methods(Xtr, ytr, Xte, L):
    """Each method: scores for Xte, using only the library and the training decisions."""
    acc, rej = Xtr[ytr == 1], Xtr[ytr == 0]
    out = {"old: cos to library centroid": Xte @ _centroid(L)}
    if len(acc) and len(rej):
        fb = np.maximum(0, Xte @ _centroid(acc) - Xte @ _centroid(rej))
        out["old: centroid + (accepted - rejected) feedback"] = 0.65 * (Xte @ _centroid(L)) + 0.35 * fb
    out[f"new: kNN library only (k={emb.K_LIBRARY})"] = emb.score_vectors(Xte, L, None)[0]
    out["new: production scorer (kNN library + accepted, 0.65/0.35)"] = emb.score_vectors(Xte, L, acc)[0]
    return out


def report(name, y_true, s, k_frac=0.2):
    top = np.argsort(-s)[:max(1, int(round(k_frac * len(s))))]
    return (roc_auc_score(y_true, s), average_precision_score(y_true, s),
            y_true[top].sum() / max(1, y_true.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user"); ap.add_argument("collection_key")
    ap.add_argument("--split-date", default="2026-09-01")
    a = ap.parse_args()

    texts, lib_texts, y, dates = load(a.user, a.collection_key)
    cache = emb._CACHE_DIR
    X = emb._embed_cached(texts, cache / f"vecs_{emb._MODEL_TAG}_eval_{emb._safe(a.user)}.npz", log_label="Decisions")
    L = emb.library_vectors(lib_texts, a.collection_key)
    if X is None or L is None:
        raise SystemExit(f"Neural model unavailable: {emb.model_status}")
    L = L[(L @ X.T).max(1) < 0.97]  # drop library copies of decided papers
    print(f"user={a.user}: {len(y)} decisions ({y.sum()} accepted), library={len(L)} after dedup")

    res: dict = {}
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0).split(X, y):
        for k, s in methods(X[tr], y[tr], X[te], L).items():
            res.setdefault(k, []).append(report(k, y[te], s))
    tr, te = dates < a.split_date, dates >= a.split_date
    temporal = None
    if y[tr].sum() and y[te].sum():
        temporal = {k: report(k, y[te], s) for k, s in methods(X[tr], y[tr], X[te], L).items()}
        print(f"temporal: train {tr.sum()} ({y[tr].sum()} acc) / test {te.sum()} ({y[te].sum()} acc), "
              f"random AP={y[te].mean():.3f}")
    print(f"{'method':62s} | {'CV AUC':>6} {'AP':>6} {'R@20%':>6} | {'T AUC':>6} {'AP':>6} {'R@20%':>6}")
    for k, v in res.items():
        m = np.mean(v, 0)
        t = temporal[k] if temporal else (np.nan,) * 3
        print(f"{k:62s} | {m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f} | {t[0]:6.3f} {t[1]:6.3f} {t[2]:6.3f}")


if __name__ == "__main__":
    main()
