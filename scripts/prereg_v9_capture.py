#!/usr/bin/env python3
"""prereg_v9_capture.py — Pass 1 (P1, D(ii) LOO) + Pass 2 (P2, 11T transforms).

Prereg SEALED: reports/prereg_v9.md (2026-10-22 seed, 77a64289 split).
Chạy trên <GPU_SERVER>: <PYTHON_ENV>/bin/python

Output (mặc định): /tmp/prereg_v7/prereg_v9_gap_20261121/
  pass1_capture.npz  — per-query 145 identity scores + meta (2 encoders)
  pass2_capture.npz  — per-(query,transform) scores cho siglip2_I
  capture_meta.json  — hashes, counts, definitions
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reuse prereg_v7 helpers on server
sys.path.insert(0, "/tmp/prereg_v7")
sys.path.insert(0, "<REDACTED_PATH>/tools")

import tierA_smoke as T  # noqa: E402
import tierA_v2 as V  # noqa: E402
import tierA_v3 as V3  # noqa: E402

SEED = 20261022
SPLIT_HASH_EXPECTED = "77a64289"
THETA = 0.98
ENCODERS = ["siglip2", "clip"]  # image-only I
ENCODER_IDS = {"siglip2": "siglip2_I", "clip": "clip_I"}
# P2 canonical 6T / stress 5T (prereg_v9 §2 + make_transforms.py CLASSES)
CANONICAL_T = ["G1", "L1", "L2", "L3", "L4", "L5"]
STRESS_T = ["G2", "X1", "X2", "X3", "X4"]
ALL_T = CANONICAL_T + STRESS_T


def log(*a):
    print(*a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rq1_split(path, expected_prefix=SPLIT_HASH_EXPECTED):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    h = d.get("split_hash", "")
    if not h.startswith(expected_prefix):
        raise ValueError(f"split_hash {h} does not start with {expected_prefix}")
    calib = list(d["calib"])
    test = list(d["test"])
    if len(calib) != 73 or len(test) != 72:
        raise ValueError(f"unexpected split sizes: {len(calib)}/{len(test)}")
    if set(calib) & set(test):
        raise ValueError("calib/test overlap")
    return d, calib, test


def identity_scores_for_row(S_row, ids, id_cols):
    """Max-pool image scores over identity columns. S_row: (1667,) with masked -inf."""
    out = np.empty(len(ids), dtype=np.float32)
    for j, v in enumerate(ids):
        cols = id_cols[v]
        out[j] = float(S_row[cols].max())
    return out


def top2_within_identity(S_row, cols):
    """Top-2 image-level scores within one identity's column set."""
    vals = S_row[cols]
    if vals.size == 0:
        return -np.inf, -np.inf, -1, -1
    order = np.argsort(-vals, kind="mergesort")
    i1 = int(order[0])
    s1 = float(vals[i1])
    if vals.size == 1:
        return s1, -np.inf, int(cols[i1]), -1
    i2 = int(order[1])
    s2 = float(vals[i2])
    return s1, s2, int(cols[i1]), int(cols[i2])


def build_id_cols(meta, ids):
    id_arr = np.array([m["variant"] for m in meta])
    return {v: np.where(id_arr == v)[0].astype(np.int64) for v in ids}


def build_cells_I(paths_can):
    """Build image-only I cells for siglip2, clip in canonical order."""
    cells = {}
    pins = {}
    for enc in ENCODERS:
        index, meta = T.load_meta_index(enc)
        perm, _, same = V3.perm_for(enc, paths_can)
        pins[enc] = dict(n_rows_same_order=int(same), n=int(perm.size))
        Xs_r, _ = V.renorm(T.stored_vectors(index))
        Xq_r, _ = V.renorm(T.get_emb(enc, meta))
        Simg = Xq_r @ Xs_r.T
        Simg = Simg[np.ix_(perm, perm)].astype(np.float32)
        np.fill_diagonal(Simg, -np.inf)
        cells[enc] = Simg
        log(f"  [{enc}] I cell {Simg.shape} ready (moved {perm.size - same})")
    return cells, pins


def load_dedup(paths_can, own, cat):
    """Load dedup_components.json from /tmp/prereg_v7 or compute."""
    dedup_path = Path("/tmp/prereg_v7/dedup_components.json")
    audit_path = Path("/tmp/prereg_v7/audit_dedup.json")
    if not dedup_path.exists():
        raise FileNotFoundError(f"missing {dedup_path}")
    doc = json.loads(dedup_path.read_text(encoding="utf-8"))
    paths_doc = [im["path"] for im in doc["images"]]
    if paths_doc != paths_can:
        raise ValueError("dedup_components.json path order mismatch vs canonical")
    comp98 = np.array([int(im["component_098"][1:]) for im in doc["images"]], dtype=np.int64)
    groups98 = V3.group_rows(comp98)
    inD_i = np.array([sum(1 for m in doc["images"] if m["identity"] == im["identity"]) >= 2 for im in doc["images"]], dtype=bool)
    # more directly
    ids = sorted(set(own.tolist()))
    nc = {v: int((own == v).sum()) for v in ids}
    inD_i = np.array([nc[v] >= 2 for v in own], dtype=bool)
    inD_ii = V3.build_D2(comp98, groups98, own, ids, nc)
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    return comp98, groups98, inD_i, inD_ii, nc, ids, audit


def masked_row(S, i, comp, groups):
    row = S[i].copy()
    mask = np.asarray(groups[int(comp[i])], dtype=np.int64)
    row[mask] = -np.inf
    return row


def capture_pass1(out_dir, split_path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index, meta = T.load_meta_index("siglip2")
    order = np.argsort([m["path"] for m in meta], kind="stable")
    paths_can = [meta[i]["path"] for i in order]
    own = np.array([meta[i]["variant"] for i in order])
    cat = np.array([meta[i]["category"] for i in order])

    split_doc, calib_ids, test_ids = load_rq1_split(split_path)
    split_map = {v: "calib" for v in calib_ids}
    split_map.update({v: "test" for v in test_ids})

    comp98, groups98, inD_i, inD_ii, nc, ids, audit = load_dedup(paths_can, own, cat)
    id_cols = build_id_cols([{"variant": v} for v in own], ids)
    # build proper id_cols with original meta structure
    meta_can = [{"path": p, "variant": o, "category": c} for p, o, c in zip(paths_can, own, cat)]
    id_cols = build_id_cols(meta_can, ids)

    n = len(paths_can)
    n_ids = len(ids)
    assert n == 1667 and n_ids == 145

    cells, pins = build_cells_I(paths_can)

    # D(ii) query indices in canonical order
    q_idx = np.where(inD_ii)[0].astype(np.int64)
    log(f"  D(ii) queries: {len(q_idx)} (expected 1663)")

    # pre-allocate
    res = {}
    for enc in ENCODERS:
        res[enc] = {
            "id_scores": np.full((len(q_idx), n_ids), -np.inf, dtype=np.float32),
            "s1_img": np.full(len(q_idx), -np.inf, dtype=np.float32),
            "s2_img_same_id": np.full(len(q_idx), -np.inf, dtype=np.float32),
            "s1_img_idx": np.full(len(q_idx), -1, dtype=np.int32),
            "s2_img_idx": np.full(len(q_idx), -1, dtype=np.int32),
            "pred_identity_idx": np.full(len(q_idx), -1, dtype=np.int32),
            "correct": np.zeros(len(q_idx), dtype=np.uint8),
        }

    gt_idx = np.array([ids.index(v) for v in own[q_idx]], dtype=np.int32)
    split_arr = np.array([split_map.get(own[i], "unknown") for i in q_idx])
    n_unknown_split = int((split_arr == "unknown").sum())
    if n_unknown_split:
        raise ValueError(f"{n_unknown_split} queries have identity not in RQ-1 split")

    t0 = time.time()
    for k, i in enumerate(q_idx):
        for enc in ENCODERS:
            row = masked_row(cells[enc], i, comp98, groups98)
            scores = identity_scores_for_row(row, ids, id_cols)
            res[enc]["id_scores"][k] = scores
            j_pred = int(np.argmax(scores))
            res[enc]["pred_identity_idx"][k] = j_pred
            res[enc]["correct"][k] = uint8 = np.uint8(j_pred == gt_idx[k])
            # top-2 images within predicted identity Y_c
            pred_id = ids[j_pred]
            s1, s2, i1, i2 = top2_within_identity(row, id_cols[pred_id])
            res[enc]["s1_img"][k] = s1
            res[enc]["s2_img_same_id"][k] = s2
            res[enc]["s1_img_idx"][k] = i1
            res[enc]["s2_img_idx"][k] = i2
        if (k + 1) % 200 == 0:
            log(f"  pass1 {k+1}/{len(q_idx)} ({time.time()-t0:.1f}s)")

    log(f"  pass1 done in {time.time()-t0:.1f}s")

    # Verify A(ii) accuracy matches tierA_v3 for sanity
    sanity = {}
    for enc in ENCODERS:
        acc = res[enc]["correct"].mean() * 100
        sanity[ENCODER_IDS[enc]] = float(acc)
        log(f"  sanity {ENCODER_IDS[enc]} A(ii) on D(ii): {acc:.4f}%")

    meta_out = {
        "prereg": "prereg_v9.md SEALED",
        "split_hash_expected_prefix": SPLIT_HASH_EXPECTED,
        "split_hash": split_doc.get("split_hash"),
        "split_sha256": sha256_file(split_path),
        "theta_cos": THETA,
        "n_gallery": n,
        "n_identities": n_ids,
        "n_Dii": int(len(q_idx)),
        "n_calib_ids": len(calib_ids),
        "n_test_ids": len(test_ids),
        "encoders": ENCODERS,
        "encoder_ids": ENCODER_IDS,
        "ids": ids,
        "q_idx": q_idx.tolist(),
        "q_paths": [paths_can[i] for i in q_idx],
        "q_identities": [own[i] for i in q_idx],
        "q_categories": [cat[i] for i in q_idx],
        "q_nc": [int(nc[own[i]]) for i in q_idx],
        "q_split": split_arr.tolist(),
        "q_component_098": [int(comp98[i]) for i in q_idx],
        "sanity_acc_pct": sanity,
        "pins": pins,
        "sc4_definition": "ab4/spec_v2: m_id = s1_id - s2_id (gap between top-1 and top-2 identity scores, identity-level max-pool). Also saved: s1_img, s2_img_same_id within predicted identity for transparency.",
        "score_definitions": {
            "Sc1_sim": "max identity score (top-1 identity)",
            "Sc2_MSP": "max softmax probability over 145 identity scores at T* (argmin E-AURC on calib)",
            "Sc3_energy": "T*logsumexp(s/T*) (higher=more confident), T* argmin E-AURC on calib",
            "Sc4_gap": "m_id = s1_id - s2_id (top1-top2 identity scores)"
        }
    }

    npz_path = out_dir / "pass1_capture.npz"
    np.savez_compressed(
        npz_path,
        q_idx=q_idx,
        gt_idx=gt_idx,
        **{f"{e}_id_scores": res[e]["id_scores"] for e in ENCODERS},
        **{f"{e}_s1_img": res[e]["s1_img"] for e in ENCODERS},
        **{f"{e}_s2_img_same_id": res[e]["s2_img_same_id"] for e in ENCODERS},
        **{f"{e}_pred_idx": res[e]["pred_identity_idx"] for e in ENCODERS},
        **{f"{e}_correct": res[e]["correct"] for e in ENCODERS},
    )
    (out_dir / "pass1_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  wrote {npz_path}")
    return meta_out, res


def load_transform_map(manifest_csv):
    """Map (src_path, transform) -> abs_path."""
    m = {}
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[(r["src_path"], r["transform"])] = r["abs_path"]
    return m


def capture_pass2(out_dir, split_path, manifest_csv, transform_root, out_prefix="pass2b"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index, meta = T.load_meta_index("siglip2")
    order = np.argsort([m["path"] for m in meta], kind="stable")
    paths_can = [meta[i]["path"] for i in order]
    own = np.array([meta[i]["variant"] for i in order])
    cat = np.array([meta[i]["category"] for i in order])

    comp98, groups98, inD_i, inD_ii, nc, ids, _ = load_dedup(paths_can, own, cat)
    meta_can = [{"path": p, "variant": o, "category": c} for p, o, c in zip(paths_can, own, cat)]
    id_cols = build_id_cols(meta_can, ids)

    # Build siglip2 I stored matrix (renorm) in canonical order
    index_s, meta_s = T.load_meta_index("siglip2")
    perm, _, _ = V3.perm_for("siglip2", paths_can)
    Xs_r, _ = V.renorm(T.stored_vectors(index_s))
    Xs_r = Xs_r[perm].astype(np.float32)
    # diag mask: we'll mask component per query; self is in component if singleton

    tmap = load_transform_map(manifest_csv)
    log(f"  transform map entries: {len(tmap)}")

    n = len(paths_can)
    n_ids = len(ids)
    q_idx = np.arange(n, dtype=np.int64)  # all 1667 queries, as prereg says 11T x 1667
    # Actually prereg says "18.337 query biến đổi (11 T × 1.667)". All gallery images.
    # Mask component like phase-2 masked: mask entire component of source x.

    nT = len(ALL_T)
    total = n * nT
    log(f"  P2: {n} queries x {nT} transforms = {total}")

    # store: for each (q, t) save s1 (top identity score), s2_id (2nd identity score),
    # m_id, s1_img_within_pred, s2_img_within_pred. Also delta can be computed later against pass1 baseline.
    # We need baseline m_id and s1 for original x under same masking? P2 says "Δm_id = m_id(T(x)) - m_id(x)".
    # m_id(x) should be computed on same A(ii) component-masked setting. Pass1 gives us siglip2 for D(ii) only; but P2 uses all 1667.
    # We'll compute baseline for all 1667 in pass2 as T="orig" or just compute.
    # Efficient: store [n, nT] for key scalars.

    shape = (n, nT)
    s1_id = np.full(shape, -np.inf, dtype=np.float32)
    s2_id = np.full(shape, -np.inf, dtype=np.float32)
    s1_img = np.full(shape, -np.inf, dtype=np.float32)
    s2_img_same = np.full(shape, -np.inf, dtype=np.float32)
    pred_idx = np.full(shape, -1, dtype=np.int32)
    correct = np.zeros(shape, dtype=np.uint8)
    # D12-part2: 145-dim identity score vectors per (query, transform) for Y_c-restricted Sc4
    id_scores_all = np.full((n, nT, n_ids), -np.inf, dtype=np.float32)

    # also baseline (original image, same masking)
    b_s1_id = np.full(n, -np.inf, dtype=np.float32)
    b_s2_id = np.full(n, -np.inf, dtype=np.float32)
    b_s1_img = np.full(n, -np.inf, dtype=np.float32)
    b_s2_img_same = np.full(n, -np.inf, dtype=np.float32)
    b_pred_idx = np.full(n, -1, dtype=np.int32)
    b_correct = np.zeros(n, dtype=np.uint8)
    b_id_scores = np.full((n, n_ids), -np.inf, dtype=np.float32)

    gt_idx = np.array([ids.index(v) for v in own], dtype=np.int32)

    # build baseline from Simg for siglip2
    S_base = Xs_r @ Xs_r.T  # but wait - baseline should be query embedding vs gallery. For original, query embedding = stored vector (self-retrieval case) but with component masked. This is same as A(ii) for that query.
    # Better: use get_emb cache for siglip2 to get query embeddings in canonical order.
    Xq_r, _ = V.renorm(T.get_emb("siglip2", meta_s))
    Xq_r = Xq_r[perm].astype(np.float32)
    S_orig = Xq_r @ Xs_r.T
    S_orig = S_orig.astype(np.float32)

    t0 = time.time()
    for i in range(n):
        row = masked_row(S_orig, i, comp98, groups98)
        scores = identity_scores_for_row(row, ids, id_cols)
        b_id_scores[i] = scores
        order_ids = np.argsort(-scores, kind="mergesort")
        j1 = int(order_ids[0]); j2 = int(order_ids[1]) if len(order_ids) > 1 else -1
        b_s1_id[i] = scores[j1]
        b_s2_id[i] = scores[j2] if j2 >= 0 else -np.inf
        b_pred_idx[i] = j1
        b_correct[i] = np.uint8(j1 == gt_idx[i])
        ps1, ps2, _, _ = top2_within_identity(row, id_cols[ids[j1]])
        b_s1_img[i] = ps1
        b_s2_img_same[i] = ps2
        if (i + 1) % 500 == 0:
            log(f"  baseline {i+1}/{n}")

    log(f"  baseline done {time.time()-t0:.1f}s")

    # transforms
    import torch
    from PIL import Image
    from siglip_engine import load_siglip
    wrap = load_siglip()

    def embed_image(path):
        img = Image.open(path).convert("RGB")
        try:
            t = wrap.preprocess(img).unsqueeze(0)
            with torch.no_grad():
                e = wrap.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True)
            v = e.float().cpu().numpy().astype(np.float32)[0]
        finally:
            img.close()
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.astype(np.float32)

    # batch by transform for cache locality? Or by image. By image is easier.
    t0 = time.time()
    done = 0
    for i in range(n):
        src = paths_can[i]
        comp_mask = np.asarray(groups98[int(comp98[i])], dtype=np.int64)
        for ti, tr in enumerate(ALL_T):
            key = (src, tr)
            p = tmap.get(key)
            if p is None:
                # try relative/abs variations
                p = tmap.get((src.replace("\\", "/"), tr))
            if p is None or not os.path.exists(p):
                raise FileNotFoundError(f"missing transform {tr} for {src} -> {p}")
            qv = embed_image(p)
            row = (qv @ Xs_r.T).astype(np.float32)
            row[comp_mask] = -np.inf
            scores = identity_scores_for_row(row, ids, id_cols)
            id_scores_all[i, ti] = scores
            order_ids = np.argsort(-scores, kind="mergesort")
            j1 = int(order_ids[0]); j2 = int(order_ids[1]) if len(order_ids) > 1 else -1
            s1_id[i, ti] = scores[j1]
            s2_id[i, ti] = scores[j2] if j2 >= 0 else -np.inf
            pred_idx[i, ti] = j1
            correct[i, ti] = np.uint8(j1 == gt_idx[i])
            ps1, ps2, _, _ = top2_within_identity(row, id_cols[ids[j1]])
            s1_img[i, ti] = ps1
            s2_img_same[i, ti] = ps2
            done += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            log(f"  pass2 {i+1}/{n} ({done}/{total}, {rate:.1f} q/s, ETA {eta/60:.1f}m)")

    log(f"  pass2 done in {time.time()-t0:.1f}s")

    np.savez_compressed(
        out_dir / f"{out_prefix}_capture.npz",
        s1_id=s1_id, s2_id=s2_id, s1_img=s1_img, s2_img_same=s2_img_same,
        pred_idx=pred_idx, correct=correct,
        b_s1_id=b_s1_id, b_s2_id=b_s2_id, b_s1_img=b_s1_img, b_s2_img_same=b_s2_img_same,
        b_pred_idx=b_pred_idx, b_correct=b_correct,
        b_id_scores=b_id_scores,
        id_scores_all=id_scores_all,
        gt_idx=gt_idx,
        comp98=comp98,
    )
    meta_out = {
        "prereg": "prereg_v9.md SEALED",
        "n_queries": n,
        "n_transforms": nT,
        "total": total,
        "transforms": ALL_T,
        "canonical_T": CANONICAL_T,
        "stress_T": STRESS_T,
        "encoder": "siglip2_I",
        "mask": "A(ii) component mask theta=0.98, same as tierA_v3",
        "m_id_definition": "s1_id - s2_id (top1-top2 identity max-pooled scores); global identity top-2 from capture scalars",
        "s1_definition": "top1 identity max-pooled score (absolute similarity)",
        "delta_definition": "delta = value(T(x)) - value(x) on same masked setting",
        "ids": ids,
        "q_paths": paths_can,
        "q_identities": own.tolist(),
        "q_categories": cat.tolist(),
        "out_prefix": out_prefix,
        "has_id_scores_all": True,
        "id_scores_shape": [n, nT, n_ids],
    }
    (out_dir / f"{out_prefix}_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  wrote {out_prefix}_capture.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/prereg_v7/prereg_v9_gap_20261121")
    ap.add_argument("--split", default="/tmp/prereg_v7/rq1_split_77a64289.json")
    ap.add_argument("--manifest", default="/tmp/prereg_v7/tierB_phase1/transforms_manifest.csv")
    ap.add_argument("--stage", choices=["pass1", "pass2", "pass2b", "all"], default="all")
    ap.add_argument("--out-prefix", default="pass2b")
    args = ap.parse_args()

    log(f"# prereg_v9_capture.py {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    log(f"# out={args.out_dir}")
    if args.stage in ("pass1", "all"):
        capture_pass1(args.out_dir, args.split)
    if args.stage in ("pass2", "all", "pass2b"):
        capture_pass2(args.out_dir, args.split, args.manifest, None, out_prefix=args.out_prefix)
    log("# done")


if __name__ == "__main__":
    main()
