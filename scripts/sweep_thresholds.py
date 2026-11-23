#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Buoc 1 — Sweep 16 nguong nhom (khong chon nguong o buoc nay).

cos in {0.98, 0.96, 0.94, 0.92} x dham in {6, 8, 10, 12} = 16 cau hinh.
Moi cau hinh: union-find toan bo S2, bao so nhom, so anh trong nhom da anh,
va canh bao bac cau (>50% anh cua 1 identity nam trong 1 nhom).

Doc chi tu S2_freeze/clip_b32_s2.npz.
Output: S2_freeze/sweep_thresholds.json
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("DATA_ROOT", "<REDACTED_PATH>"))
S2_RUN = BASE / "data/visual_logs/runs/S2_freeze"
COS_GRID = [0.98, 0.96, 0.94, 0.92]
DHAM_GRID = [6, 8, 10, 12]


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def run_config(emb, dh, var, n, cos_th, dham_th):
    dsu = DSU(n)
    # dham: pairwise qua batching
    for i0 in range(0, n, 256):
        i1 = min(i0 + 256, n)
        hd = (dh[i0:i1, None, :] != dh[None, :, :]).sum(axis=2)
        for bi in range(i1 - i0):
            gi = i0 + bi
            row = hd[bi, gi + 1:]
            for j0 in np.where(row <= dham_th)[0]:
                dsu.union(gi, gi + 1 + int(j0))
    # cos
    S = emb @ emb.T
    for i in range(n):
        row = S[i, i + 1:]
        for j0 in np.where(row >= cos_th)[0]:
            dsu.union(i, i + 1 + int(j0))
    roots = np.array([dsu.find(i) for i in range(n)])
    comp = defaultdict(list)
    for i, r in enumerate(roots):
        comp[r].append(i)
    groups = list(comp.values())
    sizes = Counter(len(g) for g in groups)
    n_multi = sum(v for k, v in sizes.items() if k >= 2)
    # canh bao bac cau
    by_var = defaultdict(list)
    for i in range(n):
        by_var[var[i]].append(i)
    warns = []
    for v, rows in by_var.items():
        r2g = Counter()
        for i in rows:
            r2g[roots[i]] += 1
        top_g, top_c = r2g.most_common(1)[0]
        if len(rows) >= 4 and top_c > 0.5 * len(rows):
            warns.append({"variant": v, "n": len(rows), "dom": int(top_c),
                          "frac": round(top_c / len(rows), 3)})
    warns.sort(key=lambda x: -x["frac"])
    return {"n_groups": len(groups),
            "size_hist": {str(k): v for k, v in sorted(sizes.items())},
            "n_images_in_multi": int(n_multi),
            "n_warnings": len(warns), "warnings": warns}


def main():
    z = np.load(Path(os.environ.get("CACHE_NPZ", str(S2_RUN / "clip_b32_s2.npz"))),
                 allow_pickle=False)
    emb = z["emb"]
    dh = z["dhash"].astype(bool)
    var = z["variants"]
    n = len(var)
    print(f"[sweep] n={n} — {len(COS_GRID)*len(DHAM_GRID)} cau hinh", flush=True)

    rows = []
    for cos_th in COS_GRID:
        for dham_th in DHAM_GRID:
            r = run_config(emb, dh, var, n, cos_th, dham_th)
            rows.append({"cos": cos_th, "dham": dham_th, **r})
            print(f"  cos>={cos_th} dham<={dham_th}: nhom={r['n_groups']} "
                  f"anh_da_nhom={r['n_images_in_multi']} "
                  f"canh_bao={r['n_warnings']} "
                  f"top_warn={r['warnings'][0] if r['warnings'] else '-'}",
                  flush=True)

    (S2_RUN / "sweep_thresholds.json").write_text(
        json.dumps({"cos_grid": COS_GRID, "dham_grid": DHAM_GRID, "rows": rows},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {S2_RUN}/sweep_thresholds.json", flush=True)


if __name__ == "__main__":
    main()
