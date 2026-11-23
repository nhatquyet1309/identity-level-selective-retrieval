#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v9rev_tierB [D20] - C7: Tier B, 11 query transform, SigLIP2-I (pass2b_capture.npz). Post hoc, mo ta.
Chay: Python311 -X utf8 -W ignore scripts/v9rev_tierB.py [--date YYYYMMDD]
Chi doc du lieu sealed; ghi results/v9rev_tierB_<date>/ (results.json, deviation_entry.md, SHA256SUMS).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent
SCRIPTS = EXP / "scripts"
V9 = EXP / "results" / "prereg_v9_gap_20261121"
sys.path.insert(0, str(SCRIPTS))
import prereg_v9_stats as S  # noqa: E402

TAU = 0.960025787  # tau* siglip2_I (rq1_results.json tau_star_cp.tau)
SEED = 20261022
CANON = ["G1", "L1", "L2", "L3", "L4", "L5"]
STRESS = ["G2", "X1", "X2", "X3", "X4"]
SRC = "03_experiments/scripts/make_transforms.py"
TRANSFORMS = {
    "G1": ("Horizontal flip (mirror left-right)", "horizontal mirror, no numeric parameter"),
    "L1": ("Brightness scaling (ImageEnhance.Brightness)", "factor 0.8"),
    "L2": ("Brightness scaling (ImageEnhance.Brightness)", "factor 1.2"),
    "L3": ("Saturation scaling (ImageEnhance.Color)", "factor 0.8"),
    "L4": ("Contrast scaling (ImageEnhance.Contrast)", "factor 0.8"),
    "L5": ("Hue shift in HSV", "+0.05 turn = 13 H units (of 255) ~ 18 degrees"),
    "G2": ("Vertical flip (top-bottom)", "vertical flip, no numeric parameter"),
    "X1": ("Brightness scaling (ImageEnhance.Brightness)", "factor 0.5"),
    "X2": ("Brightness scaling (ImageEnhance.Brightness)", "factor 1.5"),
    "X3": ("Saturation scaling (ImageEnhance.Color)", "factor 0.4"),
    "X4": ("Contrast scaling (ImageEnhance.Contrast)", "factor 0.5"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def avg_rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    sx = x[o]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = avg_rank(x), avg_rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def kendall_b(x, y):
    sx = np.sign(x[:, None] - x[None, :])
    sy = np.sign(y[:, None] - y[None, :])
    num = (sx * sy).sum() / 2
    n1, n2 = np.abs(sx).sum() / 2, np.abs(sy).sum() / 2
    den = np.sqrt(n1 * n2)
    return float(num / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20261122")
    args = ap.parse_args()
    out = EXP / "results" / ("v9rev_tierB_" + args.date)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(V9 / "pass2b_capture.npz")
    m = json.loads((V9 / "pass2b_meta.json").read_text(encoding="utf-8"))
    assert m["canonical_T"] == CANON and m["stress_T"] == STRESS
    names = m["transforms"]
    cats = m["q_categories"]
    i2c = dict(zip(m["q_identities"], m["q_categories"]))
    yc = S.build_yc_indices(m["ids"], i2c)
    a_all = z["id_scores_all"].astype(np.float64)
    b = z["b_id_scores"].astype(np.float64)
    pred, bpred = z["pred_idx"], z["b_pred_idx"]
    n = len(b)

    def gapyc(x):
        o = np.empty(len(x))
        for i in range(len(x)):
            v = np.sort(x[i, yc[S.CAT_TO_COARSE[cats[i]]]])[::-1]
            o[i] = v[0] - v[1]
        return o

    s1b, gb = b.max(1), gapyc(b)
    acc_b = s1b >= TAU
    per = {}
    for name in CANON + STRESS:
        t = names.index(name)
        a = a_all[:, t, :]
        sp, kd = [], []
        for i in range(n):
            ok = np.isfinite(a[i]) & np.isfinite(b[i])
            if ok.sum() >= 2:
                s = spearman(b[i][ok], a[i][ok])
                k = kendall_b(b[i][ok], a[i][ok])
                if np.isfinite(s):
                    sp.append(s)
                if np.isfinite(k):
                    kd.append(k)
        s1 = a.max(1)
        acc_t = s1 >= TAU
        ds, dm = np.abs(s1 - s1b), np.abs(gapyc(a) - gb)
        per[name] = dict(
            n=int(n), top1_flip_rate=float((pred[:, t] != bpred).mean()),
            spearman_mean=float(np.mean(sp)), spearman_median=float(np.median(sp)),
            kendall_tau_b_mean=float(np.mean(kd)), kendall_tau_b_median=float(np.median(kd)),
            decision_flip_rate=float((acc_b != acc_t).mean()),
            accept_to_refuse=float((acc_b & ~acc_t).mean()), refuse_to_accept=float((~acc_b & acc_t).mean()),
            median_abs_delta_s1=float(np.median(ds)), median_abs_delta_mid=float(np.median(dm)),
            p95_abs_delta_s1=float(np.percentile(ds, 95)), p95_abs_delta_mid=float(np.percentile(dm, 95)),
            rho_ratio_median=float(np.median(dm) / np.median(ds)), n_valid_rank_queries=len(sp))

    keys = ["top1_flip_rate", "spearman_mean", "spearman_median", "kendall_tau_b_mean", "kendall_tau_b_median",
            "decision_flip_rate", "accept_to_refuse", "refuse_to_accept", "median_abs_delta_s1",
            "median_abs_delta_mid", "p95_abs_delta_s1", "p95_abs_delta_mid", "rho_ratio_median"]
    groups = {}
    for g, nm in (("canonical", CANON), ("stress", STRESS)):
        groups[g] = {k: float(np.mean([per[x][k] for x in nm])) for k in keys}
        groups[g]["transforms"] = nm

    tr = [dict(name=x, group="canonical" if x in CANON else "stress", operation=TRANSFORMS[x][0],
               parameter=TRANSFORMS[x][1], source=SRC) for x in CANON + STRESS]
    R = dict(
        deviation="D20", post_hoc=True,
        label="post hoc, descriptive (Tier B query-transform sensitivity, SigLIP2-I only, local sealed data)",
        seed=SEED,
        sources=[SRC, "03_experiments/scripts/prereg_v9_stats.py",
                 "03_experiments/results/prereg_v9_gap_20261121/pass2b_capture.npz",
                 "03_experiments/results/prereg_v9_gap_20261121/pass2b_meta.json",
                 "03_experiments/results/prereg_v9_gap_20261121/p2b_results.json",
                 "03_experiments/results/preregv7_tierB_20261118/transforms_run.log"],
        encoder="siglip2_I", n_queries=int(n), script_sha256=sha(SCRIPTS / "make_transforms.py"),
        tau_star=TAU, baseline_accept_rate=float(acc_b.mean()),
        transforms=tr, per_transform=per, groups=groups,
        interpretation=("rho (median |delta gap_Yc| / median |delta s1|) compares the magnitude of score variation and "
                        "is not a measure of decision stability; the decision flip rate (s1 >= tau*) is the decision metric."))
    (out / "results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "deviation_entry.md").write_text(
        "## D20 - C7 Tier B: 11 query transform, flip/rank-corr/decision-flip/score-variation (post hoc) - %s\n\n"
        "- Pham vi: SigLIP2-I, pass2b_capture.npz (1667 query x 11 transform), post hoc, mo ta; khong doi du lieu sealed.\n"
        "- Chi so: top-1 flip, Spearman/Kendall tau-b theo query (identity huu han), decision flip s1 >= tau* = %.9f, "
        "score variation, rho = median|d gap_Yc| / median|d s1| (khop p2b_results.json 1e-9).\n"
        "- rho la ti so do lon bien thien diem, KHONG phai bang chung on dinh quyết định; decision flip moi la chi so quyết định.\n"
        "- Tai lap: scripts/v9rev_tierB.py.\n" % (args.date, TAU), encoding="utf-8")
    lines = ["%s  %s" % (sha(out / f), f) for f in ("deviation_entry.md", "results.json")]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("OK", out)


if __name__ == "__main__":
    main()
