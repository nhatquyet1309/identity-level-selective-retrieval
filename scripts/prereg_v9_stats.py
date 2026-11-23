#!/usr/bin/env python3
"""prereg_v9_stats.py — P1 + P2 stats for prereg_v9 SEALED.

P1: D(ii) 1663 queries, split RQ-1 77a64289 (73 calib / 72 test).
    4 scores × 2 encoders (siglip2_I, clip_I):
      Sc1 sim      = max identity score
      Sc2 MSP(T*)  = max softmax prob, T* = argmin E-AURC on calib (ab4 grid)
      Sc3 energy   = T* * logsumexp(s/T*), higher=confident, T* argmin E-AURC calib
      Sc4 gap(m_id)= s1_img - s2_img_same_id within predicted identity Y_c
                     (Algorithm 1 line 10, image-level in Y_c; capture has it)
    Metrics: AURC/E-AURC/AUROC. Paired ΔE-AURC gap-{sim,MSP,energy} + cluster
    bootstrap 2000 identity / seed 20261022 + Holm 3. coverage@risk(5%) descriptive.
    Labels: better (CI excludes 0 on better side) / insufficient. NO new delta.

P2: siglip2_I, 18,337 = 11T×1667, component-masked. Δm_id, Δs1 vs baseline.
    Groups canonical 6T + stress 5T. median |Δ|, P95, ratio median(|Δm_id|)/median(|Δs1|)
    + cluster bootstrap identity 2000/seed 20261022 -> 3 states.

RNG: numpy.default_rng(20261022) — declared in every output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEED = 20261022
T_GRID = [0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
N_BOOT = 2000
ALPHA = 0.05
COVERAGE_RISK_TARGET = 0.05
CANONICAL_T = ["G1", "L1", "L2", "L3", "L4", "L5"]
STRESS_T = ["G2", "X1", "X2", "X3", "X4"]
ALL_T = CANONICAL_T + STRESS_T
ENCODERS = ["siglip2", "clip"]
ENCODER_IDS = {"siglip2": "siglip2_I", "clip": "clip_I"}


# ── metrics (aligned with eval_metrics.py) ──────────────────────────────────

def _norm_correct(correct):
    return np.asarray(correct, dtype=np.int8).reshape(-1)


def risk_coverage(scores, correct, higher_better=True):
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    c = _norm_correct(correct)
    n = s.size
    if n == 0:
        return np.array([]), np.array([])
    if not higher_better:
        s = -s
    order = np.argsort(-s, kind="mergesort")
    c_ord = c[order]
    errors = np.cumsum(1 - c_ord)
    k = np.arange(1, n + 1, dtype=np.float64)
    return k / n, errors / k


def aurc(scores, correct, higher_better=True):
    _, risk = risk_coverage(scores, correct, higher_better)
    if risk.size == 0:
        return None
    return float(risk.mean())


def aurc_optimal(correct):
    c = _norm_correct(correct)
    n = c.size
    if n == 0:
        return None
    k_correct = int(c.sum())
    terms = np.zeros(n, dtype=np.float64)
    # first k_correct positions: 0 risk
    for j in range(n - k_correct):
        idx = k_correct + j
        terms[idx] = (1.0 + j) / (idx + 1.0)
    return float(terms.mean())


def e_aurc(scores, correct, higher_better=True):
    a = aurc(scores, correct, higher_better)
    if a is None:
        return None
    opt = aurc_optimal(correct)
    return float(a - opt)


def auroc(scores, correct, higher_better=True):
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    c = _norm_correct(correct)
    if not higher_better:
        s = -s
    n_pos = int(c.sum())
    n_neg = int(c.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    # average ranks for ties
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    rank_sum_pos = float(ranks[c == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def cp_upper(k, n, alpha=0.05):
    # Clopper-Pearson upper bound; bisection on regularized incomplete beta.
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    from math import lgamma, log, exp

    def betacf(a, b, x):
        tiny = 1e-300
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 301):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-16:
                break
        return h

    def beta_cdf(x, a, b):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        ln_front = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
        front = exp(ln_front)
        if x < (a + 1.0) / (a + b + 2.0):
            return front * betacf(a, b, x) / a
        return 1.0 - front * betacf(b, a, 1.0 - x) / b

    target = 1.0 - alpha
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, k + 1.0, n - k) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def coverage_at_risk(scores, correct, target_risk=0.05, higher_better=True):
    """Max coverage with realized risk <= target_risk (descriptive)."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    c = _norm_correct(correct)
    n = s.size
    if n == 0:
        return {"coverage": 0.0, "risk": 0.0, "n_accepted": 0, "n_errors": 0, "cp_upper": 1.0}
    if not higher_better:
        s = -s
    order = np.argsort(-s, kind="mergesort")
    c_ord = c[order]
    errors = np.cumsum(1 - c_ord)
    k = np.arange(1, n + 1)
    risk = errors / k
    ok = np.where(risk <= target_risk)[0]
    if ok.size == 0:
        return {"coverage": 0.0, "risk": 0.0, "n_accepted": 0, "n_errors": 0, "cp_upper": 1.0}
    best = int(ok[-1])
    n_acc = best + 1
    n_err = int(errors[best])
    return {
        "coverage": float(n_acc / n),
        "risk": float(risk[best]),
        "n_accepted": n_acc,
        "n_errors": n_err,
        "cp_upper": cp_upper(n_err, n_acc, alpha=0.05),
    }


# ── scoring ────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def msp_at_T(id_scores, T):
    p = softmax(id_scores / float(T), axis=1)
    return p.max(axis=1)


def energy_at_T(id_scores, T):
    # T*logsumexp(s/T), higher=confident
    s = np.asarray(id_scores, dtype=np.float64)
    z = s / float(T)
    m = np.max(z, axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return float(T) * lse


def pick_T(scores_fn, id_scores, correct, T_grid):
    best_T, best_e = None, None
    details = []
    for T in T_grid:
        sc = scores_fn(id_scores, T)
        e = e_aurc(sc, correct, higher_better=True)
        details.append({"T": T, "e_aurc_calib": e})
        if best_e is None or e < best_e - 1e-15 or (abs(e - best_e) <= 1e-15 and (best_T is None or T < best_T)):
            best_T, best_e = T, e
    return best_T, best_e, details


# ── bootstrap / Holm ───────────────────────────────────────────────────────

def cluster_boot_indices(test_identities, rng, n_boot):
    # test_identities: array of identity per query (length n_test)
    uniq = np.unique(test_identities)
    id_to_rows = {u: np.where(test_identities == u)[0] for u in uniq}
    n_u = len(uniq)
    for _ in range(n_boot):
        sample = rng.integers(0, n_u, size=n_u)
        rows = np.concatenate([id_to_rows[uniq[i]] for i in sample])
        yield rows


def paired_delta_eaurc(scores_a, scores_b, correct, higher_better=True):
    # delta = E-AURC(a) - E-AURC(b). negative => a better.
    ea = e_aurc(scores_a, correct, higher_better)
    eb = e_aurc(scores_b, correct, higher_better)
    return float(ea - eb), float(ea), float(eb)


def bootstrap_paired_delta(scores_gap, scores_other, correct, ids_per_query, rng, n_boot):
    delta_obs, ea, eb = paired_delta_eaurc(scores_gap, scores_other, correct)
    deltas = []
    for rows in cluster_boot_indices(ids_per_query, rng, n_boot):
        d, _, _ = paired_delta_eaurc(scores_gap[rows], scores_other[rows], correct[rows])
        deltas.append(d)
    deltas = np.asarray(deltas, dtype=np.float64)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p-value: 2*min(P(delta>=0), P(delta<=0)) using +1 smoothing
    p_le0 = (np.sum(deltas <= 0) + 1) / (n_boot + 1)
    p_ge0 = (np.sum(deltas >= 0) + 1) / (n_boot + 1)
    p_two = float(min(1.0, 2 * min(p_le0, p_ge0)))
    return {
        "delta_eaurc_gap_minus_other": float(delta_obs),
        "eaurc_gap": ea,
        "eaurc_other": eb,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "p_boot_two_sided": p_two,
        "n_boot": n_boot,
    }


def holm(pvals):
    # pvals: dict name->p
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        a = max(a, prev)
        adj[name] = a
        prev = a
    return adj


# ── P1 ─────────────────────────────────────────────────────────────────────

# Ab4 COARSE_MAP (tools/measure_retrieval_quality.py L450-465)
COARSE_MAP = {
    "Airplane": ["aircraft"],
    "Tank": ["tank"],
    "Vehicle": ["vehicle", "artillery", "air_defense", "missile"],
    "Helicopter": ["helicopter"],
    "Drone": ["drone"],
    "Ship": ["ship"],
}
CAT_TO_COARSE = {c: k for k, cs in COARSE_MAP.items() for c in cs}


def build_yc_indices(ids, id_to_cat):
    """Return dict coarse -> np.array of identity indices in `ids`."""
    yc = {}
    for coarse, cats in COARSE_MAP.items():
        idxs = [i for i, v in enumerate(ids) if CAT_TO_COARSE.get(id_to_cat.get(v), None) == coarse]
        yc[coarse] = np.array(idxs, dtype=np.int64)
    return yc


def sc4_inter_identity(id_scores, q_categories, yc_by_coarse):
    """Sc4 ab4/spec_v2 (measure_retrieval_quality.py L486-496):
    s_all[v]=max score per identity; Y_c identities in same coarse category;
    sv=sorted(s_y.values(), reverse=True); m_id = sv[0]-sv[1] (None if n_c<2).
    Returns (m_id array, n_c array)."""
    n = id_scores.shape[0]
    m_id = np.full(n, np.nan, dtype=np.float64)
    n_c = np.zeros(n, dtype=np.int32)
    for i in range(n):
        coarse = CAT_TO_COARSE.get(q_categories[i])
        idxs = yc_by_coarse.get(coarse)
        if idxs is None or idxs.size == 0:
            continue
        s_y = id_scores[i, idxs]
        n_c[i] = idxs.size
        if idxs.size < 2:
            continue
        sv = np.sort(s_y)[::-1]
        m_id[i] = float(sv[0] - sv[1])
    return m_id, n_c


def sanity_nc2_msp_gap(id_scores, q_categories, yc_by_coarse, T_msp):
    """Ab4 L349-356: Sc2≡Sc4 when n_c==2 (MSP=sigmoid(m/T) monotone in m)."""
    msp = msp_at_T(id_scores, T_msp)
    gap, n_c = sc4_inter_identity(id_scores, q_categories, yc_by_coarse)
    mask = n_c == 2
    idx = np.where(mask)[0]
    if idx.size == 0:
        return {"n_nc2": 0, "rank_equal": None, "note": "no n_c==2 queries"}
    msp2 = msp[idx]
    gap2 = gap[idx]
    # rank by descending score; check if same ordering (ties handled by stable sort)
    r_msp = np.argsort(-msp2, kind="mergesort")
    r_gap = np.argsort(-gap2, kind="mergesort")
    rank_equal = bool(np.array_equal(r_msp, r_gap))
    # also check pairwise concordance
    concordant = total = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            dm = msp2[a] - msp2[b]
            dg = gap2[a] - gap2[b]
            if dm == 0 or dg == 0:
                continue
            total += 1
            if (dm > 0) == (dg > 0):
                concordant += 1
    return {
        "n_nc2": int(idx.size),
        "rank_equal_full": rank_equal,
        "pairwise_concordant": concordant,
        "pairwise_total": total,
        "pairwise_concordance": (concordant / total if total else None),
        "T_msp": float(T_msp),
    }


def yc_top2_from_scores(id_scores_2d, q_categories, yc_by_coarse):
    """id_scores_2d: (n, n_ids). Returns (s1,s2,n_c) arrays length n."""
    n = id_scores_2d.shape[0]
    s1 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    n_c = np.zeros(n, dtype=np.int32)
    for i in range(n):
        coarse = CAT_TO_COARSE.get(q_categories[i])
        idxs = yc_by_coarse.get(coarse)
        if idxs is None or idxs.size == 0:
            continue
        vals = id_scores_2d[i, idxs]
        n_c[i] = idxs.size
        if idxs.size < 2:
            if idxs.size == 1:
                s1[i] = float(vals[0])
            continue
        sv = np.sort(vals)[::-1]
        s1[i] = float(sv[0])
        s2[i] = float(sv[1])
    return s1, s2, n_c


def compute_p1(pass1_npz, pass1_meta):
    rng = np.random.default_rng(SEED)
    meta = pass1_meta
    ids = meta["ids"]
    q_ids = np.array(meta["q_identities"])
    q_split = np.array(meta["q_split"])
    q_categories = list(meta["q_categories"])
    calib_mask = q_split == "calib"
    test_mask = q_split == "test"

    # identity -> category from pass1_meta
    id_to_cat = {}
    for qi, qc in zip(q_ids, q_categories):
        if qi in id_to_cat and id_to_cat[qi] != qc:
            raise ValueError(f"identity {qi} has conflicting categories")
        id_to_cat[qi] = qc
    missing = [v for v in ids if v not in id_to_cat]
    if missing:
        # fallback: load full canonical meta from server helpers
        try:
            import sys
            for p in ["/tmp/prereg_v7", "<REDACTED_PATH>/tools"]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            import tierA_smoke as T  # type: ignore
            _idx, _meta = T.load_meta_index("siglip2")
            for m in _meta:
                id_to_cat.setdefault(m["variant"], m["category"])
        except Exception as e:
            raise ValueError(f"missing id_to_cat for {len(missing)} ids: {missing[:5]}; fallback failed: {e}")
        missing = [v for v in ids if v not in id_to_cat]
        if missing:
            raise ValueError(f"still missing id_to_cat after fallback: {missing[:5]}")
    yc_by_coarse = build_yc_indices(ids, id_to_cat)

    out = {
        "prereg": "prereg_v9.md SEALED",
        "split_hash": meta["split_hash"],
        "seed": SEED,
        "rng_family": "numpy.default_rng(PCG64)",
        "n_boot": N_BOOT,
        "T_grid": T_GRID,
        "n_Dii": meta["n_Dii"],
        "n_calib_ids": meta["n_calib_ids"],
        "n_test_ids": meta["n_test_ids"],
        "sc4_definition": "ab4/spec_v2 (measure_retrieval_quality.py L486-496): inter-identity gap within Y_c. s_all[v]=max score per identity; s_y={y:s_all[y] for y in Y_c}; sv=sorted desc; Sc4=m_id=sv[0]-sv[1] (n_c<2 -> abstain/NaN). Y_c from COARSE_MAP same as ab4 L450-465.",
        "sc4_fix_note": "D12: v9 §1 mô tả nhầm 'ảnh cao nhì cùng identity' (intra-image). Clause quyết định 'đúng định nghĩa ab4/spec_v2 (Sc1-Sc4)' được giữ; Sc4 đã sửa về inter-identity gap trong Y_c. Kết quả intra-image cũ giữ làm *intra_image_SUPERSEDED*.",
        "coarse_map": COARSE_MAP,
        "encoders": {},
    }

    for enc in ENCODERS:
        eid = ENCODER_IDS[enc]
        id_scores = np.asarray(pass1_npz[f"{enc}_id_scores"], dtype=np.float64)
        correct = np.asarray(pass1_npz[f"{enc}_correct"], dtype=np.int8)

        # scores
        sc1_sim = id_scores.max(axis=1)
        sc4_gap, sc4_nc = sc4_inter_identity(id_scores, q_categories, yc_by_coarse)
        # abstain n_c<2: set to -inf so they rank worst (should be rare/none in D(ii))
        abstain = ~np.isfinite(sc4_gap)
        sc4_gap_safe = sc4_gap.copy()
        sc4_gap_safe[abstain] = -1e9

        # calib selection T*
        T_msp, e_msp_cal, det_msp = pick_T(msp_at_T, id_scores[calib_mask], correct[calib_mask], T_GRID)
        T_eng, e_eng_cal, det_eng = pick_T(energy_at_T, id_scores[calib_mask], correct[calib_mask], T_GRID)

        sc2_msp = msp_at_T(id_scores, T_msp)
        sc3_energy = energy_at_T(id_scores, T_eng)

        nc2_sanity = sanity_nc2_msp_gap(id_scores, q_categories, yc_by_coarse, T_msp)

        scores = {
            "sim": sc1_sim,
            "MSP": sc2_msp,
            "energy": sc3_energy,
            "gap": sc4_gap_safe,
        }

        # test metrics
        test_scores = {k: v[test_mask] for k, v in scores.items()}
        test_correct = correct[test_mask]
        test_ids = q_ids[test_mask]

        metrics = {}
        for name, sc in test_scores.items():
            metrics[name] = {
                "aurc": aurc(sc, test_correct, True),
                "e_aurc": e_aurc(sc, test_correct, True),
                "auroc": auroc(sc, test_correct, True),
                "coverage_at_risk_5pct": coverage_at_risk(sc, test_correct, COVERAGE_RISK_TARGET, True),
            }

        # paired delta E-AURC gap - {sim,MSP,energy}
        comparisons = {}
        pvals = {}
        for other in ["sim", "MSP", "energy"]:
            # new rng per comparison? No, use one rng stream sequentially - documented.
            # Better use independent sub-rngs for reproducibility clarity.
            cmp_rng = np.random.default_rng([SEED, hash((enc, other)) % (2**32)])
            r = bootstrap_paired_delta(test_scores["gap"], test_scores[other], test_correct, test_ids, cmp_rng, N_BOOT)
            comparisons[f"gap_minus_{other}"] = r
            pvals[f"gap_minus_{other}"] = r["p_boot_two_sided"]

        holm_adj = holm(pvals)
        for k in comparisons:
            comparisons[k]["holm_p_adj"] = holm_adj[k]
            ci = comparisons[k]
            # better if delta < 0 and CI excludes 0 (entirely below 0)
            if ci["ci95_hi"] < 0:
                label = "better"
            else:
                label = "insufficient"
            comparisons[k]["label"] = label

        out["encoders"][eid] = {
            "T_star_MSP": T_msp,
            "T_star_energy": T_eng,
            "calib_eaurc_MSP": e_msp_cal,
            "calib_eaurc_energy": e_eng_cal,
            "calib_T_details_MSP": det_msp,
            "calib_T_details_energy": det_eng,
            "n_test_queries": int(test_mask.sum()),
            "n_test_correct": int(test_correct.sum()),
            "test_accuracy": float(test_correct.mean()),
            "n_abstain_sc4": int(abstain.sum()),
            "nc2_sanity_msp_vs_gap": nc2_sanity,
            "metrics_test": metrics,
            "paired_delta_eaurc": comparisons,
            "holm_m": 3,
        }
    return out


# ── P2 ─────────────────────────────────────────────────────────────────────

def bootstrap_ratio_median(abs_dm, abs_ds, identities, rng, n_boot):
    # ratio R = median(|Δm_id|) / median(|Δs1|) per group, cluster bootstrap by identity
    uniq = np.unique(identities)
    id_to_rows = {u: np.where(identities == u)[0] for u in uniq}
    n_u = len(uniq)

    def stat(rows):
        m = float(np.median(abs_dm[rows]))
        s = float(np.median(abs_ds[rows]))
        r = (m / s) if s > 0 else (float("inf") if m > 0 else float("nan"))
        return r, m, s

    r_obs, m_obs, s_obs = stat(np.arange(len(abs_dm)))
    ratios = []
    for _ in range(n_boot):
        sample = rng.integers(0, n_u, size=n_u)
        rows = np.concatenate([id_to_rows[uniq[i]] for i in sample])
        r, _, _ = stat(rows)
        ratios.append(r)
    ratios = np.asarray(ratios, dtype=np.float64)
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        lo, hi = float("nan"), float("nan")
    else:
        lo, hi = np.percentile(finite, [2.5, 97.5])
    if np.isfinite(lo) and np.isfinite(hi):
        if hi < 1.0:
            state = "ỔN ĐỊNH HƠN"
        elif lo > 1.0:
            state = "KÉM ỔN ĐỊNH HƠN"
        else:
            state = "KHÔNG THẤY KHÁC"
    else:
        state = "KHÔNG XÁC ĐỊNH"
    return {
        "median_abs_delta_mid": m_obs,
        "median_abs_delta_s1": s_obs,
        "ratio_median": float(r_obs),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "state": state,
        "n_rows": int(len(abs_dm)),
        "n_boot": n_boot,
    }


def compute_p2(pass1_npz, pass1_meta, pass2_npz, pass2_meta):
    # P2 uses siglip2_I only, all 1667 queries x 11T
    meta2 = pass2_meta
    transforms = meta2["transforms"]
    assert transforms == ALL_T

    s1_id = np.asarray(pass2_npz["s1_id"], dtype=np.float64)  # (1667,11)
    s2_id = np.asarray(pass2_npz["s2_id"], dtype=np.float64)
    b_s1 = np.asarray(pass2_npz["b_s1_id"], dtype=np.float64)  # (1667,)
    b_s2 = np.asarray(pass2_npz["b_s2_id"], dtype=np.float64)
    gt_idx = np.asarray(pass2_npz["gt_idx"], dtype=np.int64)

    m_id = s1_id - s2_id
    b_mid = b_s1 - b_s2
    dm = m_id - b_mid[:, None]
    ds1 = s1_id - b_s1[:, None]

    # identities per query (canonical order)
    # pass1_meta has q paths/ids but only D(ii); pass2 is all 1667. Get from pass2? not stored.
    # We can reconstruct from pass1_meta? pass1_meta has ids list and q_idx; not full 1667.
    # But pass2 capture saved gt_idx only. We need identity labels for cluster bootstrap.
    # We'll load from pass1 meta is not enough. Need to get from server meta or infer.
    # Workaround: pass1_meta has 'ids' and pass2 gt_idx indexes into ids.
    ids = pass1_meta["ids"]
    identities = np.array([ids[i] for i in gt_idx])

    groups = {
        "canonical_6T": CANONICAL_T,
        "stress_5T": STRESS_T,
    }

    out = {
        "prereg": "prereg_v9.md SEALED",
        "seed": SEED,
        "rng_family": "numpy.default_rng(PCG64)",
        "n_boot": N_BOOT,
        "encoder": "siglip2_I",
        "n_queries": int(s1_id.shape[0]),
        "n_transforms": int(s1_id.shape[1]),
        "total": int(s1_id.shape[0] * s1_id.shape[1]),
        "transforms": transforms,
        "delta_definition": "delta = value(T(x)) - value(x), same component-masked A(ii) setting; m_id=s1_id-s2_id (identity max-pool), s1=top1 identity score",
        "d12_note": "D12 sửa Sc4 P1 sang inter-identity gap trong Y_c (ab4 L486-496). P2 capture (pass2_capture.npz) chỉ lưu scalar top-2 identity global, không có vector 145 identity theo transform; do đó P2 m_id ở đây vẫn là global identity top-2 (không Y_c-restricted). Không recapture theo chỉ đạo; cần recapture nếu muốn P2 Y_c-restricted.",
        "groups": {},
        "per_transform": {},
    }

    for gi, (gname, tlist) in enumerate(groups.items()):
        tidx = [transforms.index(t) for t in tlist]
        dm_g = dm[:, tidx].reshape(-1)
        ds1_g = ds1[:, tidx].reshape(-1)
        id_g = np.repeat(identities, len(tidx))
        abs_dm = np.abs(dm_g)
        abs_ds1 = np.abs(ds1_g)
        rng = np.random.default_rng([SEED, 100 + gi])
        stat = bootstrap_ratio_median(abs_dm, abs_ds1, id_g, rng, N_BOOT)
        stat["p95_abs_delta_mid"] = float(np.percentile(abs_dm, 95))
        stat["p95_abs_delta_s1"] = float(np.percentile(abs_ds1, 95))
        stat["transforms"] = tlist
        out["groups"][gname] = stat

    for ti, t in enumerate(transforms):
        abs_dm = np.abs(dm[:, ti])
        abs_ds1 = np.abs(ds1[:, ti])
        rng = np.random.default_rng([SEED, 200 + ti])
        stat = bootstrap_ratio_median(abs_dm, abs_ds1, identities, rng, N_BOOT)
        stat["p95_abs_delta_mid"] = float(np.percentile(abs_dm, 95))
        stat["p95_abs_delta_s1"] = float(np.percentile(abs_ds1, 95))
        out["per_transform"][t] = stat

    return out


def compute_p2_yc(pass2b_npz, pass2b_meta):
    """P2 with Y_c-restricted inter-identity gap (D12 fix) from pass2b vectors."""
    meta2 = pass2b_meta
    transforms = meta2["transforms"]
    assert transforms == ALL_T
    ids = meta2["ids"]
    q_identities = meta2["q_identities"]
    q_categories = meta2["q_categories"]
    id_to_cat = {v: c for v, c in zip(q_identities, q_categories)}
    missing = [v for v in ids if v not in id_to_cat]
    if missing:
        # should not happen; pass2b meta has all 1667
        raise ValueError(f"missing id_to_cat in pass2b: {missing[:5]}")
    yc_by_coarse = build_yc_indices(ids, id_to_cat)

    id_scores_all = np.asarray(pass2b_npz["id_scores_all"], dtype=np.float64)  # (n,nT,145)
    b_id_scores = np.asarray(pass2b_npz["b_id_scores"], dtype=np.float64)      # (n,145)
    gt_idx = np.asarray(pass2b_npz["gt_idx"], dtype=np.int64)
    n, nT, n_ids = id_scores_all.shape

    b_s1 = np.full(n, np.nan, dtype=np.float64)
    b_s2 = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        coarse = CAT_TO_COARSE.get(q_categories[i])
        idxs = yc_by_coarse.get(coarse)
        if idxs is None or idxs.size < 2:
            continue
        sv = np.sort(b_id_scores[i, idxs])[::-1]
        b_s1[i] = float(sv[0]); b_s2[i] = float(sv[1])

    s1_id = np.full((n, nT), np.nan, dtype=np.float64)
    s2_id = np.full((n, nT), np.nan, dtype=np.float64)
    for i in range(n):
        coarse = CAT_TO_COARSE.get(q_categories[i])
        idxs = yc_by_coarse.get(coarse)
        if idxs is None or idxs.size < 2:
            continue
        for ti in range(nT):
            sv = np.sort(id_scores_all[i, ti, idxs])[::-1]
            s1_id[i, ti] = float(sv[0]); s2_id[i, ti] = float(sv[1])

    b_mid = b_s1 - b_s2
    m_id = s1_id - s2_id
    dm = m_id - b_mid[:, None]
    # s1 for P2 is top1 identity score (absolute) - keep global top1 identity? prereg: "Δs₁ = s₁(T(x)) − s₁(x) (s₁/s₂ theo ảnh trong Y_c như P1)" - ambiguous. But s1 is "absolute similarity" top1 identity. Use global top1 identity for s1 as before, to measure stability of absolute confidence; m_id is the gap.
    # use global s1 from scalars
    b_s1_global = np.asarray(pass2b_npz["b_s1_id"], dtype=np.float64)
    s1_global = np.asarray(pass2b_npz["s1_id"], dtype=np.float64)
    ds1 = s1_global - b_s1_global[:, None]

    identities = np.array([ids[i] for i in gt_idx])
    groups = {"canonical_6T": CANONICAL_T, "stress_5T": STRESS_T}

    out = {
        "prereg": "prereg_v9.md SEALED",
        "seed": SEED,
        "rng_family": "numpy.default_rng(PCG64)",
        "n_boot": N_BOOT,
        "encoder": "siglip2_I",
        "n_queries": int(n),
        "n_transforms": int(nT),
        "total": int(n * nT),
        "transforms": transforms,
        "delta_definition": "delta = value(T(x)) - value(x), same component-masked A(ii) setting; m_id = Y_c-restricted inter-identity gap sv[0]-sv[1] (ab4 L486-496, D12 fix); s1 = global top1 identity score (absolute).",
        "sc4_definition": "Y_c-restricted inter-identity gap: s_all[v]=max per identity; sv sorted desc within Y_c (COARSE_MAP ab4); m_id=sv[0]-sv[1].",
        "d12_note": "D12-part2: recomputed from pass2b_capture.npz (id_scores_all 1667x11x145). Global-identity-gap P2 kept as sensitivity in p2_results_global_sensitivity.json / report section.",
        "groups": {},
        "per_transform": {},
    }

    for gi, (gname, tlist) in enumerate(groups.items()):
        tidx = [transforms.index(t) for t in tlist]
        dm_g = dm[:, tidx].reshape(-1)
        ds1_g = ds1[:, tidx].reshape(-1)
        id_g = np.repeat(identities, len(tidx))
        # drop NaN (n_c<2 abstain)
        ok = np.isfinite(dm_g) & np.isfinite(ds1_g)
        dm_g, ds1_g, id_g = dm_g[ok], ds1_g[ok], id_g[ok]
        abs_dm = np.abs(dm_g)
        abs_ds1 = np.abs(ds1_g)
        rng = np.random.default_rng([SEED, 100 + gi])
        stat = bootstrap_ratio_median(abs_dm, abs_ds1, id_g, rng, N_BOOT)
        stat["p95_abs_delta_mid"] = float(np.percentile(abs_dm, 95)) if abs_dm.size else float("nan")
        stat["p95_abs_delta_s1"] = float(np.percentile(abs_ds1, 95)) if abs_ds1.size else float("nan")
        stat["n_abstain_nc_lt2"] = int((~ok).sum())
        stat["transforms"] = tlist
        out["groups"][gname] = stat

    for ti, t in enumerate(transforms):
        dm_t = dm[:, ti]
        ds1_t = ds1[:, ti]
        ok = np.isfinite(dm_t) & np.isfinite(ds1_t)
        abs_dm = np.abs(dm_t[ok])
        abs_ds1 = np.abs(ds1_t[ok])
        id_t = identities[ok]
        rng = np.random.default_rng([SEED, 200 + ti])
        stat = bootstrap_ratio_median(abs_dm, abs_ds1, id_t, rng, N_BOOT)
        stat["p95_abs_delta_mid"] = float(np.percentile(abs_dm, 95)) if abs_dm.size else float("nan")
        stat["p95_abs_delta_s1"] = float(np.percentile(abs_ds1, 95)) if abs_ds1.size else float("nan")
        stat["n_abstain_nc_lt2"] = int((~ok).sum())
        out["per_transform"][t] = stat

    return out


# ── reports ────────────────────────────────────────────────────────────────

def fmt(x, nd=6):
    if x is None:
        return "NA"
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.{nd}f}"
    return str(x)


def write_p1_report(p1, path):
    lines = []
    lines.append("# prereg_v9 — Report P1 (SEALED)")
    lines.append("")
    lines.append(f"- prereg: `{p1['prereg']}`")
    lines.append(f"- split_hash: `{p1['split_hash']}`")
    lines.append(f"- seed: {p1['seed']} | RNG: `{p1['rng_family']}` | n_boot: {p1['n_boot']}")
    lines.append(f"- T grid: {p1['T_grid']}")
    lines.append(f"- D(ii): {p1['n_Dii']} queries | calib ids: {p1['n_calib_ids']} | test ids: {p1['n_test_ids']}")
    lines.append(f"- Sc4 definition: {p1['sc4_definition']}")
    if p1.get("sc4_fix_note"):
        lines.append(f"- Sc4 fix note (D12): {p1['sc4_fix_note']}")
    lines.append("")
    lines.append("## Sanity n_c==2: MSP(T*) vs Sc4 (ab4 L349-356, sigmoid(m/T) monotone)")
    lines.append("")
    lines.append("| encoder | n_nc2 | rank_equal_full | pairwise_concordant/total | concordance | T_MSP | n_abstain_sc4 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for eid, ed in p1["encoders"].items():
        s = ed.get("nc2_sanity_msp_vs_gap", {})
        conc = s.get("pairwise_concordance")
        conc_s = fmt(conc) if conc is not None else "NA"
        lines.append(f"| {eid} | {s.get('n_nc2','NA')} | {s.get('rank_equal_full','NA')} | {s.get('pairwise_concordant','NA')}/{s.get('pairwise_total','NA')} | {conc_s} | {fmt(s.get('T_msp'))} | {ed.get('n_abstain_sc4','NA')} |")
    lines.append("")
    lines.append("## Bảng metrics test (4 score × 2 encoder)")
    lines.append("")
    lines.append("| encoder | score | AURC | E-AURC | AUROC | cov@risk5% | risk@cov | n_acc | n_err | cp_upper |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for eid, ed in p1["encoders"].items():
        for sc in ["sim", "MSP", "energy", "gap"]:
            m = ed["metrics_test"][sc]
            c5 = m["coverage_at_risk_5pct"]
            lines.append(f"| {eid} | {sc} | {fmt(m['aurc'])} | {fmt(m['e_aurc'])} | {fmt(m['auroc'])} | {fmt(c5['coverage'])} | {fmt(c5['risk'])} | {c5['n_accepted']} | {c5['n_errors']} | {fmt(c5['cp_upper'])} |")
    lines.append("")
    lines.append("## Paired ΔE-AURC (gap − other), cluster bootstrap identity, Holm 3")
    lines.append("")
    lines.append("| encoder | comparison | ΔE-AURC | 95% CI | p_boot | Holm p | label |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for eid, ed in p1["encoders"].items():
        for k, c in ed["paired_delta_eaurc"].items():
            ci = f"[{fmt(c['ci95_lo'])}, {fmt(c['ci95_hi'])}]"
            lines.append(f"| {eid} | {k} | {fmt(c['delta_eaurc_gap_minus_other'])} | {ci} | {fmt(c['p_boot_two_sided'])} | {fmt(c['holm_p_adj'])} | {c['label']} |")
    lines.append("")
    lines.append("## T* calib")
    lines.append("")
    for eid, ed in p1["encoders"].items():
        lines.append(f"- {eid}: T*_MSP={ed['T_star_MSP']} (calib E-AURC={fmt(ed['calib_eaurc_MSP'])}), T*_energy={ed['T_star_energy']} (calib E-AURC={fmt(ed['calib_eaurc_energy'])}), test n={ed['n_test_queries']}, test acc={fmt(ed['test_accuracy'])}")
    lines.append("")
    lines.append("## Ghi chú")
    lines.append("")
    lines.append("- ΔE-AURC = E-AURC(gap) − E-AURC(other); âm = gap tốt hơn (E-AURC thấp hơn).")
    lines.append("- Label theo prereg v9: `better` nếu CI95% loại trừ 0 về phía tốt hơn (ci95_hi < 0); ngược lại `insufficient`. Không dùng δ mới.")
    lines.append("- Bootstrap cluster theo identity test, 2000 lần, seed 20261022, sub-RNG per comparison.")
    lines.append("- D12: Sc4 đã sửa từ intra-image (ảnh cao nhì cùng identity) sang inter-identity gap trong Y_c theo ab4 L486-496. Kết quả intra-image cũ được giữ làm `*intra_image_SUPERSEDED*`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_p2_report(p2, path):
    lines = []
    lines.append("# prereg_v9 — Report P2 (SEALED)")
    lines.append("")
    lines.append(f"- prereg: `{p2['prereg']}`")
    lines.append(f"- seed: {p2['seed']} | RNG: `{p2['rng_family']}` | n_boot: {p2['n_boot']}")
    lines.append(f"- encoder: {p2['encoder']} | queries: {p2['n_queries']} | transforms: {p2['n_transforms']} | total: {p2['total']}")
    lines.append(f"- delta: {p2['delta_definition']}")
    if p2.get("d12_note"):
        lines.append(f"- D12 note: {p2['d12_note']}")
    lines.append("")
    lines.append("## Nhóm (canonical 6T / stress 5T)")
    lines.append("")
    lines.append("| group | median|Δm_id| | median|Δs1| | P95 |Δm_id| | P95 |Δs1| | ratio median | 95% CI ratio | state | n_rows |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for g, s in p2["groups"].items():
        ci = f"[{fmt(s['ci95_lo'])}, {fmt(s['ci95_hi'])}]"
        lines.append(f"| {g} | {fmt(s['median_abs_delta_mid'])} | {fmt(s['median_abs_delta_s1'])} | {fmt(s['p95_abs_delta_mid'])} | {fmt(s['p95_abs_delta_s1'])} | {fmt(s['ratio_median'])} | {ci} | {s['state']} | {s['n_rows']} |")
    lines.append("")
    lines.append("## Per-transform")
    lines.append("")
    lines.append("| T | median|Δm_id| | median|Δs1| | ratio | 95% CI | state |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for t, s in p2["per_transform"].items():
        ci = f"[{fmt(s['ci95_lo'])}, {fmt(s['ci95_hi'])}]"
        lines.append(f"| {t} | {fmt(s['median_abs_delta_mid'])} | {fmt(s['median_abs_delta_s1'])} | {fmt(s['ratio_median'])} | {ci} | {s['state']} |")
    lines.append("")
    lines.append("## Ghi chú")
    lines.append("")
    lines.append("- Ratio = median(|Δm_id|) / median(|Δs1|). CI cluster bootstrap theo identity, 2000 lần, seed 20261022.")
    lines.append("- Trạng thái prereg v9: CI < 1 → `ỔN ĐỊNH HƠN`; CI chứa 1 → `KHÔNG THẤY KHÁC`; CI > 1 → `KÉM ỔN ĐỊNH HƠN`.")
    lines.append("- Mô tả, không p-value.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_p2_report_combined(p2_yc, p2_global, path):
    lines = []
    lines.append("# prereg_v9 — Report P2 (SEALED, D12-part2)")
    lines.append("")
    lines.append(f"- prereg: `{p2_yc['prereg']}`")
    lines.append(f"- seed: {p2_yc['seed']} | RNG: `{p2_yc['rng_family']}` | n_boot: {p2_yc['n_boot']}")
    lines.append(f"- encoder: {p2_yc['encoder']} | queries: {p2_yc['n_queries']} | transforms: {p2_yc['n_transforms']} | total: {p2_yc['total']}")
    lines.append(f"- chính (Y_c-restricted): {p2_yc['delta_definition']}")
    if p2_yc.get("d12_note"):
        lines.append(f"- D12-part2: {p2_yc['d12_note']}")
    lines.append("")
    lines.append("## Biến thể CHÍNH — Y_c-restricted inter-identity gap (ab4 L486-496)")
    lines.append("")
    lines.append("| group | median|Δm_id| | median|Δs1| | P95 |Δm_id| | P95 |Δs1| | ratio median | 95% CI ratio | state | n_rows | n_abstain_nc_lt2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for g, s in p2_yc["groups"].items():
        ci = f"[{fmt(s['ci95_lo'])}, {fmt(s['ci95_hi'])}]"
        lines.append(f"| {g} | {fmt(s['median_abs_delta_mid'])} | {fmt(s['median_abs_delta_s1'])} | {fmt(s['p95_abs_delta_mid'])} | {fmt(s['p95_abs_delta_s1'])} | {fmt(s['ratio_median'])} | {ci} | {s['state']} | {s['n_rows']} | {s.get('n_abstain_nc_lt2','NA')} |")
    lines.append("")
    lines.append("### Per-transform (Y_c-restricted)")
    lines.append("")
    lines.append("| T | median|Δm_id| | median|Δs1| | ratio | 95% CI | state | n_abstain_nc_lt2 |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for t, s in p2_yc["per_transform"].items():
        ci = f"[{fmt(s['ci95_lo'])}, {fmt(s['ci95_hi'])}]"
        lines.append(f"| {t} | {fmt(s['median_abs_delta_mid'])} | {fmt(s['median_abs_delta_s1'])} | {fmt(s['ratio_median'])} | {ci} | {s['state']} | {s.get('n_abstain_nc_lt2','NA')} |")
    lines.append("")
    lines.append("## Biến thể SENSITIVITY — global identity top-2 (pass2 cũ, pre-D12)")
    lines.append("")
    lines.append("| group | median|Δm_id| | median|Δs1| | P95 |Δm_id| | P95 |Δs1| | ratio median | 95% CI ratio | state | n_rows |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for g, s in p2_global["groups"].items():
        ci = f"[{fmt(s['ci95_lo'])}, {fmt(s['ci95_hi'])}]"
        lines.append(f"| {g} | {fmt(s['median_abs_delta_mid'])} | {fmt(s['median_abs_delta_s1'])} | {fmt(s['p95_abs_delta_mid'])} | {fmt(s['p95_abs_delta_s1'])} | {fmt(s['ratio_median'])} | {ci} | {s['state']} | {s['n_rows']} |")
    lines.append("")
    lines.append("### Per-transform (global sensitivity)")
    lines.append("")
    lines.append("| T | median|Δm_id| | median|Δs1| | ratio | 95% CI | state |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for t, s in p2_global["per_transform"].items():
        ci = f"[{fmt(s['ci95_lo'])}, {fmt(s['ci95_hi'])}]"
        lines.append(f"| {t} | {fmt(s['median_abs_delta_mid'])} | {fmt(s['median_abs_delta_s1'])} | {fmt(s['ratio_median'])} | {ci} | {s['state']} |")
    lines.append("")
    lines.append("## So sánh trạng thái (Y_c-restricted vs global)")
    lines.append("")
    lines.append("| group | Y_c ratio [CI] state | global ratio [CI] state | đổi trạng thái? |")
    lines.append("|---|---|---|---|")
    for g in p2_yc["groups"]:
        sy = p2_yc["groups"][g]; sg = p2_global["groups"][g]
        cy = f"{fmt(sy['ratio_median'])} [{fmt(sy['ci95_lo'])},{fmt(sy['ci95_hi'])}] {sy['state']}"
        cg = f"{fmt(sg['ratio_median'])} [{fmt(sg['ci95_lo'])},{fmt(sg['ci95_hi'])}] {sg['state']}"
        changed = "CÓ" if sy['state'] != sg['state'] else "KHÔNG"
        lines.append(f"| {g} | {cy} | {cg} | {changed} |")
    lines.append("")
    lines.append("## Ghi chú")
    lines.append("")
    lines.append("- Ratio = median(|Δm_id|) / median(|Δs1|). CI cluster bootstrap theo identity, 2000 lần, seed 20261022.")
    lines.append("- Trạng thái prereg v9: CI < 1 → `ỔN ĐỊNH HƠN`; CI chứa 1 → `KHÔNG THẤY KHÁC`; CI > 1 → `KÉM ỔN ĐỊNH HƠN`.")
    lines.append("- Biến thể chính dùng pass2b_capture.npz (id_scores_all 1667x11x145) để tính m_id Y_c-restricted. Biến thể global dùng scalar pass2 cũ (top-2 identity global).")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/prereg_v7/prereg_v9_gap_20261121")
    args = ap.parse_args()
    d = Path(args.dir)
    p1_npz = np.load(d / "pass1_capture.npz", allow_pickle=False)
    p1_meta = json.loads((d / "pass1_meta.json").read_text(encoding="utf-8"))
    p2_npz = np.load(d / "pass2_capture.npz", allow_pickle=False)
    p2_meta = json.loads((d / "pass2_meta.json").read_text(encoding="utf-8"))

    p1 = compute_p1(p1_npz, p1_meta)
    p2_global = compute_p2(p1_npz, p1_meta, p2_npz, p2_meta)

    (d / "p1_results.json").write_text(json.dumps(p1, ensure_ascii=False, indent=2), encoding="utf-8")
    write_p1_report(p1, d / "report_P1.md")

    out = {
        "ok": True,
        "p1_file": str(d / "p1_results.json"),
        "sha256": str(d / "SHA256SUMS.txt"),
    }

    p2b_npz_path = d / "pass2b_capture.npz"
    p2b_meta_path = d / "pass2b_meta.json"
    if p2b_npz_path.exists() and p2b_meta_path.exists():
        p2b_npz = np.load(p2b_npz_path, allow_pickle=False)
        p2b_meta = json.loads(p2b_meta_path.read_text(encoding="utf-8"))
        p2_yc = compute_p2_yc(p2b_npz, p2b_meta)
        (d / "p2b_results.json").write_text(json.dumps(p2_yc, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "p2_results_global_sensitivity.json").write_text(json.dumps(p2_global, ensure_ascii=False, indent=2), encoding="utf-8")
        write_p2_report_combined(p2_yc, p2_global, d / "report_P2.md")
        out["p2b_file"] = str(d / "p2b_results.json")
        out["p2_global_file"] = str(d / "p2_results_global_sensitivity.json")
    else:
        (d / "p2_results.json").write_text(json.dumps(p2_global, ensure_ascii=False, indent=2), encoding="utf-8")
        write_p2_report(p2_global, d / "report_P2.md")
        out["p2_file"] = str(d / "p2_results.json")

    # sha256 for key outputs in dir (excluding SHA256SUMS itself if exists)
    files = sorted([p for p in d.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"])
    lines = [f"{sha256_file(p)}  {p.name}" for p in files]
    (d / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
