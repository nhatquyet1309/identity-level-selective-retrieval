#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ-1 + McNemar §4.7 — prereg_v7 SEALED e5080142…74d1d. Stdlib-only, local.

Output dir: results/rq1_20261119/
Inputs: results/preregv7_tierA_20261117/loo_Aii.csv, loo_Ai.csv, identity_audit.json
Harness: scripts/rq1_harness.py, scripts/tierB_part2.py (reused, not rewritten).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 03_experiments
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rq1_harness import (  # noqa: E402
    C_MIN, DELTA_STAR, EPS_P, P_STAR, SEED, N_BOOT,
    auroc, auroc_ci, cp_upper, ks_distance, make_split,
    meets_coverage_floor, percentile, precision_at_tau, coverage_at_tau,
    precision_coverage_curve, score_distribution, tau_sensitivity_v2,
    tau_star, transfer, verdict_coverage, verdict_direction,
)
from tierB_part2 import ci95, cluster_draws  # noqa: E402

TIER_A = ROOT / "results" / "preregv7_tierA_20261117"
OUT = ROOT / "results" / "rq1_20261119"

CELLS = ["siglip2_S", "siglip2_I", "clip_S", "clip_I", "dinov2", "clipl14"]
CASE_I_CELLS = ["siglip2_I", "clip_I", "dinov2", "clipl14"]
CASE_S_CELLS = ["siglip2_S", "clip_S"]


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_dii(rows_aii):
    out = []
    for r in rows_aii:
        if r.get("in_D_Aii") != "1":
            continue
        item = {
            "path": r["path"],
            "identity": r["identity"],
            "category": r.get("category"),
        }
        for cell in CELLS:
            item[f"{cell}_correct"] = int(r[f"{cell}_Aii_correct"])
            item[f"{cell}_score"] = float(r[f"{cell}_Aii_score"])
            item[f"{cell}_top1"] = r[f"{cell}_Aii_top1"]
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Cluster helpers (145 identities universe, D3) — resample over fixed list
# ---------------------------------------------------------------------------

def build_groups(rows, all_identities):
    groups = {name: [] for name in all_identities}
    for i, r in enumerate(rows):
        groups[r["identity"]].append(i)
    return groups


def bootstrap_cluster_indices(groups, names, n_boot=N_BOOT, seed=SEED):
    draws, n = cluster_draws(len(names), n_boot=n_boot, seed=seed)
    for b in range(n_boot):
        idx = []
        base = b * n
        for j in range(n):
            idx.extend(groups[names[draws[base + j]]])
        yield idx


# ---------------------------------------------------------------------------
# §4.7 McNemar
# ---------------------------------------------------------------------------

def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial on discordant pairs), stdlib math.comb."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_for_pair(rows, all_identities, enc_prefix):
    """enc_prefix 'siglip2' or 'clip': compare S vs I on D(ii)."""
    s_key = f"{enc_prefix}_S_correct"
    i_key = f"{enc_prefix}_I_correct"
    # 2x2: rows = S outcome, cols = I outcome
    ss = si = is_ = ii = 0
    for r in rows:
        sc, ic = r[s_key], r[i_key]
        if sc == 1 and ic == 1:
            ss += 1
        elif sc == 1 and ic == 0:
            si += 1  # b: S correct, I wrong
        elif sc == 0 and ic == 1:
            is_ += 1  # c: S wrong, I correct
        else:
            ii += 1
    b = si
    c = is_
    n = len(rows)
    acc_s = sum(r[s_key] for r in rows) / n
    acc_i = sum(r[i_key] for r in rows) / n
    delta = acc_i - acc_s  # I - S

    # Paired bootstrap identity-clustered (145 cụm, D3)
    groups = build_groups(rows, all_identities)
    names = list(groups.keys())  # fixed 145
    reps = []
    for draw in bootstrap_cluster_indices(groups, names, n_boot=N_BOOT, seed=SEED):
        if not draw:
            continue
        ds = sum(rows[i][s_key] for i in draw) / len(draw)
        di = sum(rows[i][i_key] for i in draw) / len(draw)
        reps.append(di - ds)
    if not reps:
        raise ValueError(f"no valid bootstrap reps for {enc_prefix}")
    out = {
        "encoder": enc_prefix,
        "n": n,
        "n_identities_universe": len(all_identities),
        "table_2x2": {
            "S_correct_I_correct": ss,
            "S_correct_I_wrong": si,
            "S_wrong_I_correct": is_,
            "S_wrong_I_wrong": ii,
        },
        "b_S_correct_I_wrong": b,
        "c_S_wrong_I_correct": c,
        "n_discordant": b + c,
        "acc_S": acc_s,
        "acc_I": acc_i,
        "delta_I_minus_S": delta,
        "delta_I_minus_S_pp": 100.0 * delta,
        "mcnemar_exact_p_two_sided": mcnemar_exact(b, c),
        "paired_bootstrap_identity": {
            "seed": SEED,
            "n_boot": len(reps),
            "n_clusters": len(names),
            "cluster_unit": "identity (145 cụm, D3)",
            "ci95_delta": ci95(reps),
            "ci95_delta_pp": [100.0 * v for v in ci95(reps)],
        },
    }
    return out


# ---------------------------------------------------------------------------
# §7.2 metrics per cell
# ---------------------------------------------------------------------------

def cell_rows(rows, cell):
    return {
        "path": [r["path"] for r in rows],
        "identity": [r["identity"] for r in rows],
        "score": [r[f"{cell}_score"] for r in rows],
        "correct": [r[f"{cell}_correct"] for r in rows],
    }


def metrics_72(rows, cell, all_identities):
    d = cell_rows(rows, cell)
    dist = score_distribution(d["correct"], d["score"])
    # AUROC + CI bootstrap over 145 identities
    groups = build_groups(rows, all_identities)
    names = list(groups.keys())
    labels = d["correct"]
    scores = d["score"]
    auc_point = auroc(labels, scores)
    reps = []
    for draw in bootstrap_cluster_indices(groups, names, n_boot=N_BOOT, seed=SEED):
        if not draw:
            continue
        v = auroc([labels[i] for i in draw], [scores[i] for i in draw])
        if v is not None:
            reps.append(v)
    auc_res = {
        "auroc": auc_point,
        "ci95": ci95(reps) if reps else None,
        "n_boot": len(reps),
        "seed": SEED,
        "n_clusters": len(names),
        "cluster_unit": "identity (145 cụm, D3)",
    }
    curve = precision_coverage_curve(d["correct"], d["score"], keys=d["path"])
    # drop risk_at_k from stored curve (large) but keep aurc
    curve_store = {k: v for k, v in curve.items() if k != "risk_at_k"}
    return {
        "cell": cell,
        "n": len(rows),
        "distribution": dist,
        "auroc": auc_res,
        "aurc": curve["aurc"],
        "curve": curve_store,
        "curve_full_risks_count": len(curve["risk_at_k"]),
    }


# ---------------------------------------------------------------------------
# §7.3 tau* + evaluation
# ---------------------------------------------------------------------------

def split_rows(rows, calib_set, test_set):
    calib = [r for r in rows if r["identity"] in calib_set]
    test = [r for r in rows if r["identity"] in test_set]
    return calib, test


def bootstrap_metric_at_tau(rows, all_identities, score_key, correct_key, tau, metric_fn):
    # metric_fn(sub_rows) -> value (or None)
    groups = build_groups(rows, all_identities)
    names = list(groups.keys())
    point = metric_fn(rows)
    reps = []
    for draw in bootstrap_cluster_indices(groups, names, n_boot=N_BOOT, seed=SEED):
        if not draw:
            continue
        sub = [rows[i] for i in draw]
        v = metric_fn(sub)
        if v is not None:
            reps.append(v)
    return {"value": point, "ci95": ci95(reps) if reps else None,
            "n_boot": len(reps), "seed": SEED, "n_clusters": len(names),
            "cluster_unit": "identity (145 cụm, D3)"}


def eval_cell_73(rows_dii, cell, calib_ids, test_ids, all_identities):
    calib_set, test_set = set(calib_ids), set(test_ids)
    calib_rows, test_rows = split_rows(rows_dii, calib_set, test_set)
    sc_k = f"{cell}_score"
    cc_k = f"{cell}_correct"

    calib_pairs = [(r[sc_k], r[cc_k]) for r in calib_rows]
    ts = tau_star(calib_pairs, p_star=P_STAR)
    s1 = tau_sensitivity_v2(calib_pairs, p_star=P_STAR)

    if not ts["feasible"] or ts["tau"] is None:
        # infeasible on calib
        test_prec_point = None
        test_cov_point = coverage_at_tau([r[sc_k] for r in test_rows], float("inf"))
        test_eval = {
            "tau": None,
            "feasible": False,
            "precision": {"value": None, "ci95": None},
            "coverage": {"value": 0.0, "ci95": [0.0, 0.0]},
            "meets_c_min": False,
            "note": "KHÔNG ĐẠT p* (infeasible trên calib)"
        }
    else:
        tau = ts["tau"]
        test_scores = [r[sc_k] for r in test_rows]
        test_correct = [r[cc_k] for r in test_rows]

        def prec_fn(sub):
            return precision_at_tau([r[sc_k] for r in sub], [r[cc_k] for r in sub], tau)

        def cov_fn(sub):
            return coverage_at_tau([r[sc_k] for r in sub], tau)

        prec_boot = bootstrap_metric_at_tau(test_rows, all_identities, sc_k, cc_k, tau, prec_fn)
        cov_boot = bootstrap_metric_at_tau(test_rows, all_identities, sc_k, cc_k, tau, cov_fn)
        test_eval = {
            "tau": tau,
            "feasible": True,
            "precision": prec_boot,
            "coverage": cov_boot,
            "meets_c_min": meets_coverage_floor(cov_boot["value"], C_MIN) if cov_boot["value"] is not None else False,
        }

    return {
        "cell": cell,
        "n_calib_rows": len(calib_rows),
        "n_test_rows": len(test_rows),
        "tau_star_cp": ts,
        "sensitivity_S1_tau": s1,
        "test_at_tau_star": test_eval,
    }


def d_cov_pair(rows_test, all_identities, cell1, cell2, tau1, tau2):
    """D_cov = coverage1(tau1) - coverage2(tau2) on same test rows, paired bootstrap."""
    s1, c1 = f"{cell1}_score", f"{cell1}_correct"
    s2, c2 = f"{cell2}_score", f"{cell2}_correct"
    # same rows set, evaluate each at its own tau
    if tau1 is None or tau2 is None:
        # cannot compute meaningful D_cov for (a) verdict in the usual way
        return {"d_cov_pp": None, "ci95_pp": None, "note": "ít nhất một ô infeasible / không có tau"}

    def cov1(sub):
        return coverage_at_tau([r[s1] for r in sub], tau1)

    def cov2(sub):
        return coverage_at_tau([r[s2] for r in sub], tau2)

    point = 100.0 * (cov1(rows_test) - cov2(rows_test))
    groups = build_groups(rows_test, all_identities)
    names = list(groups.keys())
    reps = []
    for draw in bootstrap_cluster_indices(groups, names, n_boot=N_BOOT, seed=SEED):
        if not draw:
            continue
        sub = [rows_test[i] for i in draw]
        reps.append(100.0 * (cov1(sub) - cov2(sub)))
    ci = ci95(reps) if reps else None
    return {
        "d_cov_pp": point,
        "ci95_pp": ci,
        "verdict_a": verdict_coverage(ci, DELTA_STAR) if ci else None,
        "n_boot": len(reps),
        "seed": SEED,
        "n_clusters": len(names),
        "cluster_unit": "identity (145 cụm, D3)",
    }


# ---------------------------------------------------------------------------
# §7.4 / §7.5
# ---------------------------------------------------------------------------

def direction_transfer(rows_eval, all_identities, cell_j, cell_k, tau_k):
    """Apply tau_k of cell_k onto both on same rows_eval: D_prec/D_cov j-k at tau_k."""
    sj, cj = [r[f"{cell_j}_score"] for r in rows_eval], [r[f"{cell_j}_correct"] for r in rows_eval]
    sk, ck = [r[f"{cell_k}_score"] for r in rows_eval], [r[f"{cell_k}_correct"] for r in rows_eval]
    clusters = [r["identity"] for r in rows_eval]
    if tau_k is None:
        return {"d_prec_pp": None, "d_cov_pp": None, "ci_prec_pp": None, "ci_cov_pp": None,
                "verdict": None, "note": "nguồn tau_k infeasible"}
    return transfer(tau_k, sj, cj, sk, ck, clusters=clusters, n_boot=N_BOOT, seed=SEED,
                    eps_p=EPS_P, delta=DELTA_STAR)


def verdict_b(dir_k_to_j, dir_j_to_k):
    """(b) đạt nếu cả hai chiều đạt; hỏng nếu có một chiều hỏng; còn lại không đủ."""
    v1 = dir_k_to_j.get("verdict")
    v2 = dir_j_to_k.get("verdict")
    if v1 is None or v2 is None:
        return "KHÔNG ĐỦ BẰNG CHỨNG"
    if v1 == "ĐẠT" and v2 == "ĐẠT":
        return "ĐẠT"
    if v1 == "HỎNG" or v2 == "HỎNG":
        return "HỎNG"
    return "KHÔNG ĐỦ BẰNG CHỨNG"


def verdict_a_pair(eval1, eval2, rows_eval, all_identities):
    """(a) per §7.3: infeasible/c_min fail => KHÔNG ĐẠT p*; else D_cov CI verdict."""
    t1 = eval1["test_at_tau_star"]
    t2 = eval2["test_at_tau_star"]
    # check p* / c_min for each
    fail1 = (not eval1["tau_star_cp"]["feasible"]) or (not t1.get("meets_c_min", False))
    fail2 = (not eval2["tau_star_cp"]["feasible"]) or (not t2.get("meets_c_min", False))
    if fail1 or fail2:
        which = []
        if fail1:
            which.append(eval1["cell"])
        if fail2:
            which.append(eval2["cell"])
        return {"verdict": "KHÔNG ĐẠT p*", "reason": f"infeasible hoặc < c_min tại: {', '.join(which)}",
                "d_cov": None}
    tau1 = t1["tau"]
    tau2 = t2["tau"]
    dc = d_cov_pair(rows_eval, all_identities, eval1["cell"], eval2["cell"], tau1, tau2)
    # verdict_a for D_cov CI: ĐẠT if CI ⊂ [-delta,+delta]; HỎNG if outside; else không đủ
    v = dc.get("verdict_a")
    return {"verdict": v, "d_cov": dc, "reason": "D_cov(τ*_1,τ*_2) trên tập đánh giá"}


def final_label(verdict_a, verdict_b_res):
    va = verdict_a["verdict"] if isinstance(verdict_a, dict) else verdict_a
    vb = verdict_b_res
    # Priority: KHÔNG ĐẠT p* > KHÔNG SO ĐƯỢC > SO ĐƯỢC > KHÔNG ĐỦ BẰNG CHỨNG
    if va == "KHÔNG ĐẠT p*" or vb == "KHÔNG ĐẠT p*":
        return "KHÔNG ĐẠT p*"
    if va == "HỎNG" or vb == "HỎNG":
        return "KHÔNG SO ĐƯỢC"
    if va == "ĐẠT" and vb == "ĐẠT":
        return "SO ĐƯỢC"
    return "KHÔNG ĐỦ BẰNG CHỨNG"


def pair_report(rows_dii, all_identities, cell1, cell2, calib_ids, test_ids, main_calib, main_test):
    """Compute §7.3-7.5 for a pair in one direction-set (main or reverse)."""
    e1 = eval_cell_73(rows_dii, cell1, main_calib, main_test, all_identities)
    e2 = eval_cell_73(rows_dii, cell2, main_calib, main_test, all_identities)
    _, eval_rows = split_rows(rows_dii, set(main_calib), set(main_test))

    va = verdict_a_pair(e1, e2, eval_rows, all_identities)
    tau1 = e1["tau_star_cp"]["tau"]
    tau2 = e2["tau_star_cp"]["tau"]
    dir_1_to_2 = direction_transfer(eval_rows, all_identities, cell2, cell1, tau1)  # apply tau1 (of cell1) onto cell2 vs cell1
    dir_2_to_1 = direction_transfer(eval_rows, all_identities, cell1, cell2, tau2)  # apply tau2 (of cell2) onto cell1 vs cell2
    vb = verdict_b(dir_1_to_2, dir_2_to_1)
    label = final_label(va, vb)
    return {
        "pair": [cell1, cell2],
        "direction_main": {"calib": "calib73", "test": "test72"} if main_calib == calib_ids else {"calib": "test72", "test": "calib73"},
        "cell_evals": {cell1: e1, cell2: e2},
        "verdict_a": va,
        "transfer": {
            f"apply_tau_{cell1}_onto_{cell2}_vs_{cell1}": dir_1_to_2,
            f"apply_tau_{cell2}_onto_{cell1}_vs_{cell2}": dir_2_to_1,
            "verdict_b": vb,
        },
        "final_label": label,
        "codes": {"a": va["verdict"] if isinstance(va, dict) else va, "b": vb},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # §7.1 split
    split = make_split(csv_path=TIER_A / "loo_Ai.csv", audit_path=TIER_A / "identity_audit.json", verbose=False)
    expected_prefix = "77a64289"
    if not split["split_hash"].startswith(expected_prefix):
        raise SystemExit(f"STOP: split_hash {split['split_hash']} không khớp prefix {expected_prefix}")
    print(f"§7.1 OK split_hash={split['split_hash']}")

    all_identities = sorted(split["calib"] + split["test"], key=lambda s: s.encode("utf-8"))
    # all_identities should be 145
    calib_ids = split["calib"]
    test_ids = split["test"]

    rows_aii = read_csv_rows(TIER_A / "loo_Aii.csv")
    rows_dii = load_dii(rows_aii)
    print(f"D(ii): {len(rows_dii)} rows, {len(set(r['identity'] for r in rows_dii))} identities")

    # §4.7 McNemar
    mcnemar_results = {}
    for enc in ["siglip2", "clip"]:
        mcnemar_results[enc] = mcnemar_for_pair(rows_dii, all_identities, enc)
        print(f"§4.7 {enc}: delta(I-S)={mcnemar_results[enc]['delta_I_minus_S_pp']:.4f}pp p={mcnemar_results[enc]['mcnemar_exact_p_two_sided']:.6g}")
    mcnemar_results["note_clipl14_dinov2"] = "clipl14/dinov2 S≡I — không có cặp (D2: bit-exact, không tách S/I)."

    # §7.2 per cell
    metrics_cells = {}
    for cell in CELLS:
        print(f"§7.2 metrics {cell}...")
        metrics_cells[cell] = metrics_72(rows_dii, cell, all_identities)

    # §7.3 per cell (main direction calib->test)
    cell_evals_main = {}
    for cell in CELLS:
        print(f"§7.3 {cell}...")
        cell_evals_main[cell] = eval_cell_73(rows_dii, cell, calib_ids, test_ids, all_identities)

    # §7.5 pairs
    pairs_case_I = [(a, b) for i, a in enumerate(CASE_I_CELLS) for b in CASE_I_CELLS[i + 1:]]
    pairs_case_S = [("siglip2_S", "clip_S")]
    all_pairs = [(p, "I") for p in pairs_case_I] + [(p, "S") for p in pairs_case_S]

    pair_results = []
    for (c1, c2), case in all_pairs:
        print(f"§7.5 pair ({c1}, {c2}) case {case}...")
        main = pair_report(rows_dii, all_identities, c1, c2, calib_ids, test_ids, calib_ids, test_ids)
        rev = pair_report(rows_dii, all_identities, c1, c2, calib_ids, test_ids, test_ids, calib_ids)
        flag = main["final_label"] != rev["final_label"]
        pair_results.append({
            "case": case,
            "pair": [c1, c2],
            "main_calib_to_test": main,
            "reverse_test_to_calib": rev,
            "flag_reversal": flag,
            "flag": "⚑" if flag else "",
            "final_label": main["final_label"],
            "codes": main["codes"],
        })

    # Assemble outputs
    split_out = {
        "seed": split["seed"],
        "n": split["n"],
        "n_calib": split["n_calib"],
        "n_test": split["n_test"],
        "calib": split["calib"],
        "test": split["test"],
        "order_hash": split["order_hash"],
        "split_hash": split["split_hash"],
        "source": split["source"],
        "split_hash_expected_prefix": expected_prefix,
        "split_hash_match": split["split_hash"].startswith(expected_prefix),
    }

    rq1_results = {
        "meta": {
            "prereg": "prereg_v7 SEALED e5080142…74d1d",
            "date": "2026-11-19",
            "inputs": {
                "loo_Aii_csv": str(TIER_A / "loo_Aii.csv"),
                "loo_Ai_csv": str(TIER_A / "loo_Ai.csv"),
                "identity_audit_json": str(TIER_A / "identity_audit.json"),
            },
            "universe": "A(ii) trên D(ii), 6 ô, fp32 (D8)",
            "n_rows_dii": len(rows_dii),
            "n_identities_dii": len(set(r["identity"] for r in rows_dii)),
            "n_identities_universe": 145,
            "seed": SEED,
            "n_boot": N_BOOT,
            "p_star": P_STAR,
            "c_min": C_MIN,
            "eps_p_pp": EPS_P,
            "delta_star_pp": DELTA_STAR,
            "stdlib_only": True,
        },
        "section_7_1_split": {
            "split_hash": split["split_hash"],
            "order_hash": split["order_hash"],
            "n_calib": 73,
            "n_test": 72,
            "match": split_out["split_hash_match"],
        },
        "section_7_2": metrics_cells,
        "section_7_3_per_cell_main": cell_evals_main,
        "section_7_5_pairs": pair_results,
        "notes": {
            "ci_cluster": "identity (145 cụm, D3) — bootstrap resample trên universe 145 identity cố định; identity ngoài D(ii) đóng góp 0 hàng.",
            "rng": "stdlib random.Random (seed 20261022) qua cluster_draws/tierB_part2 — nhất quán harness.",
            "auc_curve_keys": "precision_coverage_curve dùng keys=path (QUY ƯỚC §4.5).",
            "direction_transfer_note": "apply_tau_X_onto_Y_vs_X = D(Y,X|tau_X)=metric_Y(tau_X)-metric_X(tau_X) theo §7.4.",
        },
    }

    # Write JSON outputs
    with (OUT / "split.json").open("w", encoding="utf-8") as f:
        json.dump(split_out, f, ensure_ascii=False, indent=2)
    with (OUT / "mcnemar.json").open("w", encoding="utf-8") as f:
        json.dump(mcnemar_results, f, ensure_ascii=False, indent=2)
    with (OUT / "rq1_results.json").open("w", encoding="utf-8") as f:
        json.dump(rq1_results, f, ensure_ascii=False, indent=2)

    # Report markdown
    report_lines = build_report(split_out, mcnemar_results, metrics_cells, cell_evals_main, pair_results, rows_dii)
    report_text = "\n".join(report_lines) + "\n"
    (OUT / "report.md").write_text(report_text, encoding="utf-8")

    # SHA256SUMS
    sums = []
    for name in ["split.json", "mcnemar.json", "rq1_results.json", "report.md"]:
        h = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        sums.append(f"{h}  {name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    print("DONE. Files written to", OUT)


def fmt_ci(ci, digits=4):
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def fmt_ci_pp(ci, digits=3):
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}] pp"


def build_report(split_out, mcnemar_results, metrics_cells, cell_evals_main, pair_results, rows_dii):
    L = []
    L.append("# RQ-1 + McNemar §4.7 — prereg_v7 (SEALED e5080142…74d1d)")
    L.append("")
    L.append("- **Ngày:** 2026-11-19")
    L.append("- **Universe:** A(ii) trên D(ii), 6 ô, fp32 (D8). `loo_Aii.csv` (tier A, đóng).")
    L.append(f"- **D(ii):** {len(rows_dii)} hàng, {len(set(r['identity'] for r in rows_dii))} identity có ảnh trong D(ii); universe 145 identity (D3).")
    L.append(f"- **Split §7.1:** seed={SEED}, 73 calib / 72 test, `split_hash={split_out['split_hash']}` (khớp prefix `77a64289` = **{split_out['split_hash_match']}**), `order_hash={split_out['order_hash']}`.")
    L.append("- **Tham số:** p*=0.95, c_min=10%, ε_p=2.0pp, δ*=5.0pp, bootstrap 2000 lượt/seed 20261022, cluster=identity (145 cụm, D3). Stdlib-only.")
    L.append("- **RNG:** `random.Random` stdlib (qua `cluster_draws`/`tierB_part2.py`) — family khai báo.")
    L.append("")

    # §4.7
    L.append("## §4.7 McNemar S vs I (D(ii), case S vs I cùng encoder)")
    L.append("")
    L.append("clipl14/dinov2 S≡I — không có cặp (D2: bit-exact).")
    L.append("")
    for enc in ["siglip2", "clip"]:
        m = mcnemar_results[enc]
        t = m["table_2x2"]
        L.append(f"### {enc}")
        L.append("")
        L.append("Bảng 2×2 (hàng = S, cột = I):")
        L.append("")
        L.append("| | I đúng | I sai |")
        L.append("|---|---:|---:|")
        L.append(f"| **S đúng** | {t['S_correct_I_correct']} | {t['S_correct_I_wrong']} |")
        L.append(f"| **S sai** | {t['S_wrong_I_correct']} | {t['S_wrong_I_wrong']} |")
        L.append("")
        L.append(f"- n={m['n']}, b (S đúng, I sai)={m['b_S_correct_I_wrong']}, c (S sai, I đúng)={m['c_S_wrong_I_correct']}, n_discordant={m['n_discordant']}")
        L.append(f"- acc(S)={m['acc_S']:.6f} ({100*m['acc_S']:.3f}%), acc(I)={m['acc_I']:.6f} ({100*m['acc_I']:.3f}%)")
        L.append(f"- Hiệu top-1(I)−top-1(S) = {m['delta_I_minus_S']:.6f} ({m['delta_I_minus_S_pp']:.3f} pp)")
        L.append(f"- **McNemar exact (binomial 2-sided): p = {m['mcnemar_exact_p_two_sided']:.10g}**")
        pb = m["paired_bootstrap_identity"]
        L.append(f"- Paired bootstrap identity (145 cụm, n_boot={pb['n_boot']}, seed={pb['seed']}), CI95% hiệu (I−S): {fmt_ci(pb['ci95_delta'],6)} = {fmt_ci_pp(pb['ci95_delta_pp'],3)}")
        L.append("")

    # §7.3 summary per cell
    L.append("## §7.3 τ* / sàn c_min (tiêu chí a — theo ô, chiều chính calib→test)")
    L.append("")
    L.append("| Cell | τ* (CP, calib 73) | CP@τ* | n@τ* (calib) | feasible | S1 τ (emp≥p*, calib) | precision test | coverage test | ≥c_min 10% |")
    L.append("|---|---:|---:|---:|:---:|---:|---:|---:|:---:|")
    for cell in CELLS:
        e = cell_evals_main[cell]
        ts = e["tau_star_cp"]
        te = e["test_at_tau_star"]
        tau_s = f"{ts['tau']:.6g}" if ts['tau'] is not None else "None"
        cp_s = f"{ts['cp_at_tau']:.6g}" if ts['cp_at_tau'] is not None else "n/a"
        n_s = str(ts['n_at_tau']) if ts['n_at_tau'] is not None else "n/a"
        feas = "yes" if ts['feasible'] else "no"
        s1 = f"{e['sensitivity_S1_tau']:.6g}" if e['sensitivity_S1_tau'] is not None else "None"
        p = te['precision']
        c = te['coverage']
        p_s = f"{p['value']:.6f} {fmt_ci(p['ci95'],4)}" if p['value'] is not None else "n/a"
        c_s = f"{c['value']:.6f} {fmt_ci(c['ci95'],4)}" if c['value'] is not None else "n/a"
        cm = "yes" if te.get('meets_c_min') else "no"
        L.append(f"| {cell} | {tau_s} | {cp_s} | {n_s} | {feas} | {s1} | {p_s} | {c_s} | {cm} |")
    L.append("")

    # §7.5 pairs
    L.append("## §7.5 Kết luận từng cặp (7 cặp, chiều chính calib→test) + cờ đảo chiều")
    L.append("")
    L.append("Thứ tự ưu tiên nhãn: `KHÔNG ĐẠT p*` > `KHÔNG SO ĐƯỢC` > `SO ĐƯỢC` > `KHÔNG ĐỦ BẰNG CHỨNG`.")
    L.append("")
    L.append("| Case | Cặp | (a) | (b) | Nhãn cuối | ⚑ đảo chiều | D_cov (a, test) CI95% | Ghi chú |")
    L.append("|---|---|---|---|---|:---:|---|---|")
    for pr in pair_results:
        va = pr["codes"]["a"]
        vb = pr["codes"]["b"]
        flag = pr["flag"]
        main = pr["main_calib_to_test"]
        dca = main["verdict_a"].get("d_cov") if isinstance(main["verdict_a"], dict) else None
        if dca and dca.get("ci95_pp") is not None:
            dc_s = f"{dca['d_cov_pp']:.3f} pp {fmt_ci_pp(dca['ci95_pp'],3)}"
        elif dca and dca.get("d_cov_pp") is not None:
            dc_s = f"{dca['d_cov_pp']:.3f} pp"
        else:
            reason = main["verdict_a"].get("reason") if isinstance(main["verdict_a"], dict) else ""
            dc_s = reason or "n/a"
        note = main["verdict_a"].get("reason","") if isinstance(main["verdict_a"], dict) else ""
        pair_s = f"{pr['pair'][0]} vs {pr['pair'][1]}"
        L.append(f"| {pr['case']} | {pair_s} | {va} | {vb} | **{pr['final_label']}** | {flag} | {dc_s} | {note} |")
    L.append("")
    L.append("### Chi tiết chuyển ngưỡng §7.4 (chiều chính, từng cặp)")
    L.append("")
    for pr in pair_results:
        main = pr["main_calib_to_test"]
        c1, c2 = pr["pair"]
        L.append(f"#### {c1} vs {c2} (case {pr['case']}) — verdict_b={main['codes']['b']}")
        for kdir, d in main["transfer"].items():
            if kdir == "verdict_b":
                continue
            L.append(f"- `{kdir}`: D_prec={d.get('d_prec_pp')} pp CI={fmt_ci_pp(d.get('ci_prec_pp'))}, D_cov={d.get('d_cov_pp')} pp CI={fmt_ci_pp(d.get('ci_cov_pp'))}, verdict={d.get('verdict')} {d.get('note','')}")
        L.append("")

    # §7.2 brief tables
    L.append("## §7.2 Chỉ tiêu theo ô (D(ii), toàn bộ)")
    L.append("")
    L.append("### 7.2.2 AUROC (cluster bootstrap identity 145 cụm)")
    L.append("")
    L.append("| Cell | AUROC | CI95% | n_boot |")
    L.append("|---|---:|---:|---:|")
    for cell in CELLS:
        a = metrics_cells[cell]["auroc"]
        L.append(f"| {cell} | {a['auroc']:.6f} | {fmt_ci(a['ci95'],6)} | {a['n_boot']} |")
    L.append("")
    L.append("### 7.2.3 AURC (thường, §7.2.3)")
    L.append("")
    L.append("| Cell | AURC | n |")
    L.append("|---|---:|---:|")
    for cell in CELLS:
        m = metrics_cells[cell]
        L.append(f"| {cell} | {m['aurc']:.6f} | {m['n']} |")
    L.append("")
    L.append("### 7.2.1 Phân phối điểm đúng/sai (tóm tắt)")
    L.append("")
    for cell in CELLS:
        dist = metrics_cells[cell]["distribution"]
        L.append(f"#### {cell} (KS={dist['ks']})")
        L.append("")
        L.append("| Nhóm | n | median | Q1 | Q3 | P10 | P30 | P50 | P70 | P90 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name in ["correct", "wrong"]:
            s = dist[name]
            if s is None:
                L.append(f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            else:
                L.append(f"| {name} | {s['n']} | {s['median']:.6f} | {s['q1']:.6f} | {s['q3']:.6f} | {s['p10']:.6f} | {s['p30']:.6f} | {s['p50']:.6f} | {s['p70']:.6f} | {s['p90']:.6f} |")
        L.append("")

    L.append("## Cửa mở / ghi chú")
    L.append("")
    L.append("- Đảo chiều calib↔test: các cặp có ⚑ không dùng làm claim (§7.5).")
    L.append("- CI bootstrap không gồm bất định của τ* (chọn trên calib); đảo chiều là kiểm tra bù (§7.3).")
    L.append("- RQ-1 hiệu lực trên LOO gallery sản xuất (seen), không suy rộng sang unseen/bậc C (§7.6).")
    L.append("- clipl14/dinov2 S≡I (D2): không có cặp S–I cho McNemar.")
    L.append("- §7.4 verdict chiều: dùng `transfer()` (CI D_prec ⊂ [−ε_p,+ε_p] và D_cov ⊂ [−δ*,+δ*]). (b) đạt khi cả hai chiều đạt.")
    L.append("- Cluster bootstrap resample theo universe 145 identity (D3); 4 identity 1-ảnh không đóng góp hàng D(ii) nhưng nằm trong khung resample.")
    return L


if __name__ == "__main__":
    main()
