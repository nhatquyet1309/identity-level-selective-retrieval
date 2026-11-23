#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D13 — độ nhạy: MSP/energy tính trên tập ứng viên Y_c (spec: 03_experiments/spec_D13.md).

Giữ nguyên v9 (sealed): sim toàn cục, gap trong Y_c, `correct` theo dự đoán toàn cục, T* đã seal.
Chỉ MSP và energy được tính lại trong Y_c. Dùng lại hàm metric/bootstrap/Holm của prereg_v9_stats.py.
Chạy bằng Python311: python d13_yc_sensitivity.py [--out-dir DIR] [--results-dir DIR]
"""
import argparse
import csv
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prereg_v9_stats as S  # noqa: E402

EXP = HERE.parent
DEFAULT_RESULTS = EXP / "results" / "prereg_v9_gap_20261121"
DEFAULT_OUT = EXP / "results" / "D13_Yc_sensitivity_20261121"
SCORES = ["sim", "MSP", "energy", "gap"]
OTHERS = ["sim", "MSP", "energy"]
DEFS = ["global", "Yc"]
STRATA = ["all_832", "nc_ge3"]


# ───────────────────────── Y_c và điểm theo Y_c ─────────────────────────
def load_yc_by_coarse(ids, pass2b_meta_path):
    """coarse -> chỉ số cột identity. Map identity->category lấy từ pass2b_meta.json (đủ 145 identity)."""
    m = json.loads(Path(pass2b_meta_path).read_text(encoding="utf-8"))
    i2c = {}
    for qi, qc in zip(m["q_identities"], m["q_categories"]):
        if qi in i2c and i2c[qi] != qc:
            raise ValueError(f"identity {qi} có category mâu thuẫn")
        i2c[qi] = qc
    missing = [v for v in ids if v not in i2c]
    if missing:
        raise ValueError(f"thiếu category cho {len(missing)} identity: {missing[:5]}")
    return S.build_yc_indices(list(ids), i2c)


def compute_yc_scores(id_scores, q_categories, yc_by_coarse, T_msp, T_energy):
    """MSP^{Yc}, -E^{Yc}, s1^{Yc}, gap (inter-identity trong Y_c), n_c. ValueError nếu category lạ/n_c<1."""
    a = np.asarray(id_scores, dtype=np.float64)
    n = a.shape[0]
    cats = list(q_categories)
    if len(cats) != n:
        raise ValueError("số category != số query")
    coarse = []
    for c in cats:
        if not isinstance(c, str) or c not in S.CAT_TO_COARSE:
            raise ValueError(f"category không thuộc COARSE_MAP: {c!r}")
        k = S.CAT_TO_COARSE[c]
        idx = yc_by_coarse.get(k)
        if idx is None or len(idx) < 1:
            raise ValueError(f"Y_c rỗng cho nhóm {k}")
        coarse.append(k)
    msp = np.zeros(n); en = np.zeros(n); s1 = np.zeros(n)
    gap = np.full(n, np.nan); nc = np.zeros(n, dtype=np.int64)
    coarse_arr = np.array(coarse, dtype=object)
    for k in set(coarse):
        rows = np.where(coarse_arr == k)[0]
        idx = np.asarray(yc_by_coarse[k], dtype=np.int64)
        v = a[np.ix_(rows, idx)]
        nc[rows] = idx.size
        z = v / float(T_msp)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        msp[rows] = (e / e.sum(axis=1, keepdims=True)).max(axis=1)
        z = v / float(T_energy)
        zm = z.max(axis=1, keepdims=True)
        en[rows] = float(T_energy) * (zm[:, 0] + np.log(np.exp(z - zm).sum(axis=1)))
        s1[rows] = v.max(axis=1)
        if idx.size >= 2:
            sv = -np.sort(-v, axis=1)
            gap[rows] = sv[:, 0] - sv[:, 1]
    return {"msp_yc": msp, "energy_yc": en, "s1_yc": s1, "gap": gap, "n_c": nc}


# ───────────────────────── thống kê theo tầng ─────────────────────────
def _metrics(sc, c):
    return {"aurc": S.aurc(sc, c), "e_aurc": S.e_aurc(sc, c), "auroc": S.auroc(sc, c),
            "coverage_at_risk_5pct": S.coverage_at_risk(sc, c, S.COVERAGE_RISK_TARGET)}


def _block(scores, c, qid, eid):
    out = {"metrics_test": {k: _metrics(v, c) for k, v in scores.items()}}
    comp, pv = {}, {}
    for o in OTHERS:
        rng = np.random.default_rng([S.SEED, zlib.crc32(f"{eid}|{o}".encode())])
        r = S.bootstrap_paired_delta(scores["gap"], scores[o], c, qid, rng, S.N_BOOT)
        comp[f"gap_minus_{o}"] = r
        pv[f"gap_minus_{o}"] = r["p_boot_two_sided"]
    adj = S.holm(pv)
    for k, r in comp.items():
        r["holm_p_adj"] = adj[k]
        r["label"] = "better" if r["ci95_hi"] < 0 else "insufficient"
    out["paired_delta_eaurc"] = comp
    return out


def _fmt(x, nd=4):
    return "NA" if x is None else f"{x:.{nd}f}"


def write_report(R, path, p1):
    L = []
    L.append("# D13 — Độ nhạy: MSP/energy tính trên tập ứng viên Y_c (so với toàn cục 145 identity của v9)\n")
    L.append("Ngày: 21/11/2026. Đặc tả: `03_experiments/spec_D13.md`. Hiện thực: `scripts/d13_yc_sensitivity.py`.\n")
    L.append("## 1. Mục đích và công bố khác biệt (D13)\n")
    L.append("- Trong v9 (sealed), **MSP và energy được tính trên toàn cục 145 identity** (softmax/logsumexp theo cả hàng điểm), "
             "còn gap là khoảng cách top-2 trong Y_c. Table 1 của paper mô tả MSP trên Y_c -> **lệch định nghĩa**, công bố tại đây.")
    L.append("- Kết quả v9 được **giữ nguyên, không đổi nhãn đã seal**. D13 là phân tích độ nhạy trên cùng `pass1_capture.npz`, "
             "dùng **T\\* đã seal** (MSP và energy từng encoder), **không chọn lại T**. T\\* = 0,01 ở mép dưới của lưới; "
             "không kết luận gì về T khác 0,01.")
    L.append("- `sim` (toàn cục), `gap` (Y_c) và `correct` (dự đoán toàn cục, argmax 145 identity) **giữ nguyên**; chỉ MSP và energy đổi.")
    L.append("- Map identity->category lấy từ `pass2b_meta.json` (đủ 145 identity); `pass1_meta.json` thiếu 4 identity "
             "(`btr80`, `ec130`, `switchblade_600`, `z19`) nên không dùng.\n")
    L.append("## 2. Định nghĩa\n")
    L.append("Toàn cục (v9): MSP_T = max_y softmax(s/T) trên 145 identity; -E_T = T·logsumexp(s/T) trên 145 identity.\n")
    L.append("Y_c (D13): cùng công thức nhưng chỉ trên Y_c = {y: coarse(y) = coarse(query)}. "
             f"Kích thước Y_c: {R['yc_sizes']}.\n")
    L.append("## 3. Kết quả theo tầng\n")
    L.append("Tầng (a) `all_832`: toàn bộ 832 query test. Tầng (b) `nc_ge3`: query có n_c >= 3 (826 query). "
             "**n_c = 2 chỉ là nhóm Ship = 6 query test (0,7%)** nên (a) và (b) gần như trùng; ý kiến 'tầng n_c=2 trộn' "
             "chỉ liên quan 6 query.\n")
    for eid, E in R["encoders"].items():
        L.append(f"### {eid} (T*_MSP={E['T_star_MSP']}, T*_energy={E['T_star_energy']})\n")
        for sk in STRATA:
            st = E["strata"][sk]
            L.append(f"**Tầng {sk}** — n={st['n']}, số lỗi={st['n_err']}\n")
            L.append("| Định nghĩa | Score | AURC | E-AURC | AUROC | cov@5% |")
            L.append("|---|---|---|---|---|---|")
            for df in DEFS:
                for s in SCORES:
                    m = st[df]["metrics_test"][s]
                    L.append(f"| {df} | {s} | {_fmt(m['aurc'], 6)} | {_fmt(m['e_aurc'], 6)} | {_fmt(m['auroc'], 4)} | "
                             f"{_fmt(m['coverage_at_risk_5pct']['coverage'], 4)} |")
            L.append("")
            L.append("| Định nghĩa | So sánh | ΔE-AURC | CI95 | p | Holm p | Nhãn |")
            L.append("|---|---|---|---|---|---|---|")
            for df in DEFS:
                for o in OTHERS:
                    c = st[df]["paired_delta_eaurc"][f"gap_minus_{o}"]
                    L.append(f"| {df} | gap−{o} | {c['delta_eaurc_gap_minus_other']:+.6f} | "
                             f"[{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}] | {c['p_boot_two_sided']:.4f} | "
                             f"{c['holm_p_adj']:.4f} | {c['label']} |")
            L.append("")
            ch = []
            for o in OTHERS:
                lg = st["global"]["paired_delta_eaurc"][f"gap_minus_{o}"]["label"]
                ly = st["Yc"]["paired_delta_eaurc"][f"gap_minus_{o}"]["label"]
                ch.append(f"gap−{o}: {lg} -> {ly} ({'ĐỔI' if lg != ly else 'không đổi'})")
            L.append("Đối chiếu nhãn toàn cục -> Y_c: " + "; ".join(ch) + "\n")
    cc = R["consistency_check"]
    L.append("## 4. Kiểm tra nhất quán với v9 (tầng all_832, định nghĩa toàn cục)\n")
    L.append(f"- Sai lệch điểm lớn nhất (AURC/E-AURC/AUROC/Δ) so với `p1_results.json`: {cc['max_abs_diff_point_estimates']:.3e} (ngưỡng 1e-6).")
    L.append(f"- Sai lệch CI lớn nhất: {cc['max_abs_diff_ci']:.4f} (ngưỡng 0,004; do seed phụ).")
    L.append(f"- Nhãn khớp v9: {cc['labels_match']}; T* chọn lại trên calib (toàn cục) khớp T* sealed: {cc['T_star_match']}.\n")
    L.append("## 5. Ghi chú tái lập\n")
    L.append("v9 dùng `hash((enc, other)) % 2**32` làm seed phụ; `hash` chuỗi bị ngẫu nhiên hóa theo tiến trình nên CI/p của v9 "
             "**không tái lập bit-by-bit** (sai số Monte-Carlo ~0,001). D13 dùng seed phụ xác định "
             "`default_rng([20261022, zlib.crc32(f\"{eid}|{other}\")])`, dùng chung mọi tầng và mọi định nghĩa (so sánh ghép). "
             "Holm: 3 so sánh cho mỗi (encoder × tầng × định nghĩa); nhãn `better` nếu ci95_hi < 0, ngược lại `insufficient` (luật v9, không δ mới).\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    rd = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(rd / "pass1_capture.npz")
    meta = json.loads((rd / "pass1_meta.json").read_text(encoding="utf-8"))
    p1 = json.loads((rd / "p1_results.json").read_text(encoding="utf-8"))
    yc = load_yc_by_coarse(meta["ids"], str(rd / "pass2b_meta.json"))
    sp = np.array(meta["q_split"])
    te, ca = sp == "test", sp == "calib"
    qid = np.array(meta["q_identities"])
    cats = list(meta["q_categories"])

    R = {
        "deviation": "D13", "prereg": "prereg_v9.md SEALED", "seed": S.SEED, "n_boot": S.N_BOOT,
        "rng_family": "numpy.default_rng(PCG64)",
        "rng_substream_rule": "default_rng([20261022, zlib.crc32(f'{eid}|{other}'.encode())]); dùng chung mọi tầng và định nghĩa",
        "split_hash": meta["split_hash"], "T_policy": "sealed_T_star_no_reselection",
        "coarse_map": S.COARSE_MAP, "category_source": "pass2b_meta.json",
        "yc_sizes": {k: int(len(v)) for k, v in yc.items()},
        "encoders": {},
    }
    max_pt, max_ci, labels_ok, t_ok = 0.0, 0.0, True, True
    csv_rows = []
    for enc in S.ENCODERS:
        eid = S.ENCODER_IDS[enc]
        a = np.asarray(z[f"{enc}_id_scores"], dtype=np.float64)
        c_all = np.asarray(z[f"{enc}_correct"], dtype=np.int8)
        P = p1["encoders"][eid]
        Tm, Te = P["T_star_MSP"], P["T_star_energy"]
        # kiểm nhất quán T*: chọn lại trên calib bằng định nghĩa toàn cục
        Tm2 = S.pick_T(S.msp_at_T, a[ca], c_all[ca], S.T_GRID)[0]
        Te2 = S.pick_T(S.energy_at_T, a[ca], c_all[ca], S.T_GRID)[0]
        t_ok &= (Tm2 == Tm and Te2 == Te)

        ry = compute_yc_scores(a, cats, yc, Tm, Te)
        gap = np.where(np.isfinite(ry["gap"]), ry["gap"], -1e9)  # abstain như v9
        glob = {"sim": a.max(axis=1), "MSP": S.msp_at_T(a, Tm), "energy": S.energy_at_T(a, Te), "gap": gap}
        ycs = {"sim": glob["sim"], "MSP": ry["msp_yc"], "energy": ry["energy_yc"], "gap": gap}
        nc = ry["n_c"]
        masks = {"all_832": te, "nc_ge3": te & (nc >= 3)}

        strata = {}
        for sk, m in masks.items():
            c = c_all[m]
            st = {"n": int(m.sum()), "n_err": int((1 - c).sum())}
            for df, sc in (("global", glob), ("Yc", ycs)):
                st[df] = _block({k: v[m] for k, v in sc.items()}, c, qid[m], eid)
            strata[sk] = st

        # kiểm nhất quán với p1_results (tầng all_832, global)
        g = strata["all_832"]["global"]
        for s in SCORES:
            for k in ("aurc", "e_aurc", "auroc"):
                max_pt = max(max_pt, abs(g["metrics_test"][s][k] - P["metrics_test"][s][k]))
        for o in OTHERS:
            x, y = g["paired_delta_eaurc"][f"gap_minus_{o}"], P["paired_delta_eaurc"][f"gap_minus_{o}"]
            max_pt = max(max_pt, abs(x["delta_eaurc_gap_minus_other"] - y["delta_eaurc_gap_minus_other"]))
            max_ci = max(max_ci, abs(x["ci95_lo"] - y["ci95_lo"]), abs(x["ci95_hi"] - y["ci95_hi"]))
            labels_ok &= (x["label"] == y["label"])

        R["encoders"][eid] = {"T_star_MSP": Tm, "T_star_energy": Te, "strata": strata}

        c_te = c_all[te]
        for df, sc in (("sealed", glob), ("Yc", ycs)):
            for s in SCORES:
                cov, risk = S.risk_coverage(sc[s][te], c_te)
                for cv, rk in zip(cov, risk):
                    csv_rows.append([eid, df, s, repr(float(cv)), repr(float(rk))])

    R["consistency_check"] = {"max_abs_diff_point_estimates": float(max_pt), "max_abs_diff_ci": float(max_ci),
                              "labels_match": bool(labels_ok), "T_star_match": bool(t_ok)}

    (out / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "rc_curves_v9.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "definition", "score", "coverage", "risk"])
        w.writerows(csv_rows)
    write_report(R, out / "report.md", p1)
    lines = []
    for name in ("report.md", "results.json", "rc_curves_v9.csv"):
        h = hashlib.sha256((out / name).read_bytes()).hexdigest()
        lines.append(f"{h}  {name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(R["consistency_check"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
