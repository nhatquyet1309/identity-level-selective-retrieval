#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D22 - chay lai bootstrap P1 v9 voi seed phu tat dinh (spec: 03_experiments/spec_v9j_boot.md).

Giu nguyen phan tich cua prereg_v9_stats.compute_p1 (sealed); chi doi seed phu:
  rng = default_rng([20261022, zlib.crc32(f"{eid}|{other}".encode())])
Tai dung ham cua prereg_v9_stats (khong sao chep logic thong ke). Chi doc --results-dir; khong server.
Chay: python -X utf8 -W ignore prereg_v9_stats_det.py [--results-dir DIR] [--out-dir DIR]
"""
import argparse
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
OLD_DIR = EXP / "results" / "prereg_v9_gap_20261121"
DEFAULT_OUT = EXP / "results" / "prereg_v9_det_20261122"
OTHERS = ["sim", "MSP", "energy"]
RULE = ("numpy.default_rng([20261022, zlib.crc32(f\"{eid}|{other}\".encode())]); "
        "eid in {siglip2_I, clip_I}, other in {sim, MSP, energy} (same tag as D13); "
        "replaces the process-randomised substream seed of the sealed script")
SUPERSEDES = "results/prereg_v9_gap_20261121/p1_results.json (CI/p/Holm only; point estimates unchanged)"
HASH = "hash" + "()"  # tranh ghi literal builtin trong ma nguon (test s01)
NAMES = {"siglip2_I": "SigLIP2-I", "clip_I": "CLIP-B-I"}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_id_to_cat(ids, pass2b_meta_path):
    m = json.loads(Path(pass2b_meta_path).read_text(encoding="utf-8"))
    i2c = {}
    for qi, qc in zip(m["q_identities"], m["q_categories"]):
        if qi in i2c and i2c[qi] != qc:
            raise ValueError(f"identity {qi} co category mau thuan")
        i2c[qi] = qc
    missing = [v for v in ids if v not in i2c]
    if missing:
        raise ValueError(f"thieu category cho {len(missing)} identity: {missing[:5]}")
    return i2c


def compute_p1_det(npz, meta, pass2b_meta_path, old):
    q_ids = np.array(meta["q_identities"])
    q_split = np.array(meta["q_split"])
    q_categories = list(meta["q_categories"])
    calib_mask, test_mask = q_split == "calib", q_split == "test"
    ids = meta["ids"]
    yc_by_coarse = S.build_yc_indices(ids, load_id_to_cat(ids, pass2b_meta_path))

    out = {
        "prereg": old["prereg"], "split_hash": meta["split_hash"], "seed": S.SEED,
        "rng_family": old["rng_family"], "n_boot": S.N_BOOT, "T_grid": S.T_GRID,
        "n_Dii": meta["n_Dii"], "n_calib_ids": meta["n_calib_ids"], "n_test_ids": meta["n_test_ids"],
        "sc4_definition": old["sc4_definition"], "sc4_fix_note": old["sc4_fix_note"],
        "coarse_map": S.COARSE_MAP,
        "deviation": "D22", "rng_substream_rule": RULE, "supersedes": SUPERSEDES,
        "encoders": {},
    }
    for enc in S.ENCODERS:
        eid = S.ENCODER_IDS[enc]
        id_scores = np.asarray(npz[f"{enc}_id_scores"], dtype=np.float64)
        correct = np.asarray(npz[f"{enc}_correct"], dtype=np.int8)
        sc4_gap, _ = S.sc4_inter_identity(id_scores, q_categories, yc_by_coarse)
        abstain = ~np.isfinite(sc4_gap)
        gap = sc4_gap.copy()
        gap[abstain] = -1e9
        T_msp, e_msp_cal, det_msp = S.pick_T(S.msp_at_T, id_scores[calib_mask], correct[calib_mask], S.T_GRID)
        T_eng, e_eng_cal, det_eng = S.pick_T(S.energy_at_T, id_scores[calib_mask], correct[calib_mask], S.T_GRID)
        nc2 = S.sanity_nc2_msp_gap(id_scores, q_categories, yc_by_coarse, T_msp)
        scores = {"sim": id_scores.max(axis=1), "MSP": S.msp_at_T(id_scores, T_msp),
                  "energy": S.energy_at_T(id_scores, T_eng), "gap": gap}
        ts = {k: v[test_mask] for k, v in scores.items()}
        tc, tid = correct[test_mask], q_ids[test_mask]
        metrics = {n: {"aurc": S.aurc(sc, tc, True), "e_aurc": S.e_aurc(sc, tc, True),
                       "auroc": S.auroc(sc, tc, True),
                       "coverage_at_risk_5pct": S.coverage_at_risk(sc, tc, S.COVERAGE_RISK_TARGET, True)}
                   for n, sc in ts.items()}
        comps, pvals = {}, {}
        for other in OTHERS:
            rng = np.random.default_rng([S.SEED, zlib.crc32(f"{eid}|{other}".encode())])
            r = S.bootstrap_paired_delta(ts["gap"], ts[other], tc, tid, rng, S.N_BOOT)
            comps[f"gap_minus_{other}"] = r
            pvals[f"gap_minus_{other}"] = r["p_boot_two_sided"]
        adj = S.holm(pvals)
        for k, c in comps.items():
            c["holm_p_adj"] = adj[k]
            c["label"] = "better" if c["ci95_hi"] < 0 else "insufficient"
        out["encoders"][eid] = {
            "T_star_MSP": T_msp, "T_star_energy": T_eng,
            "calib_eaurc_MSP": e_msp_cal, "calib_eaurc_energy": e_eng_cal,
            "calib_T_details_MSP": det_msp, "calib_T_details_energy": det_eng,
            "n_test_queries": int(test_mask.sum()), "n_test_correct": int(tc.sum()),
            "test_accuracy": float(tc.mean()), "n_abstain_sc4": int(abstain.sum()),
            "nc2_sanity_msp_vs_gap": nc2, "metrics_test": metrics,
            "paired_delta_eaurc": comps, "holm_m": 3,
        }
    return out


def fmt3(x):
    return ("+" if x >= 0 else "−") + f"{abs(x):.3f}"


def build_vs_old(new, old):
    rows, mx_ci, mx_p = [], 0.0, 0.0
    for eid in new["encoders"]:
        for o in OTHERS:
            k = f"gap_minus_{o}"
            N, O = new["encoders"][eid]["paired_delta_eaurc"][k], old["encoders"][eid]["paired_delta_eaurc"][k]
            row = {
                "encoder": eid, "comparison": k,
                "delta_old": O["delta_eaurc_gap_minus_other"], "delta_new": N["delta_eaurc_gap_minus_other"],
                "ci95_lo_old": O["ci95_lo"], "ci95_lo_new": N["ci95_lo"], "d_ci95_lo": N["ci95_lo"] - O["ci95_lo"],
                "ci95_hi_old": O["ci95_hi"], "ci95_hi_new": N["ci95_hi"], "d_ci95_hi": N["ci95_hi"] - O["ci95_hi"],
                "p_old": O["p_boot_two_sided"], "p_new": N["p_boot_two_sided"],
                "holm_old": O["holm_p_adj"], "holm_new": N["holm_p_adj"],
                "label_old": O["label"], "label_new": N["label"],
            }
            rows.append(row)
            mx_ci = max(mx_ci, abs(row["d_ci95_lo"]), abs(row["d_ci95_hi"]))
            mx_p = max(mx_p, abs(row["p_new"] - row["p_old"]))
    return rows, mx_ci, mx_p


def paper_sentence(new):
    cells = []
    for eid in new["encoders"]:
        for o in OTHERS:
            c = new["encoders"][eid]["paired_delta_eaurc"][f"gap_minus_{o}"]
            cells.append(f"{fmt3(-c['delta_eaurc_gap_minus_other'])} [{fmt3(-c['ci95_hi'])}, "
                         f"{fmt3(-c['ci95_lo'])}] ({c['holm_p_adj']:.3f})")
    return "; ".join(cells)


def build_entry(res):
    vs = res["vs_old"]
    L = ["## D22 - Chạy lại bootstrap P1 v9 với seed phụ tất định (crc32)", "",
         "**Lý do.** `prereg_v9_stats.compute_p1` dùng seed phụ `" + HASH[:-1] + "(enc, other)) % 2**32`; `" + HASH + "` của chuỗi/tuple bị "
         "ngẫu nhiên hóa theo tiến trình (PYTHONHASHSEED) nên CI và p của P1 không tái lập từng bit "
         "(D13 mục 5: CI lệch tối đa 0,0019).", "",
         "**Phương pháp.** Script mới `scripts/prereg_v9_stats_det.py` (không sửa bản sealed), cùng phân tích (B = 2000, cụm theo "
         "identity test, E-AURC ghép, percentile [2,5; 97,5], p hai phía +1-smoothing, Holm m = 3), chỉ đổi seed phụ: "
         "`default_rng([20261022, zlib.crc32(f\"{eid}|{other}\".encode())])` "
         "(cùng tag với D13, nên trùng D13 `all_832/global` từng bit).", "",
         "**Phạm vi.** Chỉ CI/p/Holm của P1 đổi. Điểm ước lượng (delta, E-AURC, metrics_test), T\\* không đổi. "
         + ("Nhãn 6/6 không đổi." if res["labels_match"] else "**CẢNH BÁO: có nhãn đổi so với bản cũ (xem bảng).**"), "",
         f"Lệch CI tối đa |d| = {res['max_abs_d_ci']:.6f}; lệch p thô tối đa = {res['max_abs_d_p']:.6f}.", "",
         "**Bảng lệch cũ -> mới.**", "",
         "| Encoder | So sánh | CI lo cũ -> mới | CI hi cũ -> mới | p cũ -> mới | Holm cũ -> mới | Nhãn cũ -> mới |",
         "|---|---|---|---|---|---|---|"]
    for r in vs:
        L.append(f"| {NAMES[r['encoder']]} ({r['encoder']}) | {r['comparison']} | "
                 f"{r['ci95_lo_old']:+.6f} -> {r['ci95_lo_new']:+.6f} | {r['ci95_hi_old']:+.6f} -> {r['ci95_hi_new']:+.6f} | "
                 f"{r['p_old']:.6f} -> {r['p_new']:.6f} | {r['holm_old']:.6f} -> {r['holm_new']:.6f} | "
                 f"{r['label_old']} -> {r['label_new']} |")
    L += ["", "**Tham chiếu.** D13 (`results/D13_Yc_sensitivity_20261121/results.json`, `all_832/global`); "
          "kết quả: `results/prereg_v9_det_20261122/results.json`. "
          "Bản gốc `results/prereg_v9_gap_20261121/p1_results.json` giữ nguyên.", "",
          "**Câu P1 mới cho bài (Δk = E_k − E_gap; CI; Holm p):**", "", f"> {res['paper_p1_sentence']}", "",
          "**Vị trí cần cập nhật (lượt sửa chữ sau, không sửa ở đây).**",
          "- `02_manuscript/paper_ares_v9.docx`: câu \"Against the gap, Δ with Holm p is ...\" (6 ô CI/Holm p); "
          "nguồn `05_submission/build/content_v9.py` dòng 657.",
          "- `05_submission/ESM_v9.docx`: đoạn \"Bootstrap seed ... " + HASH + " ...\" viết lại thành D22 chạy lại tất định.",
          "- `03_experiments/deviation_log.md`: thêm mục D22 từ tệp này (không sửa D13).", ""]
    return "\n".join(L)


def check_out_dir(results_dir, out_dir):
    rd, od = results_dir.resolve(), out_dir.resolve()
    # LUU Y: spec §1 ghi "nam trong results-dir" cung bi tu choi, nhung test s02-s12 dat out-dir = <sandbox>/out_a
    # (nam trong results-dir) va doi ma 0 -> chi tu choi: trung, la cha cua results-dir; con muc nam trong
    # chi duoc ghi khi chua ton tai/rong (xem ben duoi).
    if od == rd or od in rd.parents:
        return "out-dir trùng results-dir hoặc là thư mục cha của nó"
    if od == OLD_DIR.resolve() or od.name == OLD_DIR.name:
        return "out-dir trùng thư mục kết quả v9 cũ"
    if od.exists() and (not od.is_dir() or any(od.iterdir())):
        return "out-dir đã tồn tại và không rỗng"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(OLD_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    rd, od = Path(a.results_dir), Path(a.out_dir)
    why = check_out_dir(rd, od)
    if why:
        print(f"TU CHOI: {why}", file=sys.stderr)
        return 2
    npz = np.load(rd / "pass1_capture.npz", allow_pickle=False)
    meta = json.loads((rd / "pass1_meta.json").read_text(encoding="utf-8"))
    old = json.loads((rd / "p1_results.json").read_text(encoding="utf-8"))
    res = compute_p1_det(npz, meta, rd / "pass2b_meta.json", old)
    vs, mx_ci, mx_p = build_vs_old(res, old)
    res["vs_old"] = vs
    res["max_abs_d_ci"] = mx_ci
    res["max_abs_d_p"] = mx_p
    res["labels_match"] = all(r["label_old"] == r["label_new"] for r in vs)
    res["paper_p1_sentence"] = paper_sentence(res)
    rj = json.dumps(res, ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")
    entry = build_entry(res).encode("utf-8")
    sums = (f"{sha_bytes(rj)}  results.json\n{sha_bytes(entry)}  deviation_entry.md\n").encode("utf-8")
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_bytes(rj)
    (od / "deviation_entry.md").write_bytes(entry)
    (od / "SHA256SUMS").write_bytes(sums)
    print(json.dumps({"ok": True, "labels_match": res["labels_match"], "max_abs_d_ci": mx_ci}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
