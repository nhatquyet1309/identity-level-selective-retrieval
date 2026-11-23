#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[C4] v9 revision, D17 — ablation MÔ TẢ conflict gate (có / không cổng).

Cổng: conflict = 1 -> score = -inf (luôn dưới mọi query khác; ties giữa chúng, mergesort ổn định).
ab4: conflict = cột `conflict` (== unfiltered_top_identity != y1); ab4_test = test n_c >= 3, ab4_calib = calib.
v9: conflict = nhãn global != nhãn Y_c; score sealed (s1 toàn cục, MSP/energy toàn cục T*, gap Y_c), nhãn global, 832 test.
Chỉ mô tả, không chứng nhận. Đặc tả: 05_submission/spec_v9_revision.md mục 5 [C4].
Chạy bằng Python311: python v9rev_gate.py [--out-dir DIR]
"""
import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v9rev_yc as Y  # noqa: E402
import prereg_v9_stats as S  # noqa: E402

T_MSP_AB4, T_EN_AB4 = 0.02, 0.01   # đã đăng ký cho ab4


def ab4_scores(rows):
    s1, msp, en = [], [], []
    for r in rows:
        v = np.array(list(r["s_y"].values()), float)
        s1.append(v.max())
        z = v / T_MSP_AB4
        e = np.exp(z - z.max())
        msp.append((e / e.sum()).max())
        z = v / T_EN_AB4
        en.append(T_EN_AB4 * (z.max() + math.log(np.exp(z - z.max()).sum())))
    return {"s1": np.array(s1), "MSP": np.array(msp), "energy": np.array(en),
            "gap": np.array([r["m_id"] for r in rows], float)}


def gate_block(sc, cor, conf):
    m = conf == 1
    blk = {"n": int(len(cor)), "n_conflict": int(m.sum()),
           "err_rate_conflict": float((1 - cor[m]).mean()) if m.any() else None,
           "err_rate_nonconflict": float((1 - cor[~m]).mean()) if (~m).any() else None, "scores": {}}
    for k, s in sc.items():
        g = np.where(m, -np.inf, s)
        no = {"e_aurc": Y.eaurc(s, cor), "auroc": Y.auroc(s, cor)}
        yes = {"e_aurc": Y.eaurc(g, cor), "auroc": Y.auroc(g, cor)}
        blk["scores"][k] = {"without_gate": no, "with_gate": yes,
                            "delta_eaurc_gate_minus_nogate": yes["e_aurc"] - no["e_aurc"]}
    return blk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Y.default_out("gate")))
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(Y.AB4 / "predictions.jsonl", encoding="utf-8") if l.strip()]
    V = Y.load_v9()
    R = {"deviation": "D17", "post_hoc": True, "seed": Y.SEED,
         "label": "post hoc, descriptive ablation: conflict gate co/khong (khong ket luan chung nhan)",
         "note": "Descriptive ablation only: conflict gate (refuse khi conflict=1) co/khong tren cung quan the query; "
                 "khong phai ket luan chung nhan hay kiem dinh gia thuyet.",
         "sources": [Y.rel(Y.AB4 / "predictions.jsonl")] + Y.v9_sources(),
         "gate_rule": "conflict=1 -> score=-inf (duoi moi query khac, ties on dinh); khong cong = score goc",
         "ab4_temperatures": {"MSP": T_MSP_AB4, "energy": T_EN_AB4}, "datasets": {}}
    for name, sel in (("ab4_test", lambda r: r["split"] == "test" and r["n_c"] >= 3), ("ab4_calib", lambda r: r["split"] == "calib")):
        rs = [r for r in rows if sel(r)]
        R["datasets"][name] = gate_block(ab4_scores(rs), np.array([r["correct"] for r in rs], int),
                                         np.array([r["conflict"] for r in rs], int))
    te = V["test"]
    for e, eid in Y.ENC.items():
        E = V["enc"][e]
        a = E["a"]
        sc = {"s1": a.max(1), "MSP": S.msp_at_T(a, E["Tm"]), "energy": S.energy_at_T(a, E["Te"]), "gap": E["gap"]}
        R["datasets"][f"v9_{eid}_test"] = gate_block({k: v[te] for k, v in sc.items()}, E["cor_g"][te],
                                                     (E["pred_g"] != E["pred_y"])[te].astype(int))
    d = R["datasets"]
    dev = ("## D17 — Ablation mô tả conflict gate (post hoc)\n\n"
           "- Ngày: %s. Script: `03_experiments/scripts/v9rev_gate.py` (Python311, numpy). Đặc tả: `05_submission/spec_v9_revision.md` mục 5 [C4].\n"
           "- Dữ liệu: ab4 `predictions.jsonl`; v9 `pass1_capture.npz` + Y_c + T* sealed. Không đổi dữ liệu sealed.\n"
           "- Cổng: conflict -> score = -inf. ab4_test n=%d (conflict %d), ab4_calib n=%d (conflict %d).\n"
           "- Chỉ là ablation mô tả; không kết luận chứng nhận (certified).\n") % (
        datetime.date.today().isoformat(), d["ab4_test"]["n"], d["ab4_test"]["n_conflict"], d["ab4_calib"]["n"], d["ab4_calib"]["n_conflict"])
    Y.finalize(Path(args.out_dir), R, dev)
    print(json.dumps({k: v["n_conflict"] for k, v in d.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
