#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[C3] v9 revision, D16 — image-level gap m_img so với identity gap m_id (post hoc, mô tả).

m_img = s1_img - s2_img (ảnh hạng 1 vs ảnh hạng 2 bất kể identity).
ab4: cột m_img/m_id có sẵn. v9: m_img = s1_img - max(s2_img_same_id, s2_id); khối `global` (832 test, nhãn global)
và `yc_subset` (test có nhãn global = nhãn Y_c; m_id = s1_Yc - gap_Yc).
Đặc tả: 05_submission/spec_v9_revision.md mục 5 [C3]. Chạy bằng Python311: python v9rev_imggap.py [--out-dir DIR]
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v9rev_yc as Y  # noqa: E402


def block(mid, mimg, cor, same, cl, tag):
    n = len(cor)
    blk = {"n": int(n), "n_err": int((1 - cor).sum()),
           "e_aurc": {"m_id": Y.eaurc(mid, cor), "m_img": Y.eaurc(mimg, cor)},
           "auroc": {"m_id": Y.auroc(mid, cor), "m_img": Y.auroc(mimg, cor)}}
    pt = blk["e_aurc"]["m_img"] - blk["e_aurc"]["m_id"]
    lo, hi = Y.cluster_boot(lambda idx: Y.eaurc(mimg[idx], cor[idx]) - Y.eaurc(mid[idx], cor[idx]), cl, tag)
    blk["paired_delta_eaurc_mimg_minus_mid"] = Y.delta_block(pt, lo[0], hi[0])
    r = {"n_same": int(same.sum()), "n_other": int((~same).sum()), "frac_same": float(same.sum() / n)}
    for t, m in (("same", same), ("other", ~same)):
        ok = int(m.sum()) >= 5
        r["acc_" + t] = float(cor[m].mean()) if ok else None
        r["e_aurc_mid_" + t] = Y.eaurc(mid[m], cor[m]) if ok else None
        r["e_aurc_mimg_" + t] = Y.eaurc(mimg[m], cor[m]) if ok else None
    blk["rank2_same_identity"] = r
    return blk


def ab4_rows():
    return [json.loads(l) for l in open(Y.AB4 / "predictions.jsonl", encoding="utf-8") if l.strip()]


def ab4_block(rows, tag):
    return block(np.array([r["m_id"] for r in rows], float), np.array([r["m_img"] for r in rows], float),
                 np.array([r["correct"] for r in rows], int), np.array([bool(r["top2_same_identity"]) for r in rows]),
                 np.array([r["gt_identity"] for r in rows]), tag)


def v9_blocks(V, e):
    E = V["enc"][e]
    z, a, te = V["z"], E["a"], V["test"]
    s1 = z[e + "_s1_img"].astype(float)
    s2same = z[e + "_s2_img_same_id"].astype(float)
    cor = E["cor_g"]
    out = {}
    # global: m_id = s1 - s2 trên 145 identity, nhãn global
    s2 = np.sort(a, 1)[:, ::-1][:, 1]
    mid, mimg, same = a.max(1) - s2, s1 - np.maximum(s2same, s2), s2same > s2
    out["global"] = block(mid[te], mimg[te], cor[te], same[te], V["cl"][te], f"v9_{E['eid']}_test|global")
    # yc_subset: nhãn global = nhãn Y_c; m_id = s1_Yc - gap_Yc, s2 = s1_Yc - gap
    sub = te & (E["pred_g"] == E["pred_y"])
    s1y = E["ry"]["s1_yc"]
    s2y = s1y - E["gap"]
    mid, mimg, same = s1y - s2y, s1 - np.maximum(s2same, s2y), s2same > s2y
    b = block(mid[sub], mimg[sub], cor[sub], same[sub], V["cl"][sub], f"v9_{E['eid']}_test|yc_subset")
    b["n_subset"] = int(sub.sum())
    out["yc_subset"] = b
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Y.default_out("imggap")))
    args = ap.parse_args(argv)
    rows = ab4_rows()
    V = Y.load_v9()
    R = {"deviation": "D16", "post_hoc": True, "seed": Y.SEED, "n_boot": Y.B_BOOT,
         "label": "post hoc, descriptive: image-level gap m_img vs identity gap m_id",
         "sources": [Y.rel(Y.AB4 / "predictions.jsonl")] + Y.v9_sources(),
         "rng_substream_rule": "default_rng([20261022, zlib.crc32(tag)]); tag = ten dataset (| khoi); cum = identity",
         "ab4_all_top2_same_identity": int(sum(bool(r["top2_same_identity"]) for r in rows)),
         "datasets": {}}
    R["datasets"]["ab4_test"] = ab4_block([r for r in rows if r["split"] == "test" and r["n_c"] >= 3], "ab4_test")
    R["datasets"]["ab4_calib"] = ab4_block([r for r in rows if r["split"] == "calib"], "ab4_calib")
    for e in Y.ENC:
        R["datasets"][f"v9_{Y.ENC[e]}_test"] = v9_blocks(V, e)
    d = R["datasets"]
    dev = ("## D16 — Image-level gap m_img vs identity gap m_id (post hoc)\n\n"
           "- Ngày: %s. Script: `03_experiments/scripts/v9rev_imggap.py` (Python311, numpy). Đặc tả: `05_submission/spec_v9_revision.md` mục 5 [C3].\n"
           "- Dữ liệu: ab4 `predictions.jsonl` (cột m_id, m_img, top2_same_identity), v9 `pass1_capture.npz` (s1_img, s2_img_same_id, id_scores). Không đổi dữ liệu sealed.\n"
           "- Δ E-AURC = E(m_img) - E(m_id), bootstrap cụm identity (B=2000). ab4_test n=%d, ab4_calib n=%d; top2_same_identity trên 284 query = %d.\n"
           "- Phân tích mô tả post hoc; không điều chỉnh thiết kế hay kết luận của bài.\n") % (
        datetime.date.today().isoformat(), d["ab4_test"]["n"], d["ab4_calib"]["n"], R["ab4_all_top2_same_identity"])
    Y.finalize(Path(args.out_dir), R, dev)
    print(json.dumps({k: v.get("paired_delta_eaurc_mimg_minus_mid") for k, v in d.items() if "paired_delta_eaurc_mimg_minus_mid" in v}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
