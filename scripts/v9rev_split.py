# -*- coding: utf-8 -*-
"""C1 [D14] v9rev_split: bang split/thuat ngu cho ab4, Tier C (619) va v9 RQ-1. Post hoc, chi doc du lieu sealed.
Chay: Python311 v9rev_split.py [--date YYYYMMDD]. Dac ta: 05_submission/spec_v9_revision.md muc 2, 5(C1)."""
import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent
RES = EXP / "results"
SEED = 20261022
AB4 = RES / "5070-20261026-ab4"
CC = RES / "C_caseI_20261119" / "619"
V9 = RES / "prereg_v9_gap_20261121"
SOURCES = [
    "03_experiments/results/5070-20261026-ab4/split.json",
    "03_experiments/results/5070-20261026-ab4/predictions.jsonl",
    "03_experiments/results/C_caseI_20261119/619/predictions_siglip619.jsonl",
    "03_experiments/results/C_caseI_20261119/619/predictions_clipb619.jsonl",
    "03_experiments/results/C_caseI_20261119/619/metrics_siglip619.json",
    "03_experiments/results/rq1_20261119/split.json",
    "03_experiments/results/prereg_v9_gap_20261121/pass1_meta.json",
]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_outputs(out, results, dev_text):
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "deviation_entry.md").write_text(dev_text, encoding="utf-8")
    lines = ["%s  %s" % (hashlib.sha256((out / n).read_bytes()).hexdigest(), n) for n in ("results.json", "deviation_entry.md")]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute():
    sp = jload(AB4 / "split.json")
    idx, q, so = sp["index"], sp["query"], sp["split_of_query"]
    idx_ids = {x["variant"] for x in idx}
    q_ids = {x["variant"] for x in q}
    cal_i = [int(i) for i, v in so.items() if v == "calib"]
    te_i = [int(i) for i, v in so.items() if v == "test"]
    cal_ids = {q[i]["variant"] for i in cal_i}
    te_ids = {q[i]["variant"] for i in te_i}
    ab4 = {
        "index": {"n_images": len(idx), "n_identities": len(idx_ids)},
        "query": {"n_images": len(q), "n_identities": len(q_ids)},
        "calibration": {"n_images": len(cal_i), "n_identities": len(cal_ids)},
        "test": {"n_images": len(te_i), "n_identities": len(te_ids)},
        "overlap": {
            "index_query_images": len({x["path"] for x in idx} & {x["path"] for x in q}),
            "calib_test_images": len(set(cal_i) & set(te_i)),
            "calib_test_identities": len(cal_ids & te_ids),
            "query_identities_not_in_index": len(q_ids - idx_ids),
        },
        "protocol": ("gallery_holdout: held-out query images of known identities; %.0f%% of the images of each identity are set aside "
                     "as query (seed %d); every query identity has other images in the index; calibration/test are split by image, "
                     "so they share identities" % (sp["frac"] * 100, sp["seed"])),
    }

    S, K = jl(CC / "predictions_siglip619.jsonl"), jl(CC / "predictions_clipb619.jsonl")
    ab = jl(AB4 / "predictions.jsonl")
    o169 = {r["query_id"] for r in ab if r["split"] == "test" and r["n_c"] >= 3}
    ok = lambda r: r["split"] == "test" and r["n_c"] >= 3 and r["gt_identity"] in r["Yc"]
    Sd, Kd = {r["query_id"]: r for r in S}, {r["query_id"]: r for r in K}
    pairs = sorted(x for x in Sd if ok(Sd[x]) and ok(Kd[x]))
    idset = set()
    for r in S:
        idset |= set(r["Yc"])
    m619 = jload(CC / "metrics_siglip619.json")
    n_index = int(m619["data"]["n_index"])
    tierC = {
        "index": {"n_images": n_index, "n_identities": len(idset)},
        "dropped_index_images": len(idx) - n_index,
        "lost_identities": sorted(idx_ids - idset),
        "test_pairs": {"n_before": len(o169), "n_after": len(pairs), "n_excluded": len(o169 - set(pairs)),
                       "n_clusters": len({Sd[x]["gt_identity"] for x in pairs})},
        "excluded_qids": sorted(o169 - set(pairs)),
        "overlap": {"query_identities_not_in_index": len(q_ids - idset)},
        "protocol": ("same split as ab4; index reduced to the %d images whose sources are still resolvable; "
                     "held-out query images; queries whose identity lost every index image (GT outside Y_c) are excluded from the paired test set" % n_index),
    }

    r1 = jload(RES / "rq1_20261119" / "split.json")
    m = jload(V9 / "pass1_meta.json")
    nq = {s: sum(1 for x in m["q_split"] if x == s) for s in ("calib", "test")}
    v9 = {
        "gallery": {"n_images": m["n_gallery"], "n_identities": m["n_identities"]},
        "calibration": {"n_identities": len(r1["calib"]), "n_query_rows": nq["calib"]},
        "test": {"n_identities": len(r1["test"]), "n_query_rows": nq["test"]},
        "overlap": {"calib_test_identities": len(set(r1["calib"]) & set(r1["test"]))},
        "query_protocol": ("each gallery image is a query under leave-one-out with a near-duplicate mask (cosine theta = %s); "
                           "the identity is always present in the gallery; the identity-disjoint property applies only to "
                           "calibration versus test lists, not to query versus gallery" % m["theta_cos"]),
    }
    return {
        "deviation": "D14", "post_hoc": True,
        "label": "post hoc descriptive split/terminology audit (no new statistics)",
        "seed": SEED, "sources": SOURCES,
        "datasets": {"ab4": ab4, "tierC_619": tierC, "v9_rq1": v9},
    }


DEV = """## D14 — Bảng split và thuật ngữ (ab4, Tier C 619, v9 RQ-1) — 22/11/2026

**Phạm vi.** Phân tích post hoc mô tả (C1 revision v9). Chỉ đọc dữ liệu sealed; không sửa file sealed. Script `scripts/v9rev_split.py`; kết quả `results/v9rev_split_{date}/results.json`.

**Phát hiện.**
- ab4: 790 ảnh index/100 identity; 284 ảnh query/95 identity (95 ⊂ 100); calibration 114 ảnh/66 identity, test 170 ảnh/80 identity, chung 51 identity (tách theo ảnh). Là "unseen query images of known identities", KHÔNG identity-disjoint giữa calibration/test.
- Tier C (619): index 619 ảnh/93 identity (171 ảnh mất; 7 identity mất hẳn); test ghép cặp 169 -> 160 (9 query loại), 73 cụm identity; vẫn là ảnh query held-out của identity đã có trong index.
- v9 RQ-1: gallery 1667 ảnh/145 identity; calibration 73 identity/831 dòng, test 72 identity/832 dòng; 0 identity chung (identity-disjoint chỉ đúng cho calibration vs test).

**Hiệu chỉnh.** Nhãn "Tier C (unseen identities)" của v8 bị loại; dùng "held-out query images" (xem spec_v9_revision.md [T1]-[T6]). Không số nào mới: mọi số tái tính từ split.json/predictions/pass1_meta.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20261122")
    a = ap.parse_args()
    res = compute()
    write_outputs(RES / ("v9rev_split_" + a.date), res, DEV.replace("{date}", a.date))
    print(json.dumps(res["datasets"], indent=1, ensure_ascii=False))
