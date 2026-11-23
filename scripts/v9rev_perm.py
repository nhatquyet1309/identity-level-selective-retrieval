# -*- coding: utf-8 -*-
"""C6 [D19] v9rev_perm: kiem dinh hoan vi doi dau theo cum identity (paired) cho 3 so sanh da bao McNemar. Post hoc.
Chay: Python311 v9rev_perm.py [--date YYYYMMDD]. Dac ta: 05_submission/spec_v9_revision.md muc 5 (C6)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent
RES = EXP / "results"
SEED = 20261022
B = 100000
CHUNK = 10000
CC = RES / "C_caseI_20261119" / "619"
LOO = RES / "preregv7_tierA_20261117" / "loo_Aii.csv"
SOURCES = [
    "03_experiments/results/C_caseI_20261119/619/predictions_siglip619.jsonl",
    "03_experiments/results/C_caseI_20261119/619/predictions_clipb619.jsonl",
    "03_experiments/results/5070-20261026-ab4/predictions.jsonl",
    "03_experiments/results/preregv7_tierA_20261117/loo_Aii.csv",
]
NOTE = ("McNemar treats the paired cases as independent; cases from the same identity are correlated (identity clusters), "
        "so its p-value is anti-conservative. The permutation p flips the sign of whole identity clusters (paired, two-sided) and "
        "respects this clustering; it is a post hoc sensitivity check, not a replacement of the preregistered test.")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_outputs(out, results, dev_text):
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "deviation_entry.md").write_text(dev_text, encoding="utf-8")
    lines = ["%s  %s" % (hashlib.sha256((out / n).read_bytes()).hexdigest(), n) for n in ("results.json", "deviation_entry.md")]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)


def cluster_signflip(d, clusters, B=B, seed=SEED):
    """d_i in {-1,0,1}; T = sum d / n; lat dau ca cum voi xac suat 1/2; hai phia; p = (1 + #{|T*|>=|T_obs|}) / (B + 1)."""
    d = np.asarray(d, float)
    uniq, inv = np.unique(clusters, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(uniq))
    obs = abs(s.sum()) / len(d)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B // CHUNK):
        sg = rng.integers(0, 2, size=(CHUNK, len(uniq))) * 2 - 1
        cnt += int((np.abs(sg @ s) / len(d) >= obs - 1e-12).sum())
    return (1 + cnt) / (B + 1)


def comparisons():
    S, K = jl(CC / "predictions_siglip619.jsonl"), jl(CC / "predictions_clipb619.jsonl")
    Sd, Kd = {r["query_id"]: r for r in S}, {r["query_id"]: r for r in K}
    ok = lambda r: r["split"] == "test" and r["n_c"] >= 3 and r["gt_identity"] in r["Yc"]
    q = sorted(x for x in Sd if ok(Sd[x]) and ok(Kd[x]))
    out = {"tierC_siglip2_vs_clip": (np.array([int(Sd[x]["correct"]) for x in q]), np.array([int(Kd[x]["correct"]) for x in q]),
                                      np.array([Sd[x]["gt_identity"] for x in q]), "A = SigLIP2-I, B = CLIP-B-I (619 index, paired by query_id)")}
    rows = [r for r in csv.DictReader(open(LOO, encoding="utf-8")) if r["in_D_Aii"] == "1"]
    ident = np.array([r["identity"] for r in rows])
    for e in ("siglip2", "clip"):
        out["tierA_%s_I_vs_S" % e] = (np.array([int(r[e + "_I_Aii_correct"]) for r in rows]),
                                      np.array([int(r[e + "_S_Aii_correct"]) for r in rows]), ident,
                                      "A = %s image-only (I), B = %s with text/S (S) on D(ii)" % (e, e))
    return out


def compute():
    comps = {}
    for name, (a, b, cl, desc) in comparisons().items():
        bb, cc = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
        comps[name] = {
            "description": desc, "n": int(len(a)), "n_clusters": int(len(np.unique(cl))), "b": bb, "c": cc,
            "delta": float((a.sum() - b.sum()) / len(a)),
            "mcnemar_exact_p": mcnemar_exact(bb, cc),
            "perm_p": cluster_signflip(a - b, cl), "B": B, "seed": SEED, "p_min_resolvable": 1.0 / (B + 1),
            "interpretation_note": NOTE,
        }
    return {
        "deviation": "D19", "post_hoc": True,
        "label": "post hoc cluster sign-flip permutation test (descriptive sensitivity to McNemar independence)",
        "seed": SEED, "sources": SOURCES, "comparisons": comps,
    }


DEV = """## D19 — Kiểm định hoán vị đổi dấu theo cụm identity (bổ sung McNemar) — 22/11/2026

**Phạm vi.** Phân tích post hoc mô tả (C6 revision v9). Chỉ đọc dữ liệu sealed; McNemar đã báo giữ nguyên. Script `scripts/v9rev_perm.py`; kết quả `results/v9rev_perm_{date}/results.json`.

**Phương pháp.** d_i = đúng_A − đúng_B; T = tổng d / n; lật dấu cả cụm identity với xác suất 1/2 (paired, hai phía); Monte Carlo B = 100000, seed 20261022, p = (1 + #{|T*| >= |T_obs|}) / (B + 1), p tối thiểu phân giải được 1/(B+1). Ba so sánh: Tier C SigLIP2-I vs CLIP-B-I (160 query, 73 cụm), Tier A SigLIP2 I vs S và CLIP I vs S (1663 dòng, 141 cụm).

**Diễn giải.** McNemar coi các ca độc lập; ca cùng identity tương quan nên p McNemar chống bảo thủ. p hoán vị theo cụm là kiểm tra độ nhạy post hoc, không thay phép kiểm đã đăng ký.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20261122")
    a = ap.parse_args()
    res = compute()
    write_outputs(RES / ("v9rev_perm_" + a.date), res, DEV.replace("{date}", a.date))
    for k, r in res["comparisons"].items():
        print(k, {x: r[x] for x in ("n", "n_clusters", "b", "c", "delta", "mcnemar_exact_p", "perm_p")})
