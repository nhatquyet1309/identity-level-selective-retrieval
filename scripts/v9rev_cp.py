# -*- coding: utf-8 -*-
"""C5 [D18] v9rev_cp: can tren Clopper-Pearson mot phia 95% theo nguong cho ab4 calib va v9 RQ-1 calib. Post hoc.
Chay: Python311 v9rev_cp.py [--date YYYYMMDD]. Dac ta: 05_submission/spec_v9_revision.md muc 5 (C5).
cp_upper dung lai tu prereg_v9_stats.py (sealed, chi import)."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent
RES = EXP / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prereg_v9_stats as S  # noqa: E402

SEED = 20261022
CONF = 0.95
ALPHAS = (("5pct", 0.05), ("1pct", 0.01))
AB4 = RES / "5070-20261026-ab4"
V9 = RES / "prereg_v9_gap_20261121"
V_OK, V_ABOVE, V_INS = "certified", "not certified: observed risk above target", "not certified: insufficient data"
SOURCES = [
    "03_experiments/results/5070-20261026-ab4/predictions.jsonl",
    "03_experiments/results/prereg_v9_gap_20261121/pass1_capture.npz",
    "03_experiments/results/prereg_v9_gap_20261121/pass1_meta.json",
    "03_experiments/results/prereg_v9_gap_20261121/pass2b_meta.json",
    "03_experiments/results/prereg_v9_gap_20261121/p1_results.json",
    "03_experiments/scripts/prereg_v9_stats.py",
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


def n_required(alpha):
    return int(math.ceil(math.log(0.05) / math.log(1 - alpha) - 1e-12))


def error_free_prefix(score, correct):
    o = np.argsort(-np.asarray(score, float), kind="mergesort")
    k = 0
    for i in o:
        if correct[i]:
            k += 1
        else:
            break
    return k


def verdict(cp, risk, alpha):
    if cp <= alpha:
        return V_OK
    return V_ABOVE if risk > alpha else V_INS


def block(score, correct):
    """Moi gia tri score phan biet = 1 hang; chap nhan khi score >= tau (ties cung hang)."""
    s = np.asarray(score, float)
    err = 1 - np.asarray(correct, int)
    n_cal = len(s)
    order = np.argsort(-s, kind="mergesort")
    ss, ee = s[order], err[order]
    cum_err = np.cumsum(ee)
    uniq = np.unique(s)[::-1]
    rows = []
    for tau in uniq:
        n = int(np.searchsorted(-ss, -tau, side="right"))   # so phan tu >= tau
        k = int(cum_err[n - 1])
        risk = k / n
        cp = float(S.cp_upper(k, n, 0.05))
        row = {"threshold": float(tau), "n_accepted": n, "n_errors": k, "observed_risk": risk, "cp_upper_95": cp}
        for tag, al in ALPHAS:
            row["verdict_" + tag] = verdict(cp, risk, al)
        rows.append(row)
    summ = {"max_error_free_prefix": error_free_prefix(s, np.asarray(correct, int))}
    for tag, al in ALPHAS:
        nc = sum(1 for r in rows if r["verdict_" + tag] == V_OK)
        bind = "insufficient data" if n_cal < n_required(al) else "observed risk above target"
        summ["n_certified_rows_" + tag] = nc
        summ["binding_constraint_" + tag] = bind
        summ["conclusion_" + tag] = V_OK if nc > 0 else "not certified: " + bind
    return {"rows": rows, "summary": summ}


def ab4_scores(rows, T_msp=0.02, T_en=0.01):
    s1, msp, en = [], [], []
    for r in rows:
        v = np.array(list(r["s_y"].values()), float)
        s1.append(v.max())
        z = v / T_msp
        e = np.exp(z - z.max())
        msp.append((e / e.sum()).max())
        z = v / T_en
        en.append(T_en * (z.max() + math.log(np.exp(z - z.max()).sum())))
    return {"s1": np.array(s1), "MSP": np.array(msp), "energy": np.array(en),
            "gap": np.array([r["m_id"] for r in rows], float), "m_img": np.array([r["m_img"] for r in rows], float)}


def v9_gap_yc(a, cats, yc):
    out = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        v = np.sort(a[i, yc[S.CAT_TO_COARSE[cats[i]]]].astype(float))[::-1]
        out[i] = v[0] - v[1]
    return out


def compute():
    cal = [r for r in jl(AB4 / "predictions.jsonl") if r["split"] == "calib"]
    cor = np.array([int(r["correct"]) for r in cal])
    sc = ab4_scores(cal)
    ab4 = {"n": len(cal), "n_errors": int((1 - cor).sum()),
           "encoder": "clip-vit-b32 (case I, image-only)", "temperatures": {"MSP": 0.02, "energy": 0.01},
           "scores": {k: block(v, cor) for k, v in sc.items()}}

    z = np.load(V9 / "pass1_capture.npz")
    m = jload(V9 / "pass1_meta.json")
    m2 = jload(V9 / "pass2b_meta.json")
    p1 = jload(V9 / "p1_results.json")
    yc = S.build_yc_indices(m["ids"], dict(zip(m2["q_identities"], m2["q_categories"])))
    cmask = np.array(m["q_split"]) == "calib"
    enc = {}
    for e, eid in (("siglip2", "siglip2_I"), ("clip", "clip_I")):
        a = z[e + "_id_scores"].astype(np.float64)
        c = z[e + "_correct"].astype(int)[cmask]
        P = p1["encoders"][eid]
        ac = a[cmask]
        scores = {"s1": ac.max(1), "MSP": S.msp_at_T(ac, P["T_star_MSP"]), "energy": S.energy_at_T(ac, P["T_star_energy"]),
                  "gap": v9_gap_yc(a, list(m["q_categories"]), yc)[cmask]}
        enc[eid] = {"T_star_MSP": P["T_star_MSP"], "T_star_energy": P["T_star_energy"],
                    "n_errors": int((1 - c).sum()), "scores": {k: block(v, c) for k, v in scores.items()}}
    v9 = {"n": int(cmask.sum()), "encoders": enc,
          "note": "s1/MSP/energy global over 145 identities at sealed T*; gap inside Y_c; sealed correct labels"}
    return {
        "deviation": "D18", "post_hoc": True,
        "label": "post hoc descriptive one-sided Clopper-Pearson upper bounds per threshold (no certification claim)",
        "seed": SEED, "sources": SOURCES, "confidence": CONF, "alpha_targets": [0.05, 0.01],
        "n_required_zero_error": {"0.05": n_required(0.05), "0.01": n_required(0.01)},
        "datasets": {"ab4_calib": ab4, "v9_rq1_calib": v9},
    }


def brief(R):
    out = []
    for nm, blk in [("ab4", R["datasets"]["ab4_calib"]["scores"])] + [
            (e, d["scores"]) for e, d in R["datasets"]["v9_rq1_calib"]["encoders"].items()]:
        for sname, b in blk.items():
            s = b["summary"]
            out.append("%-10s %-7s prefix=%-4d cert5=%-4d cert1=%-4d %s | %s" % (
                nm, sname, s["max_error_free_prefix"], s["n_certified_rows_5pct"], s["n_certified_rows_1pct"],
                s["conclusion_5pct"], s["conclusion_1pct"]))
    return "\n".join(out)


DEV = """## D18 — Cận trên Clopper–Pearson một phía theo ngưỡng (ab4 calibration, v9 calibration) — 22/11/2026

**Phạm vi.** Phân tích post hoc mô tả (C5 revision v9). Chỉ đọc dữ liệu sealed (ab4 predictions, v9 pass1 + T* sealed); không chọn lại T. Script `scripts/v9rev_cp.py`; kết quả `results/v9rev_cp_{date}/results.json`.

**Phương pháp.** Mỗi giá trị score phân biệt là một hàng ngưỡng (chấp nhận khi score >= tau, ties cùng hàng); cận trên CP một phía 95% của tỉ lệ lỗi (k/n; = 1 nếu k = n). certified nếu cận <= alpha; "observed risk above target" nếu k/n > alpha; còn lại "insufficient data". n cần cho 0 lỗi: 59 (5%), 299 (1%).

**Sửa câu sai v8.** Câu v8 "Calibration size exceeds the bound, so the binding constraint is the error rate" đúng nửa: ab4 có 114 mẫu calibration, prefix không lỗi tối đa 21 (< 59) nên mức 5% bất khả thi do tỉ lệ lỗi; mức 1% cần >= 299 mẫu chấp nhận không lỗi > 114 nên ràng buộc là thiếu dữ liệu.

**Nhãn.** Post hoc, mô tả; không phải chứng nhận.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20261122")
    a = ap.parse_args()
    res = compute()
    write_outputs(RES / ("v9rev_cp_" + a.date), res, DEV.replace("{date}", a.date))
    print(brief(res))
