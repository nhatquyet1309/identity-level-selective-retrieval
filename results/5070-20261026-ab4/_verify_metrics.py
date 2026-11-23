#!/usr/bin/env python3
"""Doi chieu eval_metrics.py voi so cua run ab4 (tinh lai tu predictions.jsonl)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"<REDACTED_PATH>")
from eval_metrics import aurc, auroc, coverage_at_risk, e_aurc  # noqa: E402

RUN = Path(__file__).resolve().parent
preds = [json.loads(l) for l in open(RUN / "predictions.jsonl")]
met = json.load(open(RUN / "metrics.json"))
rows = [p for p in preds if p["split"] == "test" and p["n_c"] >= 3]
T2 = float(met["paired"]["T_star"]["Sc2"])   # 0.02
T3 = float(met["paired"]["T_star"]["Sc3"])   # 0.01
import math

scores = {"Sc1": [], "Sc2": [], "Sc3": [], "Sc4": []}
corr = []
for p in rows:
    yc = p["Yc"]
    sy = p["s_y"]
    v = sorted(sy.values(), reverse=True)
    s1 = v[0] if v else 0.0
    s2 = v[1] if len(v) > 1 else 0.0
    corr.append(1 if p["y1"] == p["gt_identity"] else 0)
    scores["Sc1"].append(s1)
    scores["Sc4"].append(s1 - s2)
    # MSP_T (T*)
    z = [x / T2 for x in v]
    m = max(z)
    e = [math.exp(zz - m) for zz in z]
    scores["Sc2"].append(max(e) / sum(e))
    # Energy_T (T*, lower = trusted) -> dao dau cho higher_better=False
    z3 = [x / T3 for x in v]
    m3 = max(z3)
    lse3 = m3 + math.log(sum(math.exp(zz - m3) for zz in z3))
    scores["Sc3"].append(T3 * lse3)

print("n_test_nc3:", len(rows), "top1:", sum(corr) / len(corr))
ref = met["strata"]["n_c>=3"]["scores"]
print(f"{'score':6} {'AURC(moi)':>10} {'AURC(run)':>10} {'dAURC':>9} "
      f"{'EAURC(moi)':>11} {'EAURC(run)':>11} {'dEAURC':>9}")
for sc in ("Sc1", "Sc2", "Sc3", "Sc4"):
    # Sc3: luu duong (cao = it tin) -> higher_better=True (so sanh truc tiep)
    a = aurc(scores[sc], corr)
    e = e_aurc(scores[sc], corr)
    # lay hang AURC/E-AURC cua run: Sc2/Sc3 o T* (hang dau tien co trong metrics)
    key = sc if sc in ("Sc1", "Sc4") else sc + "_T%.02f" % (T3 if sc == "Sc3" else T2)
    if key not in ref:
        cand = [k for k in ref if k.startswith(sc)]
        key = cand[0]
    ra = ref[key]["aurc"]
    re_ = ref[key]["e_aurc"]
    print(f"{sc:6} {a:10.5f} {ra:10.5f} {a-ra:+9.5f} "
          f"{e:11.5f} {re_:11.5f} {e-re_:+9.5f}")
au = auroc(scores["Sc4"], corr)
print("Sc4 AUROC(moi) %.4f vs run %.4f" % (au, ref["Sc4"]["auroc"]))
cov, risk, n_acc, n_err, cp = coverage_at_risk(scores["Sc4"], corr, 0.55)
print("Sc4 cov@tau0.55: cov=%.3f risk=%.3f n_acc=%d n_err=%d cp=%.4f"
      % (cov, risk, n_acc, n_err, cp))
