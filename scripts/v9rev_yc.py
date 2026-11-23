#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[C2] v9 revision, D15 — 4 score (s1, MSP, energy, gap) cùng tính TRONG Y_c, hai regime nhãn.

Regime `yc_argmax` (CHÍNH): nhãn dự đoán = argmax trong Y_c (ties -> chỉ số nhỏ nhất).
Regime `global_argmax`: nhãn sealed `*_correct` (argmax 145 identity), tương thích D13.
Post hoc, chỉ dữ liệu local sealed; T* sealed (không chọn lại). Đặc tả: 05_submission/spec_v9_revision.md mục 5 [C2].
Module này cũng chứa helper dùng chung cho v9rev_imggap.py và v9rev_gate.py.
Chạy bằng Python311: python v9rev_yc.py [--out-dir DIR]
"""
import argparse
import datetime
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d13_yc_sensitivity as D13  # noqa: E402  (tái dùng load_yc_by_coarse, compute_yc_scores)
import prereg_v9_stats as S  # noqa: E402

EXP = HERE.parent
REPO = EXP.parent
RES = EXP / "results"
V9 = RES / "prereg_v9_gap_20261121"
AB4 = RES / "5070-20261026-ab4"
SEED = 20261022
B_BOOT = 2000
ENC = {"siglip2": "siglip2_I", "clip": "clip_I"}


# ───────────────────────── thống kê dùng chung ─────────────────────────
def eaurc(score, correct):
    """E-AURC: xếp giảm dần (mergesort ổn định), risk tích lũy, trừ đường tối ưu max(0,k-kc)/k."""
    s = np.asarray(score, float)
    c = np.asarray(correct, int)
    n = len(c)
    order = np.argsort(-s, kind="mergesort")
    k = np.arange(1, n + 1)
    risk = np.cumsum(1 - c[order]) / k
    kc = int(c.sum())
    opt = np.maximum(0, k - kc) / k
    return float(risk.mean() - opt.mean())


def auroc(score, correct):
    """AUROC của score dự đoán 'đúng' (Mann-Whitney, ties = 0,5). None nếu một lớp rỗng."""
    s = np.asarray(score, float)
    c = np.asarray(correct, int)
    n_pos = int(c.sum())
    n_neg = len(c) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sv = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[c == 1].sum()
    return float((rp - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cluster_boot(fn, clusters, tag, B=B_BOOT, seed=SEED):
    """Bootstrap theo cụm (identity). fn(idx) -> vector Δ. Trả (lo, hi) mảng percentile 2,5/97,5 (cùng resample cho mọi Δ)."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng([seed, zlib.crc32(tag.encode())])
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        out.append(np.atleast_1d(fn(idx)))
    out = np.asarray(out)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def label_of(lo, hi):
    """Δ = E(A) - E(B), âm = A tốt hơn. CI chứa 0 -> insufficient."""
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "insufficient"


def delta_block(delta, lo, hi):
    return {"delta": float(delta), "ci95_lo": float(lo), "ci95_hi": float(hi), "label": label_of(lo, hi)}


def finalize(out, R, dev_text):
    """Ghi results.json, deviation_entry.md, SHA256SUMS (mọi file trừ chính SHA)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "deviation_entry.md").write_text(dev_text, encoding="utf-8")
    lines = []
    for f in sorted(out.iterdir()):
        if f.is_file() and not f.name.startswith("SHA256SUMS"):
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel(p):
    return Path(p).resolve().relative_to(REPO).as_posix()


def default_out(name):
    return RES / f"v9rev_{name}_{datetime.date.today():%Y%m%d}"


# ───────────────────────── dữ liệu v9 ─────────────────────────
def load_v9():
    """Nạp pass1 (thứ tự gốc q_idx), Y_c từ pass2b_meta, T* sealed, dự đoán global/Y_c."""
    z = np.load(V9 / "pass1_capture.npz")
    meta = json.loads((V9 / "pass1_meta.json").read_text(encoding="utf-8"))
    p1 = json.loads((V9 / "p1_results.json").read_text(encoding="utf-8"))
    yc = D13.load_yc_by_coarse(meta["ids"], str(V9 / "pass2b_meta.json"))
    sp = np.array(meta["q_split"])
    cats = list(meta["q_categories"])
    gt = z["gt_idx"].astype(int)
    out = {"z": z, "meta": meta, "yc": yc, "cats": cats, "gt": gt, "test": sp == "test", "calib": sp == "calib",
           "cl": np.array(meta["q_identities"]), "enc": {}}
    for e, eid in ENC.items():
        a = np.asarray(z[f"{e}_id_scores"], dtype=np.float64)
        P = p1["encoders"][eid]
        Tm, Te = P["T_star_MSP"], P["T_star_energy"]
        ry = D13.compute_yc_scores(a, cats, yc, Tm, Te)
        pred_g = a.argmax(1)
        pred_y = np.empty(len(a), int)
        for i in range(len(a)):
            idx = np.asarray(yc[S.CAT_TO_COARSE[cats[i]]])
            pred_y[i] = idx[a[i, idx].argmax()]
        gap = np.where(np.isfinite(ry["gap"]), ry["gap"], -1e9)   # abstain như v9/D13
        out["enc"][e] = dict(eid=eid, a=a, Tm=Tm, Te=Te, ry=ry, gap=gap, pred_g=pred_g, pred_y=pred_y,
                             cor_g=z[f"{e}_correct"].astype(int), cor_y=(pred_y == gt).astype(int))
    return out


def v9_sources():
    return [rel(V9 / n) for n in ("pass1_capture.npz", "pass1_meta.json", "pass2b_meta.json", "p1_results.json")]


# ───────────────────────── C2 ─────────────────────────
def stratum_block(scores, cor, cl, tag):
    """scores: dict s1/MSP/energy/gap (đã cắt theo tầng)."""
    blk = {"n": int(len(cor)), "n_err": int((1 - cor).sum()), "metrics": {}}
    for k, s in scores.items():
        blk["metrics"][k] = {"e_aurc": eaurc(s, cor), "auroc": auroc(s, cor)}
    others = {"gap_minus_s1": "s1", "gap_minus_MSP": "MSP", "gap_minus_energy": "energy"}
    pt = {k: blk["metrics"]["gap"]["e_aurc"] - blk["metrics"][o]["e_aurc"] for k, o in others.items()}

    def fn(idx):
        c = cor[idx]
        g = eaurc(scores["gap"][idx], c)
        return np.array([g - eaurc(scores[o][idx], c) for o in others.values()])

    lo, hi = cluster_boot(fn, cl, tag)
    pd = {}
    for j, k in enumerate(others):
        pd[k] = delta_block(pt[k], lo[j], hi[j])
    pd["MSP_minus_gap"] = delta_block(-pt["gap_minus_MSP"], -hi[1], -lo[1])
    blk["paired_delta_eaurc"] = pd
    return blk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(default_out("yc")))
    args = ap.parse_args(argv)
    V = load_v9()
    te = V["test"]
    R = {"deviation": "D15", "post_hoc": True, "seed": SEED, "n_boot": B_BOOT,
         "label": "post hoc, descriptive: 4 score tren Y_c, hai regime nhan (yc_argmax chinh; global_argmax khop D13)",
         "sources": v9_sources() + [rel(RES / "D13_Yc_sensitivity_20261121" / "results.json")],
         "rng_substream_rule": "default_rng([20261022, zlib.crc32(tag)]); tag = '<eid>|<regime>|<stratum>'; cum = identity",
         "T_policy": "sealed_T_star_no_reselection",
         "regime_note": "yc_argmax: nhan = argmax trong Y_c (ties -> chi so nho nhat) la regime CHINH; "
                        "global_argmax: nhan sealed (argmax 145 identity), tuong thich D13.",
         "label_regimes": {"yc_argmax": {"encoders": {}}, "global_argmax": {"encoders": {}}},
         "label_agreement": {}, "encoder_summary": {"yc_argmax": {}, "global_argmax": {}}}
    for e, eid in ENC.items():
        E = V["enc"][e]
        ry = E["ry"]
        sc = {"s1": ry["s1_yc"], "MSP": ry["msp_yc"], "energy": ry["energy_yc"], "gap": E["gap"]}
        masks = {"all_832": te, "nc_ge3": te & (ry["n_c"] >= 3)}
        R["label_agreement"][eid] = {"n_total": int(len(E["a"])),
                                     "n_label_differ": int((E["pred_g"] != E["pred_y"]).sum()),
                                     "n_correctness_differ": int((E["cor_g"] != E["cor_y"]).sum())}
        for regime, cor in (("yc_argmax", E["cor_y"]), ("global_argmax", E["cor_g"])):
            strata = {}
            for sk, m in masks.items():
                strata[sk] = stratum_block({k: v[m] for k, v in sc.items()}, cor[m], V["cl"][m], f"{eid}|{regime}|{sk}")
            R["label_regimes"][regime]["encoders"][eid] = {"T_star_MSP": E["Tm"], "T_star_energy": E["Te"], "strata": strata}
            R["encoder_summary"][regime][eid] = {"accuracy_test": float(cor[te].mean()), "n": int(te.sum())}
    dev = ("## D15 — Độ nhạy Y_c: 4 score trên cùng candidate set, hai regime nhãn (post hoc)\n\n"
           "- Ngày: %s. Script: `03_experiments/scripts/v9rev_yc.py` (Python311, numpy). Đặc tả: `05_submission/spec_v9_revision.md` mục 5 [C2].\n"
           "- Dữ liệu: `pass1_capture.npz`, `pass1_meta.json`, `pass2b_meta.json` (Y_c đủ 145 identity), `p1_results.json` (T* sealed, không chọn lại). Không đổi dữ liệu sealed.\n"
           "- s1, MSP, energy, gap đều tính trong Y_c. Regime chính `yc_argmax` (nhãn = argmax trong Y_c); `global_argmax` (nhãn sealed) khớp D13 trong 1e-9.\n"
           "- Δ E-AURC = E(gap) - E(other), CI bootstrap cụm theo identity (B=2000, seed [20261022, crc32(tag)]); CI chứa 0 -> nhãn `insufficient`.\n"
           "- Số ca nhãn global khác nhãn Y_c: SigLIP2-I %d, CLIP-B-I %d (trong 1663); số ca đúng/sai khác: %d và %d.\n"
           "- Phân tích mô tả post hoc; không kết luận gap hơn MSP.\n") % (
        datetime.date.today().isoformat(), R["label_agreement"]["siglip2_I"]["n_label_differ"], R["label_agreement"]["clip_I"]["n_label_differ"],
        R["label_agreement"]["siglip2_I"]["n_correctness_differ"], R["label_agreement"]["clip_I"]["n_correctness_differ"])
    finalize(Path(args.out_dir), R, dev)
    print(json.dumps(R["label_agreement"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
