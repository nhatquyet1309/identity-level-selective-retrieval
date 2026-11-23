#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ-1 harness — prereg_v7 §7 toàn bộ (SEALED e5080142…74d1d).

Các điều khoản hiện thực ở đây:
  §7.1  universe A(ii) trên D(ii), 6 ô; chia theo identity: sắp tên theo UTF-8
        byte, random.Random(20261022).shuffle, 73 calib / 72 test.
        (§7.1 gốc viết 151 → 76/75; D3 chốt universe thật 145 → 73/72 theo
        quyết định Director 18/11 — deviation đã khai, prereg KHÔNG sửa.)
  §7.2.1 phân phối điểm đúng/sai (median, Q1/Q3, P10..P90) + KS.
  §7.2.2 AUROC rank-based (tie = trung bình rank, tất định) + CI bootstrap.
  §7.2.3 precision(τ)/coverage(τ) trên mọi giá trị điểm khác nhau + AURC
        = (1/n) Σ_{k=1..n} risk@k  (AURC THƯỜNG, không phải E-AURC — vá 16/11).
  §7.3  τ* = ngưỡng nhỏ nhất có CP upper một phía 95% ≤ α = 1−p*; sàn
        c_min = 10%; D_cov + verdict theo CI (bootstrap placeholder).
  §7.4  transfer(τ_k) → D_prec, D_cov; ε_p = 2,0pp, δ* = 5,0pp.

RÀNG BUỘC: Python thuần stdlib. Không numpy/scipy/PIL, không đọc index,
không remote shell. `cp_upper` tự cài bằng regularized incomplete beta (continued
fraction) rồi nghịch bằng bisection — KHÔNG dùng bảng tra/scipy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

# Import tiện ích chung từ module bậc B (cùng thư mục scripts/) — tránh nhân đôi
# percentile / RNG bootstrap. Chạy trực tiếp `python scripts/rq1_harness.py`
# hay import từ tests_local đều thấy scripts/ trên sys.path.
try:
    from tierB_part2 import SEED, N_BOOT, percentile, ci95, cluster_draws
except ImportError:  # khi import từ nơi khác
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tierB_part2 import SEED, N_BOOT, percentile, ci95, cluster_draws

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "results" / "preregv7_tierA_20261117" / "loo_Ai.csv"
DEFAULT_AUDIT = DEFAULT_CSV.with_name("identity_audit.json")

P_STAR = 0.95           # §0.1 / §7.3 — khai trước, không tối ưu sau
C_MIN = 0.10            # §7.3 — sàn độ phủ 10% (chốt 16/11)
EPS_P = 2.0             # §7.4 — biên precision 2,0 điểm % (chốt 16/11)
DELTA_STAR = 5.0        # §0.1 — δ* = 5 điểm %


# ---------------------------------------------------------------------------
# §7.1 — Universe & split
# ---------------------------------------------------------------------------

def _identity_names_from_csv(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        return sorted({r["identity"] for r in csv.DictReader(fh)},
                      key=lambda s: s.encode("utf-8"))


def make_split(names=None, *, csv_path=None, audit_path=None,
               seed=SEED, calib_n=73, verbose=True):
    """§7.1 (+ D3) — chia identity thành 73 calib / 72 test, tái lập được.

    `names`: danh sách tên identity; nếu None đọc cột `identity` của
    `csv_path` (mặc định `loo_Ai.csv`). Sắp theo UTF-8 byte rồi
    `random.Random(seed).shuffle` — đúng lời prereg, KHÔNG numpy RNG.

    LƯU Ý NGUỒN TÊN (đã đối chiếu 18/11): `identity_audit.json` chỉ ghi
    `counts.folders_with_images = 145` + 6 tên folder rỗng, KHÔNG chứa danh
    sách 145 tên. Danh sách tên lấy từ `loo_Ai.csv` (145 tên duy nhất);
    `loo_summary.json.per_identity` chỉ có 141 vì loại 4 identity 1 ảnh khỏi
    D(i). Hàm kiểm chứng số lượng với audit trước khi trả kết quả.

    In danh sách + 2 hash để reproduce. Trả dict gồm calib/test/order_hash/
    split_hash/n/seed/source.
    """
    source = "provided"
    if names is None:
        csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
        names = _identity_names_from_csv(csv_path)
        source = str(csv_path)
    names = list(names)
    if audit_path is not None:
        with Path(audit_path).open(encoding="utf-8") as fh:
            audit = json.load(fh)
        n_audit = audit.get("counts", {}).get("folders_with_images")
        if n_audit is not None and n_audit != len(names):
            raise ValueError(f"D3: audit nói {n_audit} identity có ảnh, "
                             f"nhưng nguồn tên có {len(names)}")
    names.sort(key=lambda s: s.encode("utf-8"))

    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    calib, test = shuffled[:calib_n], shuffled[calib_n:]

    order_hash = hashlib.sha256("\n".join(shuffled).encode("utf-8")).hexdigest()
    split_hash = hashlib.sha256(
        ("C:" + ",".join(calib) + "\nT:" + ",".join(test)).encode("utf-8")
    ).hexdigest()

    if verbose:
        print(f"§7.1 seed={seed}  universe={len(shuffled)} identity  "
              f"calib={len(calib)} / test={len(test)}")
        print("CALIB (73): " + ", ".join(calib))
        print("TEST  (72): " + ", ".join(test))
        print(f"order_sha256 = {order_hash}")
        print(f"split_sha256 = {split_hash}")
    return {
        "calib": calib, "test": test, "seed": seed,
        "n": len(shuffled), "n_calib": len(calib), "n_test": len(test),
        "order_hash": order_hash, "split_hash": split_hash, "source": source,
    }


# ---------------------------------------------------------------------------
# §7.2.1 — Phân phối điểm + KS
# ---------------------------------------------------------------------------

def _point_stats(xs):
    return {
        "n": len(xs),
        "median": percentile(xs, 50),
        "q1": percentile(xs, 25), "q3": percentile(xs, 75),
        "p10": percentile(xs, 10), "p30": percentile(xs, 30),
        "p50": percentile(xs, 50), "p70": percentile(xs, 70),
        "p90": percentile(xs, 90),
    }


def ks_distance(a, b):
    """Khoảng cách KS của hai mẫu thực (§7.2.1), ECDF trên mọi giá trị xuất hiện."""
    if not a or not b:
        raise ValueError("KS cần cả hai mẫu không rỗng")
    sa, sb = sorted(a), sorted(b)
    i = j = 0
    best = 0.0
    for v in sorted(set(a) | set(b)):
        while i < len(sa) and sa[i] <= v:
            i += 1
        while j < len(sb) and sb[j] <= v:
            j += 1
        best = max(best, abs(i / len(sa) - j / len(sb)))
    return best


def score_distribution(correct, scores):
    """§7.2.1 — điểm top-1: đúng vs sai; median/Q1/Q3/P10..P90 + KS."""
    pos = [float(s) for s, c in zip(scores, correct) if int(c) == 1]
    neg = [float(s) for s, c in zip(scores, correct) if int(c) == 0]
    return {
        "correct": _point_stats(pos) if pos else None,
        "wrong": _point_stats(neg) if neg else None,
        "ks": ks_distance(pos, neg) if pos and neg else None,
    }


# ---------------------------------------------------------------------------
# §7.2.2 — AUROC rank-based + CI bootstrap theo identity
# ---------------------------------------------------------------------------

def auroc(correct, scores):
    """AUROC = U/(n_pos·n_neg), rank TRUNG BÌNH cho hòa điểm (tất định).

    Điểm cao = tin cậy hơn (đúng hướng sau fix của deviation 22/10). Trả None
    nếu một trong hai lớp vắng mặt. Tie được xử lý bằng average rank nên không
    phụ thuộc thứ tự input — đúng yêu cầu "tie-handling tất định".
    """
    n = len(scores)
    labels = [int(c) for c in correct]
    npos, nneg = sum(labels), n - sum(labels)
    if npos == 0 or nneg == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    u = sum(ranks[i] for i in range(n) if labels[i] == 1) - npos * (npos + 1) / 2.0
    return u / (npos * nneg)


def auroc_ci(correct, scores, clusters, *, n_boot=N_BOOT, seed=SEED):
    """§7.2.2 — CI cluster-bootstrap theo identity (145 cụm, D3)."""
    labels = [int(c) for c in correct]
    keys = list(range(len(scores)))
    groups = {}
    for i in keys:
        groups.setdefault(clusters[i], []).append(i)
    reps = []
    for draw in _draw_groups(groups, n_boot=n_boot, seed=seed):
        v = auroc([labels[i] for i in draw], [scores[i] for i in draw])
        if v is not None:
            reps.append(v)
    if not reps:
        raise ValueError("không có lượt bootstrap có đủ 2 lớp")
    return {"auroc": auroc(labels, scores), "ci95": ci95(reps),
            "n_boot": len(reps), "seed": seed, "n_clusters": len(groups)}


def _draw_groups(groups, n_boot=N_BOOT, seed=SEED):
    """Generator: mỗi lượt trả danh sách chỉ số hàng của các cụm đã resample."""
    names = list(groups)
    draws, n = cluster_draws(len(names), n_boot=n_boot, seed=seed)
    for b in range(n_boot):
        idx = []
        base = b * n
        for j in range(n):
            idx.extend(groups[names[draws[base + j]]])
        yield idx


# ---------------------------------------------------------------------------
# §7.2.3 — precision/coverage + AURC
# ---------------------------------------------------------------------------

def precision_at_tau(scores, correct, tau):
    """precision(τ) = đúng trong {score ≥ τ} / số chấp nhận; None nếu rỗng."""
    acc = [int(c) for s, c in zip(scores, correct) if s >= tau]
    return (sum(acc) / len(acc)) if acc else None


def coverage_at_tau(scores, tau):
    """coverage(τ) = số {score ≥ τ} / n."""
    return sum(1 for s in scores if s >= tau) / len(scores)


def precision_coverage_curve(correct, scores, *, thresholds=None, keys=None):
    """§7.2.3 — bảng precision(τ)/coverage(τ) + AURC thường (không bù).

    AURC = (1/n) Σ_{k=1..n} risk@k, risk@k = tỉ lệ sai trong k ảnh điểm CAO
    NHẤT; thấp hơn tốt. KHÔNG phải E-AURC (chỉ ở bước 1 T* của §7.3 cho
    MSP/energy).

    Hòa điểm: xếp theo (−score, khoá nội dung) theo đúng QUY ƯỚC §4.5
    (`(−score, path)` cho ảnh, `(−score, identity_id)` sau gộp). Truyền
    `keys=path` từ CLI; nếu không có thì fallback chỉ số hàng (vẫn tất định
    trong một lần chạy, nhưng đổi thứ tự CSV sẽ đổi AURC — tránh khi chạy thật).
    """
    labels = [int(c) for c in correct]
    n = len(labels)
    if n == 0:
        raise ValueError("curve cần ít nhất 1 mẫu")
    ths = sorted(set(scores), reverse=True) if thresholds is None else list(thresholds)
    prec, cov = [], []
    for tau in ths:
        p = precision_at_tau(scores, labels, tau)
        prec.append(p)
        cov.append(coverage_at_tau(scores, tau))
    tie_key = (lambda i: keys[i]) if keys is not None else (lambda i: i)
    order = sorted(range(n), key=lambda i: (-scores[i], tie_key(i)))
    wrong = 0
    risk_sum = 0.0
    risks = []
    for k, i in enumerate(order, 1):
        wrong += 1 - labels[i]
        r = wrong / k
        risks.append(r)
        risk_sum += r
    return {
        "thresholds": ths, "precision": prec, "coverage": cov,
        "aurc": risk_sum / n, "risk_at_k": risks, "n": n,
    }


# ---------------------------------------------------------------------------
# Clopper–Pearson upper, tự cài (§7.3) — beta tiếp tục + bisection
# ---------------------------------------------------------------------------

def _betacf(a, b, x):
    """Regularized incomplete beta, continued fraction (Lentz, thuần stdlib)."""
    maxit, eps, fpmin = 500, 1e-15, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """I_x(a,b) = regularized incomplete beta."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def cp_upper(k, n, conf=0.95):
    """Cận trên Clopper–Pearson MỘT PHÍA tại confidence `conf`.

    Nghĩa: p_up = I^{-1}_{conf}(k+1, n−k) — nghiệm của
        I_{p_up}(k+1, n−k) = conf
    tương đương Σ_{i=0..k} C(n,i) p_up^i (1−p_up)^{n−i} = 1 − conf = α.
    Với k=0 có dạng đóng p_up = 1 − (1−conf)^{1/n} (dùng làm đường tắt,
    đồng thời là điểm đối chiếu: cp_upper(0,59,.95)=0.0495,
    cp_upper(0,114,.95)=0.0259 — §7.3 / paper §4).

    Không scipy, không bảng tra: nghịch I_x bằng bisection trên [0,1].
    """
    if n <= 0:
        raise ValueError("n phải > 0")
    if not (0 <= k <= n):
        raise ValueError(f"k={k} ngoài [0,{n}]")
    if not (0.0 < conf < 1.0):
        raise ValueError("conf phải nằm trong (0,1)")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    a, b = k + 1.0, float(n - k)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _betai(a, b, mid) < conf:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _binomial_cdf(k, n, p):
    """Σ_{i=0..k} C(n,i) p^i (1−p)^{n−i} — chỉ dùng để tự kiểm chứng CP."""
    return sum(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
               for i in range(k + 1))


# ---------------------------------------------------------------------------
# §7.3 — τ*, sàn độ phủ, D_cov + verdict
# ---------------------------------------------------------------------------

def _calib_pairs(calib):
    """Chuẩn hoá calib: list (score, correct) hoặc dict {scores, correct}."""
    if isinstance(calib, dict):
        return list(zip(calib["scores"], calib["correct"]))
    return [(s, c) for s, c in calib]


def tau_star(calib, p_star=P_STAR):
    """§7.3 bước 2 — τ* = ngưỡng NHỎ NHẤT có CP upper 95% ≤ α = 1−p*.

    Score = điểm top-1 theo identity (§7.3), không có nhiệt độ → BỎ bước 1
    T* (N/A cho họ cosine — xác nhận 17/11, §9.2). Duyệt mọi giá trị điểm
    khác nhau làm τ; chấp nhận score ≥ τ; k = số sai, n = số chấp nhận;
    cp_upper(k, n, conf=p*) ≤ α. Không có τ khả thi → `feasible=False`,
    `tau=None`, và cặp chứa ô này là `KHÔNG ĐẠT p*` (không nới lỏng).

    Trả dict: tau, feasible, alpha, n_calib, cp_at_tau, n_at_tau,
    k_at_tau, precision_at_tau.
    """
    pairs = _calib_pairs(calib)
    if not pairs:
        raise ValueError("calib rỗng")
    alpha = 1.0 - p_star
    scores = [float(s) for s, _ in pairs]
    labels = [int(c) for _, c in pairs]
    best = None
    for tau in sorted(set(scores), reverse=True):
        acc = [labels[i] for i in range(len(scores)) if scores[i] >= tau]
        n = len(acc)
        if n == 0:
            continue
        k = n - sum(acc)
        cp = cp_upper(k, n, conf=p_star)
        if cp <= alpha + 1e-12:
            if best is None or tau < best["tau"]:
                best = {"tau": tau, "cp_at_tau": cp, "n_at_tau": n,
                        "k_at_tau": k, "precision_at_tau": sum(acc) / n}
    if best is None:
        return {"tau": None, "feasible": False, "alpha": alpha,
                "n_calib": len(pairs), "cp_at_tau": None,
                "n_at_tau": None, "k_at_tau": None,
                "precision_at_tau": None}
    best.update({"feasible": True, "alpha": alpha, "n_calib": len(pairs)})
    return best


def tau_sensitivity_v2(calib, p_star=P_STAR):
    """§7.3 Sensitivity S1 — τ nhỏ nhất có precision THỰC NGHIỆM ≥ p* trên calib.

    Chạy song song, báo kèm; KHÔNG thay primary τ* (quy tắc CP ở trên).
    """
    pairs = _calib_pairs(calib)
    scores, labels = [float(s) for s, _ in pairs], [int(c) for _, c in pairs]
    for tau in sorted(set(scores), reverse=True):
        p = precision_at_tau(scores, labels, tau)
        if p is not None and p >= p_star:
            return tau
    return None


def meets_coverage_floor(coverage, c_min=C_MIN):
    """§7.3 — coverage_e(τ*_e) < c_min = 10% → `KHÔNG ĐẠT p*`."""
    return coverage >= c_min


def verdict_coverage(ci, delta=DELTA_STAR):
    """§7.3 verdict của D_cov theo CI95 (đơn vịĐIỂM %).

      `ĐẠT`                nếu CI ⊂ [−δ*, +δ*]
      `HỎNG`               nếu CI nằm hoàn toàn ngoài [−δ*, +δ*]
      `KHÔNG ĐỦ BẰNG CHỨNG` còn lại
    """
    lo, hi = float(ci[0]), float(ci[1])
    if lo >= -delta and hi <= delta:
        return "ĐẠT"
    if hi < -delta or lo > delta:
        return "HỎNG"
    return "KHÔNG ĐỦ BẰNG CHỨNG"


def verdict_direction(ci_prec, ci_cov, eps_p=EPS_P, delta=DELTA_STAR):
    """§7.4 verdict một chiều chuyển ngưỡng (đơn vị điểm %).

    Đạt nếu CI(D_prec) ⊂ [−ε_p,+ε_p] VÀ CI(D_cov) ⊂ [−δ*,+δ*]; hỏng nếu một
    trong hai CI nằm hoàn toàn ngoài biên; còn lại không đủ. (b) đạt khi cả
    hai chiều đạt; hỏng khi có một chiều hỏng (§7.4/§7.5).
    """
    vp = verdict_coverage(ci_prec, eps_p)
    vc = verdict_coverage(ci_cov, delta)
    if vp == "ĐẠT" and vc == "ĐẠT":
        return "ĐẠT"
    if vp == "HỎNG" or vc == "HỎNG":
        return "HỎNG"
    return "KHÔNG ĐỦ BẰNG CHỨNG"


# ---------------------------------------------------------------------------
# §7.4 — Chuyển ngưỡng
# ---------------------------------------------------------------------------

def transfer(tau_k, scores_j, correct_j, scores_k, correct_k, *,
             clusters=None, n_boot=N_BOOT, seed=SEED,
             eps_p=EPS_P, delta=DELTA_STAR):
    """§7.4 — áp NGUYÊN τ*_k lên e_j trên cùng query, so với e_k tại τ*_k.

    D_prec = precision_j(τ_k) − precision_k(τ_k)
    D_cov  = coverage_j(τ_k)  − coverage_k(τ_k)
    Đơn vị: điểm %. `scores_j/correct_j/scores_k/correct_k` PHẢI cùng độ dài
    và căn chỉnh theo cùng thứ tự query. Nếu `clusters` (danh sách identity
    theo từng query) được đưa vào → paired bootstrap 2.000 lượt/seed 20261022
    với CÙNG một tập cụm cho cả hai hiệu (§7.4) + verdict chiều đó.

    Khi chưa có bootstrap placeholder, chỉ trả D_prec/D_cov + ci=None,
    verdict=None — đúng những gì "tính thuần được" ở §7.4.
    """
    if not (len(scores_j) == len(correct_j) == len(scores_k) == len(correct_k)):
        raise ValueError("bốn dãy phải cùng độ dài (cùng query)")
    pj = precision_at_tau(scores_j, correct_j, tau_k)
    pk = precision_at_tau(scores_k, correct_k, tau_k)
    cj = coverage_at_tau(scores_j, tau_k)
    ck = coverage_at_tau(scores_k, tau_k)
    if pj is None or pk is None:
        return {"d_prec_pp": None, "d_cov_pp": 100.0 * (cj - ck),
                "ci_prec_pp": None, "ci_cov_pp": None, "verdict": None,
                "note": "τ_k nằm ngoài miền điểm của ít nhất một encoder"}
    d_prec, d_cov = 100.0 * (pj - pk), 100.0 * (cj - ck)
    out = {"d_prec_pp": d_prec, "d_cov_pp": d_cov,
           "ci_prec_pp": None, "ci_cov_pp": None, "verdict": None,
           "n": len(scores_j), "tau_k": tau_k}
    if clusters is not None:
        if len(clusters) != len(scores_j):
            raise ValueError("clusters phải cùng độ dài với scores")
        groups = {}
        for i, c in enumerate(clusters):
            groups.setdefault(c, []).append(i)
        rp, rc = [], []
        for draw in _draw_groups(groups, n_boot=n_boot, seed=seed):
            ppj = precision_at_tau([scores_j[i] for i in draw],
                                   [int(correct_j[i]) for i in draw], tau_k)
            ppk = precision_at_tau([scores_k[i] for i in draw],
                                   [int(correct_k[i]) for i in draw], tau_k)
            if ppj is None or ppk is None:
                continue
            rp.append(100.0 * (ppj - ppk))
            rc.append(100.0 * (coverage_at_tau([scores_j[i] for i in draw], tau_k)
                               - coverage_at_tau([scores_k[i] for i in draw], tau_k)))
        if rp:
            out["ci_prec_pp"] = ci95(rp)
            out["ci_cov_pp"] = ci95(rc)
            out["verdict"] = verdict_direction(out["ci_prec_pp"],
                                                out["ci_cov_pp"], eps_p, delta)
            out["n_boot"] = len(rp)
            out["seed"] = seed
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_scores(path, cell, *, in_d_col="in_D(i)"):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and in_d_col in rows[0]:
        rows = [r for r in rows if r[in_d_col] == "1"]
    return {
        "path": [r["path"] for r in rows],
        "identity": [r["identity"] for r in rows],
        "score": [float(r[f"{cell}_score"]) for r in rows],
        "correct": [int(r[f"{cell}_correct"]) for r in rows],
    }


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="RQ-1 harness — prereg_v7 §7 (stdlib)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split", help="§7.1 chia 73/72 + hash (D3)")
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--audit", default=str(DEFAULT_AUDIT))
    p.add_argument("--out")

    p = sub.add_parser("metrics", help="§7.2 phân phối + AUROC + AURC")
    p.add_argument("--csv", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--out")

    p = sub.add_parser("tau", help="§7.3 τ* + CP (calib JSON: scores/correct)")
    p.add_argument("--calib", required=True)
    p.add_argument("--p-star", type=float, default=P_STAR)
    p.add_argument("--out")

    p = sub.add_parser("transfer", help="§7.4 D_prec/D_cov theo cùng query")
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--j", required=True, help="CSV encoder j")
    p.add_argument("--k", required=True, help="CSV encoder k (nguồn τ_k)")
    p.add_argument("--cell-j", required=True)
    p.add_argument("--cell-k", required=True)
    p.add_argument("--out")

    a = ap.parse_args(argv)
    if a.cmd == "split":
        out = make_split(csv_path=a.csv, audit_path=a.audit)
    elif a.cmd == "metrics":
        d = _read_scores(a.csv, a.cell)
        out = {"cell": a.cell, "n": len(d["score"]),
               "distribution": score_distribution(d["correct"], d["score"]),
               "auroc": auroc_ci(d["correct"], d["score"], d["identity"]),
               "curve": precision_coverage_curve(d["correct"], d["score"],
                                                 keys=d["path"])}
        out["curve"].pop("risk_at_k", None)  # quá dài cho stdout
    elif a.cmd == "tau":
        with Path(a.calib).open(encoding="utf-8") as fh:
            calib = json.load(fh)
        out = tau_star(calib, p_star=a.p_star)
        out["sensitivity_v2_tau"] = tau_sensitivity_v2(calib, a.p_star)
    else:  # transfer
        dj, dk = _read_scores(a.j, a.cell_j), _read_scores(a.k, a.cell_k)
        if dj["path"] != dk["path"]:
            raise ValueError("hai CSV không cùng thứ tự query — không ghép cặp được")
        out = transfer(a.tau, dj["score"], dj["correct"],
                       dk["score"], dk["correct"], clusters=dj["identity"])
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return out


if __name__ == "__main__":
    _cli()
