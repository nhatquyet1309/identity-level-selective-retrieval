# D13 — Độ nhạy: MSP/energy tính trên tập ứng viên Y_c (so với toàn cục 145 identity của v9)

Ngày: 21/11/2026. Đặc tả: `03_experiments/spec_D13.md`. Hiện thực: `scripts/d13_yc_sensitivity.py`.

## 1. Mục đích và công bố khác biệt (D13)

- Trong v9 (sealed), **MSP và energy được tính trên toàn cục 145 identity** (softmax/logsumexp theo cả hàng điểm), còn gap là khoảng cách top-2 trong Y_c. Table 1 của paper mô tả MSP trên Y_c -> **lệch định nghĩa**, công bố tại đây.
- Kết quả v9 được **giữ nguyên, không đổi nhãn đã seal**. D13 là phân tích độ nhạy trên cùng `pass1_capture.npz`, dùng **T\* đã seal** (MSP và energy từng encoder), **không chọn lại T**. T\* = 0,01 ở mép dưới của lưới; không kết luận gì về T khác 0,01.
- `sim` (toàn cục), `gap` (Y_c) và `correct` (dự đoán toàn cục, argmax 145 identity) **giữ nguyên**; chỉ MSP và energy đổi.
- Map identity->category lấy từ `pass2b_meta.json` (đủ 145 identity); `pass1_meta.json` thiếu 4 identity (`btr80`, `ec130`, `switchblade_600`, `z19`) nên không dùng.

## 2. Định nghĩa

Toàn cục (v9): MSP_T = max_y softmax(s/T) trên 145 identity; -E_T = T·logsumexp(s/T) trên 145 identity.

Y_c (D13): cùng công thức nhưng chỉ trên Y_c = {y: coarse(y) = coarse(query)}. Kích thước Y_c: {'Airplane': 30, 'Tank': 22, 'Vehicle': 52, 'Helicopter': 20, 'Drone': 19, 'Ship': 2}.

## 3. Kết quả theo tầng

Tầng (a) `all_832`: toàn bộ 832 query test. Tầng (b) `nc_ge3`: query có n_c >= 3 (826 query). **n_c = 2 chỉ là nhóm Ship = 6 query test (0,7%)** nên (a) và (b) gần như trùng; ý kiến 'tầng n_c=2 trộn' chỉ liên quan 6 query.

### siglip2_I (T*_MSP=0.01, T*_energy=0.01)

**Tầng all_832** — n=832, số lỗi=148

| Định nghĩa | Score | AURC | E-AURC | AUROC | cov@5% |
|---|---|---|---|---|---|
| global | sim | 0.074955 | 0.057995 | 0.7668 | 0.4447 |
| global | MSP | 0.044931 | 0.027971 | 0.8737 | 0.6827 |
| global | energy | 0.080106 | 0.063146 | 0.7529 | 0.4231 |
| global | gap | 0.047502 | 0.030542 | 0.8642 | 0.6803 |
| Yc | sim | 0.074955 | 0.057995 | 0.7668 | 0.4447 |
| Yc | MSP | 0.045460 | 0.028500 | 0.8730 | 0.6899 |
| Yc | energy | 0.079406 | 0.062446 | 0.7564 | 0.4231 |
| Yc | gap | 0.047502 | 0.030542 | 0.8642 | 0.6803 |

| Định nghĩa | So sánh | ΔE-AURC | CI95 | p | Holm p | Nhãn |
|---|---|---|---|---|---|---|
| global | gap−sim | -0.027453 | [-0.0511, -0.0070] | 0.0090 | 0.0180 | better |
| global | gap−MSP | +0.002571 | [-0.0006, +0.0063] | 0.1189 | 0.1189 | insufficient |
| global | gap−energy | -0.032604 | [-0.0586, -0.0103] | 0.0040 | 0.0120 | better |
| Yc | gap−sim | -0.027453 | [-0.0511, -0.0070] | 0.0090 | 0.0140 | better |
| Yc | gap−MSP | +0.002042 | [+0.0006, +0.0037] | 0.0070 | 0.0140 | insufficient |
| Yc | gap−energy | -0.031904 | [-0.0581, -0.0097] | 0.0040 | 0.0120 | better |

Đối chiếu nhãn toàn cục -> Y_c: gap−sim: better -> better (không đổi); gap−MSP: insufficient -> insufficient (không đổi); gap−energy: better -> better (không đổi)

**Tầng nc_ge3** — n=826, số lỗi=147

| Định nghĩa | Score | AURC | E-AURC | AUROC | cov@5% |
|---|---|---|---|---|---|
| global | sim | 0.074559 | 0.057582 | 0.7688 | 0.4479 |
| global | MSP | 0.045267 | 0.028290 | 0.8724 | 0.6816 |
| global | energy | 0.079694 | 0.062717 | 0.7550 | 0.4262 |
| global | gap | 0.047850 | 0.030873 | 0.8626 | 0.6792 |
| Yc | sim | 0.074559 | 0.057582 | 0.7688 | 0.4479 |
| Yc | MSP | 0.045781 | 0.028804 | 0.8716 | 0.6889 |
| Yc | energy | 0.078998 | 0.062021 | 0.7585 | 0.4262 |
| Yc | gap | 0.047850 | 0.030873 | 0.8626 | 0.6792 |

| Định nghĩa | So sánh | ΔE-AURC | CI95 | p | Holm p | Nhãn |
|---|---|---|---|---|---|---|
| global | gap−sim | -0.026709 | [-0.0487, -0.0069] | 0.0080 | 0.0160 | better |
| global | gap−MSP | +0.002583 | [-0.0006, +0.0061] | 0.1109 | 0.1109 | insufficient |
| global | gap−energy | -0.031844 | [-0.0584, -0.0092] | 0.0020 | 0.0060 | better |
| Yc | gap−sim | -0.026709 | [-0.0487, -0.0069] | 0.0080 | 0.0120 | better |
| Yc | gap−MSP | +0.002069 | [+0.0006, +0.0038] | 0.0060 | 0.0120 | insufficient |
| Yc | gap−energy | -0.031148 | [-0.0572, -0.0087] | 0.0020 | 0.0060 | better |

Đối chiếu nhãn toàn cục -> Y_c: gap−sim: better -> better (không đổi); gap−MSP: insufficient -> insufficient (không đổi); gap−energy: better -> better (không đổi)

### clip_I (T*_MSP=0.01, T*_energy=0.01)

**Tầng all_832** — n=832, số lỗi=446

| Định nghĩa | Score | AURC | E-AURC | AUROC | cov@5% |
|---|---|---|---|---|---|
| global | sim | 0.405701 | 0.225627 | 0.6523 | 0.0000 |
| global | MSP | 0.311480 | 0.131405 | 0.7789 | 0.0264 |
| global | energy | 0.433722 | 0.253648 | 0.6238 | 0.0024 |
| global | gap | 0.316058 | 0.135984 | 0.7711 | 0.0505 |
| Yc | sim | 0.405701 | 0.225627 | 0.6523 | 0.0000 |
| Yc | MSP | 0.305118 | 0.125043 | 0.7950 | 0.0493 |
| Yc | energy | 0.406349 | 0.226275 | 0.6527 | 0.0096 |
| Yc | gap | 0.316058 | 0.135984 | 0.7711 | 0.0505 |

| Định nghĩa | So sánh | ΔE-AURC | CI95 | p | Holm p | Nhãn |
|---|---|---|---|---|---|---|
| global | gap−sim | -0.089643 | [-0.1349, -0.0443] | 0.0010 | 0.0030 | better |
| global | gap−MSP | +0.004579 | [-0.0159, +0.0263] | 0.7666 | 0.7666 | insufficient |
| global | gap−energy | -0.117664 | [-0.1708, -0.0707] | 0.0010 | 0.0030 | better |
| Yc | gap−sim | -0.089643 | [-0.1349, -0.0443] | 0.0010 | 0.0030 | better |
| Yc | gap−MSP | +0.010941 | [+0.0042, +0.0177] | 0.0030 | 0.0030 | insufficient |
| Yc | gap−energy | -0.090291 | [-0.1401, -0.0446] | 0.0010 | 0.0030 | better |

Đối chiếu nhãn toàn cục -> Y_c: gap−sim: better -> better (không đổi); gap−MSP: insufficient -> insufficient (không đổi); gap−energy: better -> better (không đổi)

**Tầng nc_ge3** — n=826, số lỗi=442

| Định nghĩa | Score | AURC | E-AURC | AUROC | cov@5% |
|---|---|---|---|---|---|
| global | sim | 0.404920 | 0.225572 | 0.6519 | 0.0000 |
| global | MSP | 0.308187 | 0.128838 | 0.7820 | 0.0266 |
| global | energy | 0.433134 | 0.253786 | 0.6231 | 0.0024 |
| global | gap | 0.304307 | 0.124959 | 0.7768 | 0.1223 |
| Yc | sim | 0.404920 | 0.225572 | 0.6519 | 0.0000 |
| Yc | MSP | 0.292634 | 0.113286 | 0.8005 | 0.1053 |
| Yc | energy | 0.405653 | 0.226305 | 0.6521 | 0.0097 |
| Yc | gap | 0.304307 | 0.124959 | 0.7768 | 0.1223 |

| Định nghĩa | So sánh | ΔE-AURC | CI95 | p | Holm p | Nhãn |
|---|---|---|---|---|---|---|
| global | gap−sim | -0.100613 | [-0.1433, -0.0583] | 0.0010 | 0.0030 | better |
| global | gap−MSP | -0.003879 | [-0.0202, +0.0101] | 0.5967 | 0.5967 | insufficient |
| global | gap−energy | -0.128827 | [-0.1786, -0.0834] | 0.0010 | 0.0030 | better |
| Yc | gap−sim | -0.100613 | [-0.1433, -0.0583] | 0.0010 | 0.0030 | better |
| Yc | gap−MSP | +0.011673 | [+0.0048, +0.0184] | 0.0020 | 0.0030 | insufficient |
| Yc | gap−energy | -0.101346 | [-0.1492, -0.0596] | 0.0010 | 0.0030 | better |

Đối chiếu nhãn toàn cục -> Y_c: gap−sim: better -> better (không đổi); gap−MSP: insufficient -> insufficient (không đổi); gap−energy: better -> better (không đổi)

## 4. Kiểm tra nhất quán với v9 (tầng all_832, định nghĩa toàn cục)

- Sai lệch điểm lớn nhất (AURC/E-AURC/AUROC/Δ) so với `p1_results.json`: 0.000e+00 (ngưỡng 1e-6).
- Sai lệch CI lớn nhất: 0.0019 (ngưỡng 0,004; do seed phụ).
- Nhãn khớp v9: True; T* chọn lại trên calib (toàn cục) khớp T* sealed: True.

## 5. Ghi chú tái lập

v9 dùng `hash((enc, other)) % 2**32` làm seed phụ; `hash` chuỗi bị ngẫu nhiên hóa theo tiến trình nên CI/p của v9 **không tái lập bit-by-bit** (sai số Monte-Carlo ~0,001). D13 dùng seed phụ xác định `default_rng([20261022, zlib.crc32(f"{eid}|{other}")])`, dùng chung mọi tầng và mọi định nghĩa (so sánh ghép). Holm: 3 so sánh cho mỗi (encoder × tầng × định nghĩa); nhãn `better` nếu ci95_hi < 0, ngược lại `insufficient` (luật v9, không δ mới).

