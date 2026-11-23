# Kết quả Khối B — chất lượng truy xuất (identity-level)

- **run_id**: `5070-20261026-ab4` · **seed**: 20261022
- **Kích thước gallery**: 1074 ảnh · 100 identity
- **Sau tách (sợ bộ, seed=20261022, frac=0.3)**:
  - **Index** (gallery tham chiếu): **790 ảnh**
  - **Query** (sợ bộ holdout): **284 ảnh** → CALIB 114 · TEST 170
- **Ứng viên `Y_c`**: cùng coarse class, nguồn = **oracle**
- **Lưới T**: [0.01, 0.02, 0.05, 0.1, 0.5, 1.0] · **α**: [0.01, 0.05] · **bootstrap**: 2000 replicate, cluster = identity
- **Quy tắc phá hoà**: argmax, stable order by identity name

> Mọi số đọc thẳng từ `metrics.json` (sinh từ `predictions.jsonl`) — không ước lượng.

## 1. Phân bố số ứng viên `n_c` (toàn bộ sợ bộ)

| n_c | số truy vấn | ghi chú |
|---|---|---|
| 2 | 1 | tầng không thể so sánh score (Sc2≡Sc4) |
| 10 | 45 | tầng so sánh được |
| 15 | 36 | tầng so sánh được |
| 18 | 51 | tầng so sánh được |
| 19 | 57 | tầng so sánh được |
| 36 | 94 | tầng so sánh được |

- Số truy vấn **n_c < 2 (từ chối)**: **0**
- Trong đó `m_img` vẫn xác định được (đối chứng Sc4′): **0**

## 2. Mô tả tầng thư nghiệm (tập TEST)

| Tầng | n | top-1 | top-3 | `r_same` |
|---|---|---|---|---|
| `n_c>=3` | 169 | 58.0 % | 79.9 % | 40.8 % |
| `n_c==2` | 1 | 100.0 % | 100.0 % | 0.0 % |

## 3.1. So sánh 4 score — tầng `n_c>=3` (n = 169)

| Score | AURC ↓ | E-AURC ↓ (CI 95 %) | AUROC ↑ (CI 95 %) |
|---|---|---|---|
| Sc1 — do tuong doi (absolute similarity, s1) | 0.23345 | 0.12808 [0.08225, 0.19045] | 0.7387 [0.6461, 0.8173] |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.01) | 0.22981 | 0.12445 [0.08114, 0.17651] | 0.7519 [0.6766, 0.8267] |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.02) | 0.23426 | 0.12889 [0.08410, 0.17994] | 0.7389 [0.6568, 0.8204] |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.05) | 0.25444 | 0.14908 [0.09812, 0.20499] | 0.7002 [0.6141, 0.7893] |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.1) | 0.28841 | 0.18304 [0.12663, 0.24671] | 0.6189 [0.5250, 0.7121] |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.5) | 0.35972 | 0.25436 [0.18569, 0.33093] | 0.5221 [0.4261, 0.6160] |
| Sc2 — xac suat softmax toi da (MSP_T) (T=1.0) | 0.36987 | 0.26450 [0.19265, 0.34351] | 0.5079 [0.4131, 0.6001] |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.01) | 0.24428 | 0.13891 [0.08939, 0.20319] | 0.7235 [0.6283, 0.8065] |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.02) | 0.26816 | 0.16280 [0.10535, 0.23504] | 0.6907 [0.5887, 0.7766] |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.05) | 0.30833 | 0.20297 [0.13617, 0.29163] | 0.6214 [0.5114, 0.7101] |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.1) | 0.31596 | 0.21060 [0.14521, 0.30234] | 0.5890 [0.4806, 0.6824] |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.5) | 0.33259 | 0.22723 [0.15913, 0.31396] | 0.5473 [0.4516, 0.6435] |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=1.0) | 0.32952 | 0.22416 [0.15724, 0.30919] | 0.5552 [0.4603, 0.6513] |
| Sc4 — khoang cach danh tinh (m = s1 - s2) | 0.22989 | 0.12453 [0.08189, 0.17554] | 0.7498 [0.6761, 0.8205] |

## 3.2. So sánh 4 score — tầng `n_c==2` (n = 1)

| Score | AURC ↓ | E-AURC ↓ (CI 95 %) | AUROC ↑ (CI 95 %) |
|---|---|---|---|
| Sc1 — do tuong doi (absolute similarity, s1) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.01) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.02) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.05) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.1) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.5) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc2 — xac suat softmax toi da (MSP_T) (T=1.0) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.01) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.02) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.05) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.1) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.5) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=1.0) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |
| Sc4 — khoang cach danh tinh (m = s1 - s2) | 0.00000 | 0.00000 [0.00000, 0.00000] | n/a n/a |

## 4. Hiệu chuẩn ngưỡng `τ_α` (chọn trên CALIB) → kiểm tra trên TEST

### Tầng `n_c>=3`

| Score | α | τ_α | n chấp nhận (calib) | CP trên (calib) | coverage (test) | risk thực (test) | confident-error rate |
|---|---|---|---|---|---|---|---|
| Sc1 — do tuong doi (absolute similarity, s1) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc1 — do tuong doi (absolute similarity, s1) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.01) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.01) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.02) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.02) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.05) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.05) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.1) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.1) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.5) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.5) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=1.0) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=1.0) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.01) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.01) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.02) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.02) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.05) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.05) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.1) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.1) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.5) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.5) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=1.0) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=1.0) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc4 — khoang cach danh tinh (m = s1 - s2) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc4 — khoang cach danh tinh (m = s1 - s2) | 5 % | **n/a — bất khả thi** | | | | | |

### Tầng `n_c==2`

| Score | α | τ_α | n chấp nhận (calib) | CP trên (calib) | coverage (test) | risk thực (test) | confident-error rate |
|---|---|---|---|---|---|---|---|
| Sc1 — do tuong doi (absolute similarity, s1) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc1 — do tuong doi (absolute similarity, s1) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.01) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.01) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.02) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.02) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.05) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.05) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.1) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.1) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.5) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=0.5) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=1.0) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc2 — xac suat softmax toi da (MSP_T) (T=1.0) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.01) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.01) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.02) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.02) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.05) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.05) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.1) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.1) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.5) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=0.5) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=1.0) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) (T=1.0) | 5 % | **n/a — bất khả thi** | | | | | |
| Sc4 — khoang cach danh tinh (m = s1 - s2) | 1 % | **n/a — bất khả thi** | | | | | |
| Sc4 — khoang cach danh tinh (m = s1 - s2) | 5 % | **n/a — bất khả thi** | | | | | |

## 5. So sánh ghép cặp E-AURC (Sc4 làm chuẩn) — tầng `n_c>=3`, n_test = 169, n_calib = 114

**Quy tắc chọn T\*** (khai báo trước): argmin E-AURC tren split CALIB (khai bao truoc, chon mot lan cho ca so sanh chinh)
→ T\* = {'Sc1': None, 'Sc2': 0.02, 'Sc3': 0.01}

| So sánh | ΔE-AURC (đối với Sc4) | CI 95 % (bootstrap cụm) | p (bootstrap 2 phía) | Kết luận |
|---|---|---|---|---|
| Sc1 — do tuong doi (absolute similarity, s1) | 0.00356 | [-0.06171, 0.07227] | 0.8910 | khong du bang chung |
| Sc2 — xac suat softmax toi da (MSP_T) | 0.00437 | [-0.01615, 0.02472] | 0.6710 | khong du bang chung |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) | 0.01439 | [-0.05480, 0.09023] | 0.6870 | khong du bang chung |

**Hiệu chỉnh Holm** (3 so sánh chính):

| So sánh | p thô | p sau Holm |
|---|---|---|
| Sc2 | 0.6710 | 1.0000 |
| Sc3 | 0.6870 | 1.0000 |
| Sc1 | 0.8910 | 1.0000 |

### 5.1. Phân tích độ nhạy theo lưới T (không phải so sánh riêng)

| Score | T | ΔE-AURC vs Sc4 | CI 95 % | Kết luận |
|---|---|---|---|---|
| Sc2 — xac suat softmax toi da (MSP_T) | 0.01 | -0.00008 | [-0.01098, 0.01090] | khong du bang chung |
| Sc2 — xac suat softmax toi da (MSP_T) | 0.02 | 0.00437 | [-0.01615, 0.02472] | khong du bang chung |
| Sc2 — xac suat softmax toi da (MSP_T) | 0.05 | 0.02455 | [-0.01234, 0.06309] | khong du bang chung |
| Sc2 — xac suat softmax toi da (MSP_T) | 0.1 | 0.05852 | [0.00461, 0.11521] | Sc4 tot hon |
| Sc2 — xac suat softmax toi da (MSP_T) | 0.5 | 0.12983 | [0.05953, 0.20798] | Sc4 tot hon |
| Sc2 — xac suat softmax toi da (MSP_T) | 1.0 | 0.13998 | [0.06724, 0.22538] | Sc4 tot hon |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) | 0.01 | 0.01439 | [-0.05480, 0.09023] | khong du bang chung |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) | 0.02 | 0.03827 | [-0.03956, 0.12299] | khong du bang chung |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) | 0.05 | 0.07844 | [-0.01201, 0.18217] | khong du bang chung |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) | 0.1 | 0.08607 | [-0.00087, 0.19050] | khong du bang chung |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) | 0.5 | 0.10270 | [0.01744, 0.19498] | Sc4 tot hon |
| Sc3 — nang luong (Energy_T, DAO CHIEU: nho hon = tin) | 1.0 | 0.09963 | [0.01572, 0.19193] | Sc4 tot hon |

## 6. Kiểm tra tính đúng đắn của code (mục 8.7)

| Kiểm tra | Kết quả |
|---|---|
| m_img = min(s1-a, m_id) violations | 0 |
| n_nc_lt2_counted | 0 |
| r_same_overall | 0.3908 |

## 7. Ràng buộc bắt buộc khi trích dẫn

1. **Ứng viên `Y_c` dùng coarse source = `oracle`** (lấy từ thư mục category của gallery). Gallery crop **không** đi qua detector YOLO nên không có nhãn lớp thật. Kết quả này **không** mô tả được đường chạy 2 giai đoạn đầy đủ (detector → coarse class → gallery). Phải ghi rõ khi trích dẫn.
2. **Chỉ có một miền truy vấn** (`gallery`, sợ bộ holdout). Chuyển giao ngưỡng hai chiều (A→B) **không đánh giá được** — chưa có miền thứ hai. Không được viết kết luận chuyển giao.
3. **Không có nhãn góc nhìn** (viewpoint). Các khung "viewpoint shift" của bài báo **chưa được kiểm chứng bằng số**; nếu giữ khung đó trong tiêu đề thì phải công khai giới hạn này.
4. **Gallery còn chưa hoàn thiện** (nhiều identity chỉ có 1 ảnh) → phân bố `n_c` và số cụm nhỏ → khoảng tin cậy rộng. Số ở đây là **bản đo trên gallery hiện tại**, không phải kết quả cuối.
5. **Không được so AURC giữa các score khác nhau khi chúng đổi `y₁`** — ở đây cả 4 score đều tính trên cùng một `s_y` và cùng một `y₁` nên so sánh là hợp lệ.
