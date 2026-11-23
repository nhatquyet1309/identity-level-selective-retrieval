## D19 — Kiểm định hoán vị đổi dấu theo cụm identity (bổ sung McNemar) — 22/11/2026

**Phạm vi.** Phân tích post hoc mô tả (C6 revision v9). Chỉ đọc dữ liệu sealed; McNemar đã báo giữ nguyên. Script `scripts/v9rev_perm.py`; kết quả `results/v9rev_perm_20261122/results.json`.

**Phương pháp.** d_i = đúng_A − đúng_B; T = tổng d / n; lật dấu cả cụm identity với xác suất 1/2 (paired, hai phía); Monte Carlo B = 100000, seed 20261022, p = (1 + #{|T*| >= |T_obs|}) / (B + 1), p tối thiểu phân giải được 1/(B+1). Ba so sánh: Tier C SigLIP2-I vs CLIP-B-I (160 query, 73 cụm), Tier A SigLIP2 I vs S và CLIP I vs S (1663 dòng, 141 cụm).

**Diễn giải.** McNemar coi các ca độc lập; ca cùng identity tương quan nên p McNemar chống bảo thủ. p hoán vị theo cụm là kiểm tra độ nhạy post hoc, không thay phép kiểm đã đăng ký.
