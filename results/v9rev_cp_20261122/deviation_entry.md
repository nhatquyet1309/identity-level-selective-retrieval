## D18 — Cận trên Clopper–Pearson một phía theo ngưỡng (ab4 calibration, v9 calibration) — 22/11/2026

**Phạm vi.** Phân tích post hoc mô tả (C5 revision v9). Chỉ đọc dữ liệu sealed (ab4 predictions, v9 pass1 + T* sealed); không chọn lại T. Script `scripts/v9rev_cp.py`; kết quả `results/v9rev_cp_20261122/results.json`.

**Phương pháp.** Mỗi giá trị score phân biệt là một hàng ngưỡng (chấp nhận khi score >= tau, ties cùng hàng); cận trên CP một phía 95% của tỉ lệ lỗi (k/n; = 1 nếu k = n). certified nếu cận <= alpha; "observed risk above target" nếu k/n > alpha; còn lại "insufficient data". n cần cho 0 lỗi: 59 (5%), 299 (1%).

**Sửa câu sai v8.** Câu v8 "Calibration size exceeds the bound, so the binding constraint is the error rate" đúng nửa: ab4 có 114 mẫu calibration, prefix không lỗi tối đa 21 (< 59) nên mức 5% bất khả thi do tỉ lệ lỗi; mức 1% cần >= 299 mẫu chấp nhận không lỗi > 114 nên ràng buộc là thiếu dữ liệu.

**Nhãn.** Post hoc, mô tả; không phải chứng nhận.
