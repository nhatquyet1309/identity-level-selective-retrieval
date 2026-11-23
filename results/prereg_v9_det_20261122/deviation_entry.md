## D22 - Chạy lại bootstrap P1 v9 với seed phụ tất định (crc32)

**Lý do.** `prereg_v9_stats.compute_p1` dùng seed phụ `hash((enc, other)) % 2**32`; `hash()` của chuỗi/tuple bị ngẫu nhiên hóa theo tiến trình (PYTHONHASHSEED) nên CI và p của P1 không tái lập từng bit (D13 mục 5: CI lệch tối đa 0,0019).

**Phương pháp.** Script mới `scripts/prereg_v9_stats_det.py` (không sửa bản sealed), cùng phân tích (B = 2000, cụm theo identity test, E-AURC ghép, percentile [2,5; 97,5], p hai phía +1-smoothing, Holm m = 3), chỉ đổi seed phụ: `default_rng([20261022, zlib.crc32(f"{eid}|{other}".encode())])` (cùng tag với D13, nên trùng D13 `all_832/global` từng bit).

**Phạm vi.** Chỉ CI/p/Holm của P1 đổi. Điểm ước lượng (delta, E-AURC, metrics_test), T\* không đổi. Nhãn 6/6 không đổi.

Lệch CI tối đa |d| = 0.001908; lệch p thô tối đa = 0.019990.

**Bảng lệch cũ -> mới.**

| Encoder | So sánh | CI lo cũ -> mới | CI hi cũ -> mới | p cũ -> mới | Holm cũ -> mới | Nhãn cũ -> mới |
|---|---|---|---|---|---|---|
| SigLIP2-I (siglip2_I) | gap_minus_sim | -0.050575 -> -0.051112 | -0.008614 -> -0.007012 | 0.001000 -> 0.008996 | 0.002999 -> 0.017991 | better -> better |
| SigLIP2-I (siglip2_I) | gap_minus_MSP | -0.000665 -> -0.000601 | +0.006585 -> +0.006336 | 0.117941 -> 0.118941 | 0.117941 -> 0.118941 | insufficient -> insufficient |
| SigLIP2-I (siglip2_I) | gap_minus_energy | -0.058336 -> -0.058587 | -0.009967 -> -0.010269 | 0.002999 -> 0.003998 | 0.005997 -> 0.011994 | better -> better |
| CLIP-B-I (clip_I) | gap_minus_sim | -0.135171 -> -0.134851 | -0.042403 -> -0.044311 | 0.001000 -> 0.001000 | 0.002999 -> 0.002999 | better -> better |
| CLIP-B-I (clip_I) | gap_minus_MSP | -0.016443 -> -0.015909 | +0.026107 -> +0.026274 | 0.746627 -> 0.766617 | 0.746627 -> 0.766617 | insufficient -> insufficient |
| CLIP-B-I (clip_I) | gap_minus_energy | -0.169259 -> -0.170756 | -0.069389 -> -0.070693 | 0.001000 -> 0.001000 | 0.002999 -> 0.002999 | better -> better |

**Tham chiếu.** D13 (`results/D13_Yc_sensitivity_20261121/results.json`, `all_832/global`); kết quả: `results/prereg_v9_det_20261122/results.json`. Bản gốc `results/prereg_v9_gap_20261121/p1_results.json` giữ nguyên.

**Câu P1 mới cho bài (Δk = E_k − E_gap; CI; Holm p):**

> +0.027 [+0.007, +0.051] (0.018); −0.003 [−0.006, +0.001] (0.119); +0.033 [+0.010, +0.059] (0.012); +0.090 [+0.044, +0.135] (0.003); −0.005 [−0.026, +0.016] (0.767); +0.118 [+0.071, +0.171] (0.003)

**Vị trí cần cập nhật (lượt sửa chữ sau, không sửa ở đây).**
- `02_manuscript/paper_ares_v9.docx`: câu "Against the gap, Δ with Holm p is ..." (6 ô CI/Holm p); nguồn `05_submission/build/content_v9.py` dòng 657.
- `05_submission/ESM_v9.docx`: đoạn "Bootstrap seed ... hash() ..." viết lại thành D22 chạy lại tất định.
- `03_experiments/deviation_log.md`: thêm mục D22 từ tệp này (không sửa D13).
