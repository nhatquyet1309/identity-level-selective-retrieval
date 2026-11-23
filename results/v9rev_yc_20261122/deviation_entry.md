## D15 — Độ nhạy Y_c: 4 score trên cùng candidate set, hai regime nhãn (post hoc)

- Ngày: 2026-11-22. Script: `03_experiments/scripts/v9rev_yc.py` (Python311, numpy). Đặc tả: `05_submission/spec_v9_revision.md` mục 5 [C2].
- Dữ liệu: `pass1_capture.npz`, `pass1_meta.json`, `pass2b_meta.json` (Y_c đủ 145 identity), `p1_results.json` (T* sealed, không chọn lại). Không đổi dữ liệu sealed.
- s1, MSP, energy, gap đều tính trong Y_c. Regime chính `yc_argmax` (nhãn = argmax trong Y_c); `global_argmax` (nhãn sealed) khớp D13 trong 1e-9.
- Δ E-AURC = E(gap) - E(other), CI bootstrap cụm theo identity (B=2000, seed [20261022, crc32(tag)]); CI chứa 0 -> nhãn `insufficient`.
- Số ca nhãn global khác nhãn Y_c: SigLIP2-I 23, CLIP-B-I 233 (trong 1663); số ca đúng/sai khác: 9 và 67.
- Phân tích mô tả post hoc; không kết luận gap hơn MSP.
