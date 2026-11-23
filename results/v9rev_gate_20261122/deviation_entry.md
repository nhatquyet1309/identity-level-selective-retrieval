## D17 — Ablation mô tả conflict gate (post hoc)

- Ngày: 2026-11-22. Script: `03_experiments/scripts/v9rev_gate.py` (Python311, numpy). Đặc tả: `05_submission/spec_v9_revision.md` mục 5 [C4].
- Dữ liệu: ab4 `predictions.jsonl`; v9 `pass1_capture.npz` + Y_c + T* sealed. Không đổi dữ liệu sealed.
- Cổng: conflict -> score = -inf. ab4_test n=169 (conflict 20), ab4_calib n=114 (conflict 17).
- Chỉ là ablation mô tả; không kết luận chứng nhận (certified).
