## D16 — Image-level gap m_img vs identity gap m_id (post hoc)

- Ngày: 2026-11-22. Script: `03_experiments/scripts/v9rev_imggap.py` (Python311, numpy). Đặc tả: `05_submission/spec_v9_revision.md` mục 5 [C3].
- Dữ liệu: ab4 `predictions.jsonl` (cột m_id, m_img, top2_same_identity), v9 `pass1_capture.npz` (s1_img, s2_img_same_id, id_scores). Không đổi dữ liệu sealed.
- Δ E-AURC = E(m_img) - E(m_id), bootstrap cụm identity (B=2000). ab4_test n=169, ab4_calib n=114; top2_same_identity trên 284 query = 111.
- Phân tích mô tả post hoc; không điều chỉnh thiết kế hay kết luận của bài.
