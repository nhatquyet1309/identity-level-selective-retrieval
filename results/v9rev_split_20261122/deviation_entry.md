## D14 — Bảng split và thuật ngữ (ab4, Tier C 619, v9 RQ-1) — 22/11/2026

**Phạm vi.** Phân tích post hoc mô tả (C1 revision v9). Chỉ đọc dữ liệu sealed; không sửa file sealed. Script `scripts/v9rev_split.py`; kết quả `results/v9rev_split_20261122/results.json`.

**Phát hiện.**
- ab4: 790 ảnh index/100 identity; 284 ảnh query/95 identity (95 ⊂ 100); calibration 114 ảnh/66 identity, test 170 ảnh/80 identity, chung 51 identity (tách theo ảnh). Là "unseen query images of known identities", KHÔNG identity-disjoint giữa calibration/test.
- Tier C (619): index 619 ảnh/93 identity (171 ảnh mất; 7 identity mất hẳn); test ghép cặp 169 -> 160 (9 query loại), 73 cụm identity; vẫn là ảnh query held-out của identity đã có trong index.
- v9 RQ-1: gallery 1667 ảnh/145 identity; calibration 73 identity/831 dòng, test 72 identity/832 dòng; 0 identity chung (identity-disjoint chỉ đúng cho calibration vs test).

**Hiệu chỉnh.** Nhãn "Tier C (unseen identities)" của v8 bị loại; dùng "held-out query images" (xem spec_v9_revision.md [T1]-[T6]). Không số nào mới: mọi số tái tính từ split.json/predictions/pass1_meta.
