# C-case I (ab4ci) — báo cáo thực thi (2026-11-19)

> Phạm vi: phase C-case I + tìm lại ảnh + **Phase C3 (P-a, Director chấp nhận)**. **Prereg SEALED không sửa**. Không sửa `tools/measure_retrieval_quality.py` gốc (md5 `585ba4292f5677629bf3bf183366e8e6` giữ nguyên), không đụng run ab4 cũ, không đụng prereg, không đụng file lô khác. Không xóa.
> Server: `<GPU_SERVER>` (remote shell) (<GPU_SERVER>) · Python: `<PYTHON_ENV>/bin/python` · <SERVICE>: `<REDACTED_PATH>`.

## 1) D10 (deviation log)
- **Đã ghi:** `## D10` trong `03_experiments/deviation_log.md` (local, 19/11/2026).
- (i) ab4 là case I (image-only; `s_y` thuần cosine, không `use_text`/`w_img`) → đổi nhãn “ab4-S” thành **“ab4 (case I)”**; §6.2 prereg (web-blend) sai thực tế.
- (ii) Universe tier C = split ab4 (`5070-20261026-ab4/split.json`: **790 index / 284 query**, test `n_c>=3` **n=169** cùng mẫu) thay `snapshot_S2`.
- (iii) §6.3 (CLIP-B S vs I) bất khả thi bậc C → thay bằng **§6.4** SigLIP2-I vs CLIP-B-I (paired). S-vs-I: tier A §4.7 (SigLIP2 +22,0 pp p=1,3e-74 · CLIP-B +1,6 pp p=0,15).

## 2) D11 (deviation log — P-a, Director chấp nhận)
- **Đã ghi:** append `## D11` vào `03_experiments/deviation_log.md` (local, 19/11/2026).
- (i) **Universe bậc C thu hẹp 790 → 619 ảnh index (93 identity)** do **171/790 ảnh nguồn mất, không khôi phục được** (quét 29.757 file toàn <SERVICE> + `_quarantine/**` + backup/`snapshot_S2` + `<REDACTED_HOME>`, `<REDACTED_PATH>`, `/tmp`, `build_expand_aliases.py`: **120 mất hẳn** — 99 `Screenshot 2026-09-28...png` bị xóa tay, 15 hash-named; **51** va chạm basename sai vị trí/identity; `_quarantine/trim_20261111` không chứa bản đúng; ab4 chỉ lưu vector+meta, **không lưu ảnh index gốc** → bài học reproducibility/`gallery_hash`). Mất hoàn toàn 7 identity: `ababil, bmp3, fmtv, fpv_kamikaze_aze, karrar_tank, lav25, t72b3`; 38 identity giảm ảnh.
- (ii) **Test ghép cặp 169 → 160** (160 query vẫn có GT trong `Y_c` và `n_c>=3` trên universe 619).
- (iii) **Cả HAI encoder chạy lại trên 619** (re-encode từ 619 ảnh, cùng đường, công bằng — không dùng vector cũ). **Run A full 98/169 = 58,0% (repro 0 pp, `5070-261119-ab4ci-clipb`, full 790) giữ làm mốc riêng, KHÔNG so trực tiếp** với universe 619.
- (iv) §6.4 McNemar paired trên đúng **160** query.

## 3) Runner v2 (bản sao, gốc nguyên vẹn)
- `measure_retrieval_quality.py` → `measure_retrieval_quality_v2.py` (server). **Gốc không sửa** (md5 `585ba429…`).
- Vá: `--encoder {clip_b32,siglip2}`, `--split-from`, ghi `encoder_id`+`use_text:false`; CLIP-B dim 512, SigLIP2 (`siglip_engine.load_siglip`, so400m-patch14-384) dim 1152. Bổ sung kỹ thuật: `--index-from` (nạp index+meta), resolve đường dẫn, **`--drop-missing-index`** (lọc index chỉ còn ảnh nguồn resolvable đúng relpath — dùng cho P-a 619).
- `py_compile`: **OK**. `--self-test` (default `clip_b32`): **19/19 PASS, giống hệt gốc** (Exit 0).

## 4) Poll GPU / PID 28108
- PID 28108: **đã thoát** (EXITED, không kill). GPU rảnh trước các lần chạy.

## 5) Tìm lại 171 ảnh index nguồn (bước 0)
- 790 mục index (ab4 meta): **619 OK** (orig 493 + `snapshot_S2` relpath 126), **171 MISSING** đúng relpath.
- Quét toàn diện (29.757 ảnh <SERVICE>, `_quarantine/**` 171 file incl. `trim_20261111/95`, `trim_20261112/39`, `trim_20261113/1` + nhánh khác, backup/holdout/snapshot_S2 basename+relpath, ngoài <SERVICE> `<REDACTED_HOME>,<REDACTED_PATH>,/tmp`, `build_expand_aliases.py`): **0/171** khôi phục đáng tin cậy. Reltail đúng: 0. Basename: 51 nhưng **toàn va chạm tên sai identity/vị trí** (38 trong `_quarantine/`, 48 trong `snapshot_S2`; 0 hash-duy-nhất). **120 mất hẳn** (99 `Screenshot 2026-09-28...png`, 15 hash-named, 3 `confirmed_*`, 3 khác). `_quarantine/trim_20261111` không chứa bản đúng. ab4 không lưu ảnh index gốc. → P-a được chấp nhận.

## 6) Kết quả run

### 6.1 Run A (full 790, repro, mốc riêng) — `5070-261119-ab4ci-clipb`
- **HOÀN TẤT** (EXIT 0). `--encoder clip_b32 --split-from ab4 --index-from ab4/index` (full 790). **Top-1 (n_c≥3, n=169): 98/169 = 58,0%**. Repro vs ab4: **lệch 0 điểm (0,0 pp)**, đúng/sai 169/169 trùng, query 169/169 trùng (assert đúng). *Mốc riêng, KHÔNG so trực tiếp với universe 619.*

### 6.2 Run C1 (P-a, 619) — `5070-261119-ab4ci619-clipb` (CLIP-B-I)
- **HOÀN TẤT** (EXIT 0). Server: `results/5070-261119-ab4ci619-clipb/`
- Lệnh: `--encoder clip_b32 --split-from ab4/split.json --drop-missing-index --out .../5070-261119-ab4ci619-clipb --gallery .../visual_gallery --seed 20261022 --frac 0.3 --calib-frac 0.4 --t-grid 0.01,0.02,0.05,0.1,0.5,1.0 --bootstrap 2000`
- **n_index=619, n_identities=93** (kept 619/790, dropped 171), n_query 284, n_calib 114, n_test 170. `drop_missing_index=true`, `use_text=false`, `encoder_id=clip-vit-b32`.
- Top-1 (stratum `n_c>=3` theo runner, n=169): 0,491124 (49,11%). **Top-1 trên tập ghép cặp 160:** **83/160 = 51,875%**.

### 6.3 Run C2 (P-a, 619) — `5070-261119-ab4ci619-siglip2` (SigLIP2-I)
- **HOÀN TẤT** (EXIT 0). Server: `results/5070-261119-ab4ci619-siglip2/`
- Lệnh: `--encoder siglip2 --split-from ab4/split.json --drop-missing-index --out .../5070-261119-ab4ci619-siglip2 --gallery .../visual_gallery --seed 20261022 --frac 0.3 --calib-frac 0.4 --t-grid 0.01,0.02,0.05,0.1,0.5,1.0 --bootstrap 2000`
- **n_index=619, n_identities=93** (kept 619/790, dropped 171), n_query 284, n_calib 114, n_test 170. `drop_missing_index=true`, `use_text=false`, `encoder_id=siglip2-so400m-patch14-384`.
- Top-1 (stratum `n_c>=3` theo runner, n=169): 0,763314 (76,33%). **Top-1 trên tập ghép cặp 160:** **129/160 = 80,625%**.

### 6.4 Verify (619, paired)
- **Assert n_index=619, n_identities=93:** đúng cho cả C1, C2.
- **Assert tập ghép cặp 160/160 giống nhau từng qid:** đúng. Định nghĩa tập ghép cặp: query thuộc `ab4` test `n_c>=3` (169) và có `gt_identity ∈ Y_c` và `n_c>=3` trên universe 619 → **160** query (identical set cho C1 & C2).
- **Loại 9/169** (GT không còn trong `Y_c` do 7 identity mất hoàn toàn): `q00123` (ababil), `q00139,q00140,q00141` (fpv_kamikaze_aze), `q00218` (karrar_tank), `q00245` (t72b3), `q00260` (bmp3), `q00278,q00279` (lav25). Số cụm identity trong tập 160: **73** (kích thước cụm 1–5).

## 7) §6.4 McNemar paired — SigLIP2-I vs CLIP-B-I (universe 619, n=160)
- Tập: **160** query ghép cặp (cùng qid C1 & C2), D11.
- **Top-1:** CLIP-B-I (619) **83/160 = 51,875%**; SigLIP2-I (619) **129/160 = 80,625%**.
- **Hiệu (SigLIP2-I − CLIP-B-I): +46/160 = +28,75 điểm phần trăm**.
- Đối chiếu ngữ cảnh (KHÔNG so trực tiếp): CLIP-B full 790 (Run A, mốc riêng) 98/169 = 58,0% repro 0 pp.

**Bảng 2×2 (CLIP-B-I đúng/sai × SigLIP2-I đúng/sai, n=160):**

|  | SigLIP2-I đúng | SigLIP2-I sai | Tổng |
|---|---:|---:|---:|
| **CLIP-B-I đúng** | a = 75 | b = 8 | 83 |
| **CLIP-B-I sai** | c = 54 | d = 23 | 77 |
| **Tổng** | 129 | 31 | 160 |

- Cặp bất hòa: n_d = b + c = **62** (b=8, c=54).
- **McNemar exact (binomial 2 phía): p = 1,7093 × 10⁻⁹** (2·Σ_{i=0}^{min(b,c)} C(n_d,i)/2^{n_d} = 2·Σ_{i=0}^{8} C(62,i)/2^{62}).
- **Paired bootstrap theo cụm identity** (73 cụm, resample cụm, B=2000, seed 20261022): hiệu = **+0,2875** (+28,75 pp), mean = **0,286754**, **CI 95% = [0,205298, 0,369427]** (+20,53 pp đến +36,94 pp). CI không cắt 0; p_exact ≪ 0,05 ⇒ SigLIP2-I vượt trội CLIP-B-I có ý nghĩa trên universe 619 (P-a, D11).

## 8) File + SHA
- Local `03_experiments/results/C_caseI_20261119/`:
  - `report.md`, `metrics_ab4ci_clipb.json` (Run A full), `predictions_ab4ci_clipb.jsonl` (Run A), `predictions_ab4_ref.jsonl`, `SHA256SUMS.txt`
  - `619/metrics_clipb619.json`, `619/metrics_siglip619.json`, `619/predictions_clipb619.jsonl`, `619/predictions_siglip619.jsonl`, `619/mcnemar_6_4.json`
- **SHA256** (xem đầy đủ trong `SHA256SUMS.txt`, bao gồm `report.md`):
  - `bcd14d15f0088f2996a3c2e8653bfe9fa05d6427426bb0557b0680dcfe98ddb3  predictions_ab4_ref.jsonl`
  - `dd1514f09423570f31a11b2784edfe2b29f7d0e03b8794a28d6709d0df87ffbf  predictions_ab4ci_clipb.jsonl`
  - `f769d17786866dde32f2329beebc707fb5e10aa243eabd3b2f8320bc0f3398c7  metrics_ab4ci_clipb.json`
  - `4d5d150e24b9415a1b4ac78ec1ae55c1ee4f94ab87fc0c139be417d36ac620c2  619/metrics_clipb619.json`
  - `d111bfd6fa3e2209ba175bd0e41b9d8778ae9fdd649a843772a3cd99855896b7  619/metrics_siglip619.json`
  - `b742b2da468f3481668fb167f78e3e3280d08a8e0107052c65354cafd41056f3  619/predictions_clipb619.jsonl`
  - `f05254382536d61c7462c2baae925e15b7fc1dad2dfd449ac99babd63b2952f8  619/predictions_siglip619.jsonl`
  - `1c82fd4b2eabc668f4e2b54e87d654d719bfc4fced0c15418ca2e21b9db92135  619/mcnemar_6_4.json`
- Server (không sửa gốc/ab4): run A `.../5070-261119-ab4ci-clipb/`, C1 `.../5070-261119-ab4ci619-clipb/`, C2 `.../5070-261119-ab4ci619-siglip2/`, v2 `.../tools/measure_retrieval_quality_v2.py` (md5 gốc `585ba429…` nguyên vẹn). `/tmp/c_casei_missing_search.json`.
- Cập nhật: `03_experiments/deviation_log.md` (D10,D11), `03_experiments/results/plan_C_caseI_20261119.md`.

## 9) Vấn đề / ghi chú
- 171 ảnh index nguồn mất vĩnh viễn (ab4 không lưu ảnh gốc) → D11; `_quarantine/trim_20261111` không giúp. Kết quả §6.4 trên universe 619 (160 query) hợp lệ nội bộ P-a, **không so trực tiếp** với 98/169 full-universe.