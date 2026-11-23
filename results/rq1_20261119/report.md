# RQ-1 + McNemar §4.7 — prereg_v7 (SEALED e5080142…74d1d)

- **Ngày:** 2026-11-19
- **Universe:** A(ii) trên D(ii), 6 ô, fp32 (D8). `loo_Aii.csv` (tier A, đóng).
- **D(ii):** 1663 hàng, 141 identity có ảnh trong D(ii); universe 145 identity (D3).
- **Split §7.1:** seed=20261022, 73 calib / 72 test, `split_hash=77a6428941ed701e2570b2ede8babe28506a941e38800db2f4444f37c864a18f` (khớp prefix `77a64289` = **True**), `order_hash=6c29a39e8dca04279977ccc70b7ceb3764d5fbe4c0e6ec8725a94b7a9db12dbc`.
- **Tham số:** p*=0.95, c_min=10%, ε_p=2.0pp, δ*=5.0pp, bootstrap 2000 lượt/seed 20261022, cluster=identity (145 cụm, D3). Stdlib-only.
- **RNG:** `random.Random` stdlib (qua `cluster_draws`/`tierB_part2.py`) — family khai báo.

## §4.7 McNemar S vs I (D(ii), case S vs I cùng encoder)

clipl14/dinov2 S≡I — không có cặp (D2: bit-exact).

### siglip2

Bảng 2×2 (hàng = S, cột = I):

| | I đúng | I sai |
|---|---:|---:|
| **S đúng** | 937 | 46 |
| **S sai** | 412 | 268 |

- n=1663, b (S đúng, I sai)=46, c (S sai, I đúng)=412, n_discordant=458
- acc(S)=0.591100 (59.110%), acc(I)=0.811185 (81.118%)
- Hiệu top-1(I)−top-1(S) = 0.220084 (22.008 pp)
- **McNemar exact (binomial 2-sided): p = 1.3304793e-74**
- Paired bootstrap identity (145 cụm, n_boot=2000, seed=20261022), CI95% hiệu (I−S): [0.158837, 0.284371] = [15.884, 28.437] pp

### clip

Bảng 2×2 (hàng = S, cột = I):

| | I đúng | I sai |
|---|---:|---:|
| **S đúng** | 588 | 150 |
| **S sai** | 177 | 748 |

- n=1663, b (S đúng, I sai)=150, c (S sai, I đúng)=177, n_discordant=327
- acc(S)=0.443776 (44.378%), acc(I)=0.460012 (46.001%)
- Hiệu top-1(I)−top-1(S) = 0.016236 (1.624 pp)
- **McNemar exact (binomial 2-sided): p = 0.1503790389**
- Paired bootstrap identity (145 cụm, n_boot=2000, seed=20261022), CI95% hiệu (I−S): [-0.025738, 0.056718] = [-2.574, 5.672] pp

## §7.3 τ* / sàn c_min (tiêu chí a — theo ô, chiều chính calib→test)

| Cell | τ* (CP, calib 73) | CP@τ* | n@τ* (calib) | feasible | S1 τ (emp≥p*, calib) | precision test | coverage test | ≥c_min 10% |
|---|---:|---:|---:|:---:|---:|---:|---:|:---:|
| siglip2_S | None | n/a | n/a | no | 0.522806 | n/a | 0.000000 [0.0000, 0.0000] | no |
| siglip2_I | 0.960026 | 0.0452395 | 137 | yes | 0.996552 | 0.965517 [0.9167, 1.0000] | 0.139423 [0.1002, 0.1821] | yes |
| clip_S | None | n/a | n/a | no | 0.650558 | n/a | 0.000000 [0.0000, 0.0000] | no |
| clip_I | None | n/a | n/a | no | 0.982433 | n/a | 0.000000 [0.0000, 0.0000] | no |
| dinov2 | None | n/a | n/a | no | 0.997736 | n/a | 0.000000 [0.0000, 0.0000] | no |
| clipl14 | None | n/a | n/a | no | 0.986209 | n/a | 0.000000 [0.0000, 0.0000] | no |

## §7.5 Kết luận từng cặp (7 cặp, chiều chính calib→test) + cờ đảo chiều

Thứ tự ưu tiên nhãn: `KHÔNG ĐẠT p*` > `KHÔNG SO ĐƯỢC` > `SO ĐƯỢC` > `KHÔNG ĐỦ BẰNG CHỨNG`.

| Case | Cặp | (a) | (b) | Nhãn cuối | ⚑ đảo chiều | D_cov (a, test) CI95% | Ghi chú |
|---|---|---|---|---|:---:|---|---|
| I | siglip2_I vs clip_I | KHÔNG ĐẠT p* | KHÔNG ĐỦ BẰNG CHỨNG | **KHÔNG ĐẠT p*** |  | infeasible hoặc < c_min tại: clip_I | infeasible hoặc < c_min tại: clip_I |
| I | siglip2_I vs dinov2 | KHÔNG ĐẠT p* | KHÔNG ĐỦ BẰNG CHỨNG | **KHÔNG ĐẠT p*** |  | infeasible hoặc < c_min tại: dinov2 | infeasible hoặc < c_min tại: dinov2 |
| I | siglip2_I vs clipl14 | KHÔNG ĐẠT p* | KHÔNG ĐỦ BẰNG CHỨNG | **KHÔNG ĐẠT p*** |  | infeasible hoặc < c_min tại: clipl14 | infeasible hoặc < c_min tại: clipl14 |
| I | clip_I vs dinov2 | KHÔNG ĐẠT p* | KHÔNG ĐỦ BẰNG CHỨNG | **KHÔNG ĐẠT p*** |  | infeasible hoặc < c_min tại: clip_I, dinov2 | infeasible hoặc < c_min tại: clip_I, dinov2 |
| I | clip_I vs clipl14 | KHÔNG ĐẠT p* | KHÔNG ĐỦ BẰNG CHỨNG | **KHÔNG ĐẠT p*** |  | infeasible hoặc < c_min tại: clip_I, clipl14 | infeasible hoặc < c_min tại: clip_I, clipl14 |
| I | dinov2 vs clipl14 | KHÔNG ĐẠT p* | KHÔNG ĐỦ BẰNG CHỨNG | **KHÔNG ĐẠT p*** |  | infeasible hoặc < c_min tại: dinov2, clipl14 | infeasible hoặc < c_min tại: dinov2, clipl14 |
| S | siglip2_S vs clip_S | KHÔNG ĐẠT p* | KHÔNG ĐỦ BẰNG CHỨNG | **KHÔNG ĐẠT p*** |  | infeasible hoặc < c_min tại: siglip2_S, clip_S | infeasible hoặc < c_min tại: siglip2_S, clip_S |

### Chi tiết chuyển ngưỡng §7.4 (chiều chính, từng cặp)

#### siglip2_I vs clip_I (case I) — verdict_b=KHÔNG ĐỦ BẰNG CHỨNG
- `apply_tau_siglip2_I_onto_clip_I_vs_siglip2_I`: D_prec=-13.694581280788176 pp CI=[-31.580, -1.846] pp, D_cov=-9.735576923076923 pp CI=[-13.582, -5.861] pp, verdict=HỎNG 
- `apply_tau_clip_I_onto_siglip2_I_vs_clip_I`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible

#### siglip2_I vs dinov2 (case I) — verdict_b=KHÔNG ĐỦ BẰNG CHỨNG
- `apply_tau_siglip2_I_onto_dinov2_vs_siglip2_I`: D_prec=3.4482758620689613 pp CI=[0.000, 8.411] pp, D_cov=-13.461538461538462 pp CI=[-17.680, -9.467] pp, verdict=HỎNG 
- `apply_tau_dinov2_onto_siglip2_I_vs_dinov2`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible

#### siglip2_I vs clipl14 (case I) — verdict_b=KHÔNG ĐỦ BẰNG CHỨNG
- `apply_tau_siglip2_I_onto_clipl14_vs_siglip2_I`: D_prec=-4.364224137931039 pp CI=[-10.385, 0.578] pp, D_cov=-6.25 pp CI=[-10.435, -2.401] pp, verdict=KHÔNG ĐỦ BẰNG CHỨNG 
- `apply_tau_clipl14_onto_siglip2_I_vs_clipl14`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible

#### clip_I vs dinov2 (case I) — verdict_b=KHÔNG ĐỦ BẰNG CHỨNG
- `apply_tau_clip_I_onto_dinov2_vs_clip_I`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible
- `apply_tau_dinov2_onto_clip_I_vs_dinov2`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible

#### clip_I vs clipl14 (case I) — verdict_b=KHÔNG ĐỦ BẰNG CHỨNG
- `apply_tau_clip_I_onto_clipl14_vs_clip_I`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible
- `apply_tau_clipl14_onto_clip_I_vs_clipl14`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible

#### dinov2 vs clipl14 (case I) — verdict_b=KHÔNG ĐỦ BẰNG CHỨNG
- `apply_tau_dinov2_onto_clipl14_vs_dinov2`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible
- `apply_tau_clipl14_onto_dinov2_vs_clipl14`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible

#### siglip2_S vs clip_S (case S) — verdict_b=KHÔNG ĐỦ BẰNG CHỨNG
- `apply_tau_siglip2_S_onto_clip_S_vs_siglip2_S`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible
- `apply_tau_clip_S_onto_siglip2_S_vs_clip_S`: D_prec=None pp CI=n/a, D_cov=None pp CI=n/a, verdict=None nguồn tau_k infeasible

## §7.2 Chỉ tiêu theo ô (D(ii), toàn bộ)

### 7.2.2 AUROC (cluster bootstrap identity 145 cụm)

| Cell | AUROC | CI95% | n_boot |
|---|---:|---:|---:|
| siglip2_S | 0.746263 | [0.703350, 0.787179] | 2000 |
| siglip2_I | 0.737926 | [0.699625, 0.772752] | 2000 |
| clip_S | 0.623179 | [0.580712, 0.662186] | 2000 |
| clip_I | 0.658904 | [0.624339, 0.691010] | 2000 |
| dinov2 | 0.653912 | [0.620105, 0.687717] | 2000 |
| clipl14 | 0.719373 | [0.682791, 0.753898] | 2000 |

### 7.2.3 AURC (thường, §7.2.3)

| Cell | AURC | n |
|---|---:|---:|
| siglip2_S | 0.229053 | 1663 |
| siglip2_I | 0.086126 | 1663 |
| clip_S | 0.450599 | 1663 |
| clip_I | 0.400525 | 1663 |
| dinov2 | 0.324306 | 1663 |
| clipl14 | 0.177552 | 1663 |

### 7.2.1 Phân phối điểm đúng/sai (tóm tắt)

#### siglip2_S (KS=0.3998085093650888)

| Nhóm | n | median | Q1 | Q3 | P10 | P30 | P50 | P70 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 983 | 0.472221 | 0.453378 | 0.485653 | 0.432310 | 0.457502 | 0.472221 | 0.483127 | 0.498953 |
| wrong | 680 | 0.446807 | 0.427658 | 0.462499 | 0.409177 | 0.433602 | 0.446807 | 0.458717 | 0.477990 |

#### siglip2_I (KS=0.36603900978785897)

| Nhóm | n | median | Q1 | Q3 | P10 | P30 | P50 | P70 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 1349 | 0.929117 | 0.891702 | 0.953698 | 0.848520 | 0.901897 | 0.929117 | 0.949225 | 0.967855 |
| wrong | 314 | 0.881991 | 0.842419 | 0.916843 | 0.798110 | 0.850817 | 0.881991 | 0.910456 | 0.937919 |

#### clip_S (KS=0.20332381161649454)

| Nhóm | n | median | Q1 | Q3 | P10 | P30 | P50 | P70 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 738 | 0.606953 | 0.587011 | 0.619353 | 0.564925 | 0.591646 | 0.606953 | 0.616820 | 0.632712 |
| wrong | 925 | 0.593916 | 0.574119 | 0.610775 | 0.546779 | 0.578289 | 0.593916 | 0.607296 | 0.624298 |

#### clip_I (KS=0.23150938177795244)

| Nhóm | n | median | Q1 | Q3 | P10 | P30 | P50 | P70 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 765 | 0.916694 | 0.889412 | 0.940917 | 0.852835 | 0.894686 | 0.916694 | 0.936670 | 0.958025 |
| wrong | 898 | 0.893978 | 0.860174 | 0.920730 | 0.821461 | 0.869078 | 0.893978 | 0.915656 | 0.937888 |

#### dinov2 (KS=0.23818168032870546)

| Nhóm | n | median | Q1 | Q3 | P10 | P30 | P50 | P70 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 906 | 0.881524 | 0.810211 | 0.925456 | 0.722290 | 0.829307 | 0.881524 | 0.919205 | 0.946431 |
| wrong | 757 | 0.827591 | 0.730083 | 0.893223 | 0.641999 | 0.754471 | 0.827591 | 0.880471 | 0.924303 |

#### clipl14 (KS=0.34943962430681613)

| Nhóm | n | median | Q1 | Q3 | P10 | P30 | P50 | P70 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct | 1133 | 0.929647 | 0.894995 | 0.948740 | 0.854958 | 0.903347 | 0.929647 | 0.944581 | 0.962263 |
| wrong | 530 | 0.891259 | 0.846265 | 0.921669 | 0.798446 | 0.854390 | 0.891259 | 0.916145 | 0.937535 |

## Cửa mở / ghi chú

- Đảo chiều calib↔test: các cặp có ⚑ không dùng làm claim (§7.5).
- CI bootstrap không gồm bất định của τ* (chọn trên calib); đảo chiều là kiểm tra bù (§7.3).
- RQ-1 hiệu lực trên LOO gallery sản xuất (seen), không suy rộng sang unseen/bậc C (§7.6).
- clipl14/dinov2 S≡I (D2): không có cặp S–I cho McNemar.
- §7.4 verdict chiều: dùng `transfer()` (CI D_prec ⊂ [−ε_p,+ε_p] và D_cov ⊂ [−δ*,+δ*]). (b) đạt khi cả hai chiều đạt.
- Cluster bootstrap resample theo universe 145 identity (D3); 4 identity 1-ảnh không đóng góp hàng D(ii) nhưng nằm trong khung resample.
