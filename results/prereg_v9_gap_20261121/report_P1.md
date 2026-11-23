# prereg_v9 — Report P1 (SEALED)

- prereg: `prereg_v9.md SEALED`
- split_hash: `77a6428941ed701e2570b2ede8babe28506a941e38800db2f4444f37c864a18f`
- seed: 20261022 | RNG: `numpy.default_rng(PCG64)` | n_boot: 2000
- T grid: [0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
- D(ii): 1663 queries | calib ids: 73 | test ids: 72
- Sc4 definition: ab4/spec_v2 (measure_retrieval_quality.py L486-496): inter-identity gap within Y_c. s_all[v]=max score per identity; s_y={y:s_all[y] for y in Y_c}; sv=sorted desc; Sc4=m_id=sv[0]-sv[1] (n_c<2 -> abstain/NaN). Y_c from COARSE_MAP same as ab4 L450-465.
- Sc4 fix note (D12): D12: v9 §1 mô tả nhầm 'ảnh cao nhì cùng identity' (intra-image). Clause quyết định 'đúng định nghĩa ab4/spec_v2 (Sc1-Sc4)' được giữ; Sc4 đã sửa về inter-identity gap trong Y_c. Kết quả intra-image cũ giữ làm *intra_image_SUPERSEDED*.

## Sanity n_c==2: MSP(T*) vs Sc4 (ab4 L349-356, sigmoid(m/T) monotone)

| encoder | n_nc2 | rank_equal_full | pairwise_concordant/total | concordance | T_MSP | n_abstain_sc4 |
|---|---:|---:|---:|---:|---:|---:|
| siglip2_I | 10 | True | 45/45 | 1.000000 | 0.010000 | 0 |
| clip_I | 10 | False | 37/45 | 0.822222 | 0.010000 | 0 |

## Bảng metrics test (4 score × 2 encoder)

| encoder | score | AURC | E-AURC | AUROC | cov@risk5% | risk@cov | n_acc | n_err | cp_upper |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| siglip2_I | sim | 0.074955 | 0.057995 | 0.766783 | 0.444712 | 0.048649 | 370 | 18 | 0.071290 |
| siglip2_I | MSP | 0.044931 | 0.027971 | 0.873726 | 0.682692 | 0.049296 | 568 | 28 | 0.066965 |
| siglip2_I | energy | 0.080106 | 0.063146 | 0.752884 | 0.423077 | 0.048295 | 352 | 17 | 0.071564 |
| siglip2_I | gap | 0.047502 | 0.030542 | 0.864164 | 0.680288 | 0.049470 | 566 | 28 | 0.067199 |
| clip_I | sim | 0.405701 | 0.225627 | 0.652327 | 0.000000 | 0.000000 | 0 | 0 | 1.000000 |
| clip_I | MSP | 0.311480 | 0.131405 | 0.778904 | 0.026442 | 0.045455 | 22 | 1 | 0.198122 |
| clip_I | energy | 0.433722 | 0.253648 | 0.623830 | 0.002404 | 0.000000 | 2 | 0 | 0.776393 |
| clip_I | gap | 0.316058 | 0.135984 | 0.771097 | 0.050481 | 0.047619 | 42 | 2 | 0.142414 |

## Paired ΔE-AURC (gap − other), cluster bootstrap identity, Holm 3

| encoder | comparison | ΔE-AURC | 95% CI | p_boot | Holm p | label |
|---|---|---:|---:|---:|---:|---|
| siglip2_I | gap_minus_sim | -0.027453 | [-0.050575, -0.008614] | 0.001000 | 0.002999 | better |
| siglip2_I | gap_minus_MSP | 0.002571 | [-0.000665, 0.006585] | 0.117941 | 0.117941 | insufficient |
| siglip2_I | gap_minus_energy | -0.032604 | [-0.058336, -0.009967] | 0.002999 | 0.005997 | better |
| clip_I | gap_minus_sim | -0.089643 | [-0.135171, -0.042403] | 0.001000 | 0.002999 | better |
| clip_I | gap_minus_MSP | 0.004579 | [-0.016443, 0.026107] | 0.746627 | 0.746627 | insufficient |
| clip_I | gap_minus_energy | -0.117664 | [-0.169259, -0.069389] | 0.001000 | 0.002999 | better |

## T* calib

- siglip2_I: T*_MSP=0.01 (calib E-AURC=0.036917), T*_energy=0.01 (calib E-AURC=0.081108), test n=832, test acc=0.822115
- clip_I: T*_MSP=0.01 (calib E-AURC=0.119546), T*_energy=0.01 (calib E-AURC=0.239035), test n=832, test acc=0.463942

## Ghi chú

- ΔE-AURC = E-AURC(gap) − E-AURC(other); âm = gap tốt hơn (E-AURC thấp hơn).
- Label theo prereg v9: `better` nếu CI95% loại trừ 0 về phía tốt hơn (ci95_hi < 0); ngược lại `insufficient`. Không dùng δ mới.
- Bootstrap cluster theo identity test, 2000 lần, seed 20261022, sub-RNG per comparison.
- D12: Sc4 đã sửa từ intra-image (ảnh cao nhì cùng identity) sang inter-identity gap trong Y_c theo ab4 L486-496. Kết quả intra-image cũ được giữ làm `*intra_image_SUPERSEDED*`.
