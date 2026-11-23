# prereg_v9 — Report P2 (SEALED, D12-part2)

- prereg: `prereg_v9.md SEALED`
- seed: 20261022 | RNG: `numpy.default_rng(PCG64)` | n_boot: 2000
- encoder: siglip2_I | queries: 1667 | transforms: 11 | total: 18337
- chính (Y_c-restricted): delta = value(T(x)) - value(x), same component-masked A(ii) setting; m_id = Y_c-restricted inter-identity gap sv[0]-sv[1] (ab4 L486-496, D12 fix); s1 = global top1 identity score (absolute).
- D12-part2: D12-part2: recomputed from pass2b_capture.npz (id_scores_all 1667x11x145). Global-identity-gap P2 kept as sensitivity in p2_results_global_sensitivity.json / report section.

## Biến thể CHÍNH — Y_c-restricted inter-identity gap (ab4 L486-496)

| group | median|Δm_id| | median|Δs1| | P95 |Δm_id| | P95 |Δs1| | ratio median | 95% CI ratio | state | n_rows | n_abstain_nc_lt2 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| canonical_6T | 0.002572 | 0.002510 | 0.011967 | 0.012161 | 1.024877 | [0.980409, 1.078720] | KHÔNG THẤY KHÁC | 10002 | 0 |
| stress_5T | 0.005906 | 0.006730 | 0.030937 | 0.065080 | 0.877539 | [0.830031, 0.924609] | ỔN ĐỊNH HƠN | 8335 | 0 |

### Per-transform (Y_c-restricted)

| T | median|Δm_id| | median|Δs1| | ratio | 95% CI | state | n_abstain_nc_lt2 |
|---|---:|---:|---:|---:|---|---:|
| G1 | 0.004609 | 0.004393 | 1.049271 | [0.958885, 1.149717] | KHÔNG THẤY KHÁC | 0 |
| L1 | 0.002300 | 0.002268 | 1.014163 | [0.917511, 1.084938] | KHÔNG THẤY KHÁC | 0 |
| L2 | 0.003375 | 0.003364 | 1.003438 | [0.927847, 1.103529] | KHÔNG THẤY KHÁC | 0 |
| L3 | 0.001601 | 0.001442 | 1.110445 | [1.022462, 1.219780] | KÉM ỔN ĐỊNH HƠN | 0 |
| L4 | 0.002308 | 0.002379 | 0.970086 | [0.894873, 1.038021] | KHÔNG THẤY KHÁC | 0 |
| L5 | 0.002450 | 0.002302 | 1.064204 | [0.971435, 1.142680] | KHÔNG THẤY KHÁC | 0 |
| G2 | 0.014505 | 0.040166 | 0.361117 | [0.316938, 0.412446] | ỔN ĐỊNH HƠN | 0 |
| X1 | 0.004222 | 0.004593 | 0.919159 | [0.858000, 1.000831] | KHÔNG THẤY KHÁC | 0 |
| X2 | 0.006004 | 0.006567 | 0.914320 | [0.842946, 0.993725] | ỔN ĐỊNH HƠN | 0 |
| X3 | 0.005176 | 0.005215 | 0.992502 | [0.911592, 1.085002] | KHÔNG THẤY KHÁC | 0 |
| X4 | 0.004489 | 0.004885 | 0.918867 | [0.835012, 1.018499] | KHÔNG THẤY KHÁC | 0 |

## Biến thể SENSITIVITY — global identity top-2 (pass2 cũ, pre-D12)

| group | median|Δm_id| | median|Δs1| | P95 |Δm_id| | P95 |Δs1| | ratio median | 95% CI ratio | state | n_rows |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| canonical_6T | 0.002580 | 0.002510 | 0.011893 | 0.012161 | 1.027798 | [0.980603, 1.081986] | KHÔNG THẤY KHÁC | 10002 |
| stress_5T | 0.005901 | 0.006730 | 0.030960 | 0.065080 | 0.876689 | [0.829723, 0.925142] | ỔN ĐỊNH HƠN | 8335 |

### Per-transform (global sensitivity)

| T | median|Δm_id| | median|Δs1| | ratio | 95% CI | state |
|---|---:|---:|---:|---:|---|
| G1 | 0.004613 | 0.004393 | 1.050221 | [0.957693, 1.152697] | KHÔNG THẤY KHÁC |
| L1 | 0.002346 | 0.002268 | 1.034108 | [0.936571, 1.094027] | KHÔNG THẤY KHÁC |
| L2 | 0.003336 | 0.003364 | 0.991743 | [0.916096, 1.084677] | KHÔNG THẤY KHÁC |
| L3 | 0.001559 | 0.001442 | 1.081015 | [0.994430, 1.188235] | KHÔNG THẤY KHÁC |
| L4 | 0.002322 | 0.002379 | 0.976149 | [0.899788, 1.046787] | KHÔNG THẤY KHÁC |
| L5 | 0.002426 | 0.002302 | 1.053719 | [0.962694, 1.137306] | KHÔNG THẤY KHÁC |
| G2 | 0.015042 | 0.040166 | 0.374486 | [0.331010, 0.425780] | ỔN ĐỊNH HƠN |
| X1 | 0.004287 | 0.004593 | 0.933277 | [0.868286, 1.013046] | KHÔNG THẤY KHÁC |
| X2 | 0.006005 | 0.006567 | 0.914330 | [0.843683, 0.988423] | ỔN ĐỊNH HƠN |
| X3 | 0.005117 | 0.005215 | 0.981324 | [0.908713, 1.073324] | KHÔNG THẤY KHÁC |
| X4 | 0.004560 | 0.004885 | 0.933312 | [0.847989, 1.020324] | KHÔNG THẤY KHÁC |

## So sánh trạng thái (Y_c-restricted vs global)

| group | Y_c ratio [CI] state | global ratio [CI] state | đổi trạng thái? |
|---|---|---|---|
| canonical_6T | 1.024877 [0.980409,1.078720] KHÔNG THẤY KHÁC | 1.027798 [0.980603,1.081986] KHÔNG THẤY KHÁC | KHÔNG |
| stress_5T | 0.877539 [0.830031,0.924609] ỔN ĐỊNH HƠN | 0.876689 [0.829723,0.925142] ỔN ĐỊNH HƠN | KHÔNG |

## Ghi chú

- Ratio = median(|Δm_id|) / median(|Δs1|). CI cluster bootstrap theo identity, 2000 lần, seed 20261022.
- Trạng thái prereg v9: CI < 1 → `ỔN ĐỊNH HƠN`; CI chứa 1 → `KHÔNG THẤY KHÁC`; CI > 1 → `KÉM ỔN ĐỊNH HƠN`.
- Biến thể chính dùng pass2b_capture.npz (id_scores_all 1667x11x145) để tính m_id Y_c-restricted. Biến thể global dùng scalar pass2 cũ (top-2 identity global).
