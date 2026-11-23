# Why MSP beats the gap

Post hoc, descriptive analysis (deviation D21) of the difference between the identity gap and MSP.

This section is post hoc and descriptive (deviation D21). It reads the folder v10_diag (results.json), which reuses the captured scores of the preregistered run: 1663 rows (831 calibration, 832 test) and two encoders, SigLIP2-I and CLIP-B-I, with the sealed temperature T* = 0.01, not re-selected. Nothing in the main paper changes.

The hypothesis about δ below was formed after the 832 test queries had been seen (D13, D15 and the first diagnostic round), so the test split has only exploratory value here and none as a confirmation.

With s_1 ≥ s_2 ≥ … the sorted candidate scores of one query, m_id = s_1 − s_2 is the identity gap, and the temperature-scaled MSP splits exactly as

```
logit(MSP_T) = m_id/T − δ,    δ = log(1 + Σ_{j≥3} exp((s_j − s_2)/T))
```

Thus δ ≥ 0 measures the competitor mass beyond the second candidate, and the gap ignores it. δ = 0 when Y_c has two candidates (n_c = 2), and then the gap order equals the MSP order. In the data only Ship has n_c = 2 (4 calibration and 6 test queries); the other strata have n_c in {19, 20, 22, 30, 52}.

Sign conventions. ΔE-AURC is gap minus MSP, so a positive value means the gap is worse. ΔAUROC is MSP minus gap, so a positive value means MSP ranks the correct answers higher. Δlog-loss is the cross-validated log-loss without δ minus the log-loss with δ, so a positive value means δ helps; it exists only on the calibration split (M0 minus M3: δ added to m_id; M2 minus M1: δ added when m_id and n_c are already present). Intervals are 95% cluster bootstrap by identity.

Point estimates equal those of main-text section 5.3; the 95% bounds differ by up to 0.0003 because this section uses an independent bootstrap draw (D21 versus D15).


| Encoder | Split (n) | ΔE-AURC, gap minus MSP | Δlog-loss, M0 minus M3 | Δlog-loss, M2 minus M1 |
|---|---|---|---|---|
| SigLIP2-I | calibration (831) | +0.0007 [−0.0005; 0.0021] | +0.0011 [−0.0061; 0.0079] | +0.0009 [−0.0048; 0.0061] |
| SigLIP2-I | test (832) | +0.0018 [0.0004; 0.0035] | n/a | n/a |
| CLIP-B-I | calibration (831) | +0.0076 [0.0022; 0.0132] | +0.0113 [0.0002; 0.0230] | +0.0126 [0.0009; 0.0247] |
| CLIP-B-I | test (832) | +0.0086 [0.0022; 0.0147] | n/a | n/a |


| Encoder | Split (n) | ΔAUROC, MSP minus gap, rank 1 vs 2 (n) | ΔAUROC, MSP minus gap, rank 1 vs ≥3 (n) | MSP right in opposite pairs |
|---|---|---|---|---|
| SigLIP2-I | calibration (831) | −0.0008 [−0.0076; 0.0062] (n = 83) | +0.0095 [0.0016; 0.0171] (n = 79) | 57.0% (1,881 of 3,301) |
| SigLIP2-I | test (832) | −0.0035 [−0.0125; 0.0050] (n = 63) | +0.0169 [0.0105; 0.0242] (n = 80) | 63.9% (1,794 of 2,808) |
| CLIP-B-I | calibration (831) | −0.0130 [−0.0301; 0.0064] (n = 94) | +0.0237 [0.0119; 0.0352] (n = 326) | 59.8% (8,156 of 13,634) |
| CLIP-B-I | test (832) | −0.0142 [−0.0337; 0.0029] (n = 118) | +0.0334 [0.0184; 0.0494] (n = 293) | 60.9% (9,540 of 15,663) |

Opposite pairs are pairs of queries (correct against wrong) that the gap and MSP order in opposite ways; the last column is the share of them that MSP orders correctly.

ΔAUROC compares the correct queries (the correct label at rank 1) with the queries whose correct label is at rank 2, and with those at rank 3 or lower (≥3), for each encoder and split. Split (n) is the number of queries in the split. In the ΔAUROC columns, (n) is the number of queries whose correct label is at rank 2 (first column) or at rank 3 or lower (second column).

When the correct queries (rank 1) are compared with the queries whose correct label is at rank 2, the ΔAUROC interval contains zero in all four cells, so the gap and MSP are not separated.

When the correct queries (rank 1) are compared with the queries whose correct label is at rank 3 or lower (≥3), MSP has a higher AUROC than the gap in all four cells, with every interval above zero.

On calibration SigLIP2-I the ΔE-AURC interval and both Δlog-loss intervals contain zero, so the signal there is unresolved.

The variant score m_id − λTδ was not evaluated: λ = 0 is the gap and λ = 1 is the MSP order, so it only interpolates between the two, and λ > 1 only adds a further δ penalty, which is no longer the gap; there is also no data independent of the 832 test queries on which to evaluate it.
