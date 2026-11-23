# Tables

One CSV per table of the supplementary material (23 tables), UTF-8, cell text normalised. File name `esm_sNN_tK.csv`: NN = section number, K = table order inside the section. "In paper v10": `yes` = the numbers of the table (or of its merged form, Table 2 for YC-1 and YC-3) appear in the paper; `partly` = only some of them appear, in the running text; `no` = not in the paper.

| File | Description | In paper v10 |
|---|---|---|
| `esm_s01_t1.csv` | Split sizes: images, identities and query rows per dataset and set | partly |
| `esm_s01_t2.csv` | Identity overlap between calibration, test and index per dataset | partly |
| `esm_s02_t1.csv` | CP-1: longest error-free prefix per score on ab4 calibration (Clopper-Pearson screen) | partly |
| `esm_s02_t2.csv` | CP-2: thresholds meeting the nominal bound per score and level on v9 calibration | yes |
| `esm_s02_t3.csv` | CP-3: selected threshold rows with accepted n, errors, upper bound and verdict | no |
| `esm_s03_t1.csv` | YC-1: E-AURC and AUROC of four scores on the common candidate set Yc, main regime | yes |
| `esm_s03_t2.csv` | YC-2: same four scores with global labels (D13 sensitivity regime) | no |
| `esm_s03_t3.csv` | YC-3: paired E-AURC difference of the gap against the other scores with 95% CI | yes |
| `esm_s04_t1.csv` | Image-level gap versus identity-level gap, E-AURC and paired difference | partly |
| `esm_s05_t1.csv` | Conflict gate ablation, descriptive, change in E-AURC per score | partly |
| `esm_s06_t1.csv` | McNemar exact test versus cluster permutation test for three comparisons | yes |
| `esm_s07_t1.csv` | TB-1: the eleven Tier B query transforms and their parameters | no |
| `esm_s07_t2.csv` | TB-2: per-transform flip rates, rank correlations and decision flips (SigLIP2-I) | partly |
| `esm_s07_t3.csv` | TB-3: Tier B group means for canonical and stress transforms | partly |
| `esm_s08_t1.csv` | Tier A micro top-1 accuracy of the six encoder cells | yes |
| `esm_s08_t2.csv` | ab4 test E-AURC, AUROC and paired differences with Holm-adjusted p | partly |
| `esm_s08_t3.csv` | ab4 sensitivity of the MSP and energy differences to the temperature T | partly |
| `esm_s09_t1.csv` | Algorithm 1 (Gate-and-Score) as a numbered step table | no |
| `esm_s11_t1.csv` | Deviation log D14 to D20: topic, result folder and status | no |
| `esm_s12_t1.csv` | Fig. 2 row 2, round 1: Commons photographs, similarity and decision | no |
| `esm_s12_t2.csv` | Fig. 2 row 2, rounds 2 and 3: twelve leave-one-out candidates and scores | no |
| `esm_s13_t1.csv` | Why MSP beats the gap: E-AURC and log-loss differences (D21, post hoc) | partly |
| `esm_s13_t2.csv` | Why MSP beats the gap: AUROC differences by rank of the correct label | no |

Note on `esm_s07_t3.csv` (TB-3): its last column rho is the ratio of per-transform medians averaged over the transforms of a group (post hoc; 1.035 / 0.821). It differs from the preregistered pooled rho of the paper (1.025 [0.980, 1.079] canonical, 0.878 [0.830, 0.925] stress; `results/prereg_v9_gap_20261121/p2b_results.json`).
