# Manifest

Every file of the repository with a short description. `results/` is listed by folder (13 folders). `MANIFEST.md` and `SHA256SUMS` describe the repository itself; hashes are in `SHA256SUMS`.

## Files

| Path | Description |
|---|---|
| `.gitignore` | ignore rules (caches, credentials, weights, arrays, images) |
| `CITATION.cff` | citation metadata |
| `LICENSE` | MIT licence (code) |
| `LICENSE-data` | CC BY 4.0 notice for results/, tables/, docs/ |
| `README.md` | summary, citation, structure, main results, reproduction, limitations |
| `docs/algorithm1.md` | Algorithm 1 (Gate-and-Score) pseudocode |
| `docs/deviation_log.md` | deviation log D1 to D22 (scrubbed) |
| `docs/figure_credits.md` | credits and source pages of the Fig. 2 photographs (no images included) |
| `docs/moved_from_paper.md` | details moved out of the paper: D11, lost images, A-50, Fig. 2 row 2 selection, full confidence intervals |
| `docs/prereg_v7.md` | sealed protocol v7 (scrubbed) |
| `docs/prereg_v7.md.sha256` | sha256 of the original protocol v7 |
| `docs/prereg_v8_addendum.md` | protocol v8 addendum |
| `docs/prereg_v8_addendum.md.sha256` | sha256 of the original v8 addendum |
| `docs/prereg_v9.md` | sealed protocol v9 |
| `docs/prereg_v9.md.sha256` | sha256 of the original protocol v9 |
| `docs/qualfig_row2_selection.md` | selection of Fig. 2 row 2 (three rounds, post hoc) |
| `docs/repro_facts_v9j.md` | reproducibility facts: snapshot, hashes, index type, resize modes |
| `docs/reproducibility.md` | reproducibility record (checkpoints, preprocessing, index, seeds, manifests) |
| `docs/why_msp_beats_gap.md` | D21: why MSP beats the gap (post hoc, descriptive) |
| `paper/identity_level_selective_retrieval_icisn2027.pdf` | PDF of the paper (camera-ready layout of the submitted version) |
| `requirements-capture.txt` | pip requirements of the GPU capture step (torch, transformers, Pillow, faiss; clip not recorded) |
| `requirements.txt` | pip requirements of the statistics step (numpy) |
| `scripts/d13_yc_sensitivity.py` | D13 candidate-set sensitivity analysis |
| `scripts/prereg_v9_capture.py` | capture of the v9 scores (GPU stage) |
| `scripts/prereg_v9_stats.py` | v9 statistics from the capture arrays |
| `scripts/prereg_v9_stats_det.py` | D22 deterministic bootstrap re-run |
| `scripts/rq1_harness.py` | RQ-1 split and harness |
| `scripts/sweep_thresholds.py` | threshold sweep on the S2 cache |
| `scripts/tierA_smoke.py` | helper imported by the capture script |
| `scripts/v9rev_cp.py` | Clopper-Pearson sweep (D18) |
| `scripts/v9rev_gate.py` | conflict gate ablation (D17) |
| `scripts/v9rev_imggap.py` | image-level gap (D16) |
| `scripts/v9rev_perm.py` | cluster permutation tests (D19) |
| `scripts/v9rev_split.py` | split and terminology table |
| `scripts/v9rev_tierB.py` | Tier B transforms analysis (D20) |
| `scripts/v9rev_yc.py` | four scores over Yc (D13, D15) |
| `tables/INDEX.md` | index of the CSV tables with descriptions |
| `tables/esm_s01_t1.csv` | Split sizes: images, identities and query rows per dataset and set |
| `tables/esm_s01_t2.csv` | Identity overlap between calibration, test and index per dataset |
| `tables/esm_s02_t1.csv` | CP-1: longest error-free prefix per score on ab4 calibration (Clopper-Pearson screen) |
| `tables/esm_s02_t2.csv` | CP-2: thresholds meeting the nominal bound per score and level on v9 calibration |
| `tables/esm_s02_t3.csv` | CP-3: selected threshold rows with accepted n, errors, upper bound and verdict |
| `tables/esm_s03_t1.csv` | YC-1: E-AURC and AUROC of four scores on the common candidate set Yc, main regime |
| `tables/esm_s03_t2.csv` | YC-2: same four scores with global labels (D13 sensitivity regime) |
| `tables/esm_s03_t3.csv` | YC-3: paired E-AURC difference of the gap against the other scores with 95% CI |
| `tables/esm_s04_t1.csv` | Image-level gap versus identity-level gap, E-AURC and paired difference |
| `tables/esm_s05_t1.csv` | Conflict gate ablation, descriptive, change in E-AURC per score |
| `tables/esm_s06_t1.csv` | McNemar exact test versus cluster permutation test for three comparisons |
| `tables/esm_s07_t1.csv` | TB-1: the eleven Tier B query transforms and their parameters |
| `tables/esm_s07_t2.csv` | TB-2: per-transform flip rates, rank correlations and decision flips (SigLIP2-I) |
| `tables/esm_s07_t3.csv` | TB-3: Tier B group means for canonical and stress transforms |
| `tables/esm_s08_t1.csv` | Tier A micro top-1 accuracy of the six encoder cells |
| `tables/esm_s08_t2.csv` | ab4 test E-AURC, AUROC and paired differences with Holm-adjusted p |
| `tables/esm_s08_t3.csv` | ab4 sensitivity of the MSP and energy differences to the temperature T |
| `tables/esm_s09_t1.csv` | Algorithm 1 (Gate-and-Score) as a numbered step table |
| `tables/esm_s11_t1.csv` | Deviation log D14 to D20: topic, result folder and status |
| `tables/esm_s12_t1.csv` | Fig. 2 row 2, round 1: Commons photographs, similarity and decision |
| `tables/esm_s12_t2.csv` | Fig. 2 row 2, rounds 2 and 3: twelve leave-one-out candidates and scores |
| `tables/esm_s13_t1.csv` | Why MSP beats the gap: E-AURC and log-loss differences (D21, post hoc) |
| `tables/esm_s13_t2.csv` | Why MSP beats the gap: AUROC differences by rank of the correct label |

## Result folders

| Folder | Content |
|---|---|
| `results/prereg_v9_gap_20261121` | sealed v9 capture metadata, p1/p2 results and reports |
| `results/prereg_v9_det_20261122` | D22: deterministic re-run of the bootstrap |
| `results/D13_Yc_sensitivity_20261121` | D13: common candidate set Yc sensitivity, risk-coverage csv |
| `results/v9rev_split_20261122` | split and terminology table (ab4, Tier C, v9) |
| `results/v9rev_yc_20261122` | four scores over the same candidate set Yc (post hoc) |
| `results/v9rev_imggap_20261122` | image-level gap versus identity-level gap (post hoc) |
| `results/v9rev_gate_20261122` | conflict gate ablation (post hoc, descriptive) |
| `results/v9rev_perm_20261122` | cluster permutation tests |
| `results/v9rev_cp_20261122` | Clopper-Pearson threshold sweep (post hoc) |
| `results/v9rev_tierB_20261122` | Tier B decision-flip rates (SigLIP2-I) |
| `results/5070-20261026-ab4` | registered run ab4 (CLIP-B/32, case I): metrics, split, quality report |
| `results/C_caseI_20261119` | Tier C (619-image index) metrics, McNemar, report |
| `results/rq1_20261119` | split and results of the v9 RQ-1 set-up |
