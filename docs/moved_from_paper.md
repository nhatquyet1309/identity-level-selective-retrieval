# Details moved out of the paper

The paper (v10) keeps the results and drops the history of how the data were prepared. The details below are in this repository.
Each item names the logged files it comes from; nothing here changes a number of the paper.

## D11: the Tier C index and the 160 of 169 paired queries
The ab4 split has 790 index images (100 identities) and 284 held-out query images. For Tier C, 171 of the 790 index images could not be
recovered, so the Tier C index has 619 images of 93 identities. Of the 169 ab4 test queries with at least three candidates, 160 have their
identity in the reduced index (9 of 169 are excluded; 73 identity clusters), and the paired comparison of SigLIP2-I with CLIP-B-I uses these 160.
Sources: `deviation_log.md` (D11), `reproducibility.md`, `../tables/esm_s01_t1.csv`, `../results/C_caseI_20261119/`.

## The 171 lost images
The search for the 171 missing index images and the decision to continue on the 619-image index are recorded in `deviation_log.md` (D11)
and `reproducibility.md`. The loss affects Tier C only; Tiers A and B use the gallery of 1 667 images and 145 identities.

## A-50 mislabel
The v8 row-2 source A-50 of the qualitative figure was excluded after its trial-run scores were seen, because the gallery mislabels it
(A-50 for an E-3), and row 2 was rebuilt. Sources: `reproducibility.md` (A-50), `qualfig_row2_selection.md`.

## Selection of Fig. 2 row 2
Row 2 was chosen on the outcome, in three rounds (hand-picked, post hoc). All rounds, the twelve leave-one-out candidates and the effect of
the 15% crop are in `qualfig_row2_selection.md`, `../tables/esm_s12_t1.csv` and `../tables/esm_s12_t2.csv`. Credits: `figure_credits.md`.

## Full confidence intervals
The paper prints the paired E-AURC differences with 95% cluster-bootstrap intervals for the common candidate set (Table 2) and for the
sealed comparison against the gap. The complete set (every score, both encoders, both label regimes, the image-level gap, and the ab4
comparisons with Holm-adjusted p) is in `../tables/esm_s03_t3.csv`, `../tables/esm_s04_t1.csv`, `../tables/esm_s08_t2.csv` and `../results/`.
