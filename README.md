# Comparing Confidence Scores for Refusal in Small-Gallery Fine-Grained Identification

Code, logged results, detail tables and documentation for the paper *Comparing Confidence Scores for Refusal in Small-Gallery Fine-Grained Identification* (ICISN 2027, Lecture Notes in Networks and Systems; under review).
Repository: https://github.com/nhatquyet1309/identity-level-selective-retrieval

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24454224.svg)](https://doi.org/10.5281/zenodo.24454224)

## Summary

In a preregistered benchmark, image-only SigLIP2 identified 80.6% of held-out query images against 51.9% for CLIP-B/32 (McNemar p = 1.7×10−9) on a small gallery. A system that refuses uncertain queries, rather than returning a wrong identity, can limit the cost of errors, so we compare four confidence scores (similarity, MSP, energy and the identity gap) for identity-level selective retrieval, which refuses when the gap between the best and second-best identity is small. On the 832 test queries of a 145-identity gallery, identity-level aggregation beats the image-level gap in both encoders. In the sealed preregistered re-test, the gap is better than similarity and energy and insufficient against MSP. On a common candidate set Yc (post hoc), MSP is slightly but significantly better than the identity gap in both encoders (95% CIs exclude 0), and the gap still beats similarity and energy. The registered operating point thresholds the absolute similarity, not the gap. Code, logged results and full result tables are publicly available at https://github.com/nhatquyet1309/identity-level-selective-retrieval.

The numbers in the tables below are copied from the logged result files and are the same as in the paper, except the rho column of the Tier B table (see the note under it). Details that the paper leaves out
for space (D11, lost images, the A-50 mislabel, the selection of Fig. 2 row 2, full confidence intervals) are in `docs/moved_from_paper.md`.

## Cite as

Tran Nhat Quyet, Nguyen Quang Thi, Nguyen Thanh Anh, Nguyen Quang Thinh. *Comparing Confidence Scores for Refusal in Small-Gallery Fine-Grained Identification*. Le Quy Don Technical University, Hanoi, Vietnam. ICISN 2027 (Lecture Notes in Networks and Systems), to appear; under review.

DOI: https://doi.org/10.5281/zenodo.24454224
Version DOI (v1.0.0): https://doi.org/10.5281/zenodo.24454225

```bibtex
@inproceedings{identitygap2027,
  author    = {Tran Nhat Quyet and Nguyen Quang Thi and Nguyen Thanh Anh and Nguyen Quang Thinh},
  title     = {Comparing Confidence Scores for Refusal in Small-Gallery Fine-Grained Identification},
  booktitle = {ICISN 2027 (Lecture Notes in Networks and Systems)},
  year      = {2027},
  note      = {To appear; under review},
  doi       = {10.5281/zenodo.24454224},
  url       = {https://github.com/nhatquyet1309/identity-level-selective-retrieval}
}
```

Affiliation of all authors: Le Quy Don Technical University, Hanoi, Vietnam.
Corresponding author: Nguyen Quang Thi. The machine-readable citation is in `CITATION.cff`.

## Repository structure

```
README.md        this file
requirements.txt          numpy pin for the statistics step
requirements-capture.txt  extra pins for the capture step (GPU)
LICENSE          MIT (code)
LICENSE-data     CC BY 4.0 (results/, tables/, docs/)
CITATION.cff     citation metadata
MANIFEST.md      description of every file
SHA256SUMS       sha256 of every file except itself
scripts/         the 14 analysis scripts of the results package
results/         13 folders of logged results (JSON, CSV, reports)
tables/          one CSV per supplementary table (23) and INDEX.md
paper/           PDF of the paper (identity_level_selective_retrieval_icisn2027.pdf)
docs/            Algorithm 1, reproducibility record, deviation log D1 to D22, protocols, Fig. 2 credits,
                 details moved out of the paper (moved_from_paper.md)
```

## Main results

Four scores on the same candidate set Yc, 832 test queries, main regime (post hoc, descriptive; table `tables/esm_s03_t1.csv`):

| Encoder | Score | E-AURC | AUROC |
|---|---|---|---|
| SigLIP2-I | similarity | 0.0579 | 0.7605 |
| SigLIP2-I | MSP | 0.0283 | 0.8701 |
| SigLIP2-I | energy | 0.0629 | 0.7466 |
| SigLIP2-I | identity gap | 0.0301 | 0.8621 |
| CLIP-B-I | similarity | 0.2115 | 0.6442 |
| CLIP-B-I | MSP | 0.1226 | 0.7781 |
| CLIP-B-I | energy | 0.2370 | 0.6200 |
| CLIP-B-I | identity gap | 0.1311 | 0.7583 |

Paired difference in E-AURC, gap minus the other score (negative: the gap is better), 95% cluster bootstrap interval over identities
(table `tables/esm_s03_t3.csv`; the Holm adjustment is used for the preregistered ab4 comparisons in `tables/esm_s08_t2.csv`):

| Encoder | gap - similarity | gap - MSP | gap - energy |
|---|---|---|---|
| SigLIP2-I | −0.0278 [−0.0496; −0.0074] (better) | +0.0018 [0.0004; 0.0034] (worse) | −0.0329 [−0.0580; −0.0104] (better) |
| CLIP-B-I | −0.0804 [−0.1232; −0.0397] (better) | +0.0086 [0.0023; 0.0150] (worse) | −0.1059 [−0.1522; −0.0621] (better) |

Sealed analysis (different candidate sets for MSP and energy, as preregistered): AURC for similarity, MSP, energy and gap is
0.0750, 0.0449, 0.0801, 0.0475 for SigLIP2-I and 0.4057, 0.3115, 0.4337, 0.3161 for CLIP-B-I.

Seen retrieval (Tier A), micro top-1 accuracy of the six encoder cells (`tables/esm_s08_t1.csv`):

| Cell | Micro top-1 |
|---|---|
| SigLIP2-S | 59.11% |
| SigLIP2-I | 81.12% |
| CLIP-B-S | 44.38% |
| CLIP-B-I | 46.00% |
| DINOv2 | 54.48% |
| CLIP-L/14 | 68.13% |

Tier B (eleven query transforms, SigLIP2-I only), group means (`tables/esm_s07_t3.csv`):

| Group | top-1 flip | Spearman | Kendall | decision flip | accept to refuse | refuse to accept | rho (preregistered; pooled, as in the paper) |
|---|---|---|---|---|---|---|---|
| canonical | 0.0282 | 0.9970 | 0.9652 | 0.0231 | 0.0154 | 0.0077 | 1.025 [0.980, 1.079] |
| stress | 0.0853 | 0.9811 | 0.9105 | 0.0716 | 0.0644 | 0.0072 | 0.878 [0.830, 0.925] |

Note on rho: the column above is the preregistered pooled ratio of the paper (`results/prereg_v9_gap_20261121/p2b_results.json`). The file
`tables/esm_s07_t3.csv` keeps the supplementary-table value of this column, which is the ratio of per-transform medians averaged over the
transforms of a group (post hoc; 1.035 canonical, 0.821 stress) and differs from the pooled numbers of the paper (1.025 / 0.878).

All remaining tables are in `tables/` (see `tables/INDEX.md`). Every analysis after the sealed protocol is post hoc and descriptive.
The permutation tests use whole identity clusters (cluster bootstrap and cluster permutation, not independent cases).

## Reproducing

Needs Python 3.10 or newer. Two dependency files are provided, one for each step:

```
pip install -r requirements.txt
pip install -r requirements-capture.txt
```

`requirements.txt` (numpy only) is enough for the statistics scripts. `requirements-capture.txt` adds torch, transformers, Pillow and faiss
for the capture step, which also needs a GPU and the gallery; the OpenAI `clip` package is also imported but its version is not recorded, so it is not pinned.
Versions come from the environment record of the capture machine (`docs/reproducibility.md`, `docs/repro_facts_v9j.md`); where the record is silent the file says "not recorded".
Seed 20261022. The scripts also import six internal modules of the original project (`siglip_engine`, `classify_visual`, `visual_lab`, `tierA_v2`, `tierA_v3`, `tierB_part2`);
they are not included in this repository, so the capture step cannot be re-run from it.

**No script runs from this repository alone.** Every script requires inputs that are not included (capture arrays `*.npz`, prediction
files `*.jsonl`, model checkpoints, the gallery images). The logged results in `results/` are the outputs, so the paper numbers can be
checked without re-running anything. The order of a full re-run, once the missing inputs have been regenerated (use fresh output folders):

```
python scripts/prereg_v9_capture.py --stage all
python scripts/prereg_v9_stats.py --dir <COPY_OF_results/prereg_v9_gap_20261121>
python scripts/prereg_v9_stats_det.py --out-dir <NEW_EMPTY_FOLDER>
python scripts/d13_yc_sensitivity.py --out-dir <NEW_FOLDER>
python scripts/v9rev_yc.py --out-dir <NEW_FOLDER>
python scripts/v9rev_perm.py --date <NEW_DATE>
```

| Script | Runs from this repository | Requires (not included) |
|---|---|---|
| `prereg_v9_capture.py` | no | GPU, gallery images, SigLIP2 and CLIP ViT-B/32 checkpoints, split JSON, transforms manifest |
| `prereg_v9_stats.py`, `d13_yc_sensitivity.py`, `v9rev_cp.py`, `v9rev_tierB.py` | no | `pass1_capture.npz` or `pass2b_capture.npz` of the v9 capture |
| `v9rev_split.py`, `v9rev_yc.py`, `v9rev_imggap.py`, `v9rev_gate.py` | no | ab4 `predictions.jsonl` (and `pass1_capture.npz` for the Yc scripts) |
| `v9rev_perm.py` | no | Tier C prediction files and `loo_Aii.csv` |
| `rq1_harness.py`, `sweep_thresholds.py` | no | `loo_Ai.csv`, `identity_audit.json`, `clip_b32_s2.npz` |

Scripts that write to a default folder under `results/` would overwrite the shipped results: always pass a new `--out-dir` or `--date`.

## No training (zero-shot)

No model is trained in this work. All encoders are used zero-shot with their published weights; there is no fine-tuning and no learned head.
The only quantities fitted on data are the temperature T of the MSP and energy scores and the acceptance thresholds, both chosen on the calibration split.

| Encoder | Checkpoint / model id | Revision |
|---|---|---|
| SigLIP2 | `google/siglip2-so400m-patch14-384` | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` |
| CLIP ViT-B/32 (OpenAI `clip` package) | weights `ViT-B-32.pt`, sha256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` | not recorded (commit of the `clip` package) |
| DINOv2 | `facebook/dinov2-base` | not recorded |
| CLIP-L/14 | `openai/clip-vit-large-patch14` | not recorded |

Only values found in the stored records are given; the sources are `docs/reproducibility.md` and `docs/repro_facts_v9j.md`.

## Verify without data

The repository ships logged results and hashes, so the files can be checked without the gallery and without any dataset. From the repository root:

```
sha256sum -c SHA256SUMS
```

Every line should print `OK`. This needs no data and no GPU. The paper numbers can be compared with `results/` and `tables/` directly.

## Requesting the gallery and embeddings

The gallery images and the capture arrays (embeddings) are not distributed. To request the gallery or the embeddings for a replication, contact the corresponding author (see the paper).

## Limitations

- The analyses after the sealed protocol (candidate-set comparison, image-level gap, conflict gate, permutation tests, threshold sweep, Why MSP beats the gap) are **post hoc** and descriptive; they do not replace the sealed analysis.
- Tier B was run for SigLIP2-I only; no CLIP-B-I Tier B number is given.
- There are no unseen identities: every query identity is in the gallery. Open-set operation is not evaluated.
- One seed and one split, so intervals are wide; the Clopper-Pearson threshold sweep has not been confirmed on test queries.
- The coarse class used to build the candidate set is an oracle, and the class-to-category mapping is not validated.
- Fig. 2 is illustrative and hand-picked (three rounds, documented in `docs/qualfig_row2_selection.md`).
- Dual use: the system is a decision-support component under human oversight; a person decides on every accepted or refused output.
- No gallery images and no query images are included, and no model weights or index files; `docs/figure_credits.md` lists only credits and source pages of the Fig. 2 photographs (Wikimedia Commons).

## License

Code: MIT (`LICENSE`). Data (`results/`, `tables/`, `docs/`): CC BY 4.0 (`LICENSE-data`). Third-party photographs are not distributed
and keep their own licences (see `docs/figure_credits.md`).
