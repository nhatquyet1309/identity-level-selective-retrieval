# prereg_v7 — 3-tier benchmark × 2 score cases (scales)

> English translation of the original (Vietnamese) document; content, numbers and codes are unchanged. The sha256 recorded in `prereg_v7.md.sha256` is the hash of the original Vietnamese file, not of this translation.

- **Status: SEALED — 17/11/2026** (sha256 in `prereg_v7.md.sha256`; artefacts pinned in §8). v1 (14/11/2026) → v2 (14/11, review #1) → v3 (14/11, review #2) → v3.1 (16/11: fixed §9.1, closed pending item #3, 2 wording patches) → **SEALED (17/11: θ_cos fixed at 0.98 · T\*/τ\* confirmed · the project owner signed "ok 0.98 + seal")**.
- Follows, without modifying: `prereg_v6.md` · `proposal_giam_nguong_20261109.md` · `VISUAL_LAB_MOTA.md` (13/11 version). Hashes in §8.
- **R2/R9:** every parameter below is declared **before running**; after running, report every cell — including bad ones. No parameter is chosen/changed after seeing the numbers.
- **[FIXED]** = a parameter confirmed by the project owner (§9.1) — **all are fixed**: θ_cos = **0.98 fixed on 17/11** after viewing `dedup_hist.txt`. The [PROPOSED]/[TENTATIVE] labels in the body text have all been replaced (the review history table is kept verbatim).

## REVIEW HISTORY

**Review #1 (v1 → v2)** — 10 items, patched in v2 and **kept unchanged in v3** (except the points corrected below): RQ-1 answered by measurement; the 2 S≡I cells are a deterministic check; near-duplicate audit; denominator for 1-image ids, top-k by identity, report micro/macro/category/strata; tie-break and exact FAISS; bootstrap by identity; tier C paired; downgrade raw-score comparison to descriptive; pin hash/env; close the pending items before running tier A.

**Review #2 (v2 → v3)**:

| # | Review issue | Fix in v3 |
|---|---|---|
| 1 | The v2 RQ-1 criterion measured *performance equivalence*, not *score comparability* | §7.4: add criterion (b) **threshold transfer**; `COMPARABLE` requires both (a) §7.3 and (b) §7.4 |
| 2 | `COMPARABLE` is trivially met when both coverages are small | §7.3: **coverage floor** c_min [PROPOSED] — below the floor = `FAILS p*` |
| 3 | τ* is chosen and evaluated on the same data | §7.1: calib/test split by identity; §7.3: τ* via the ab4 T* procedure (CP bound); the v2 empirical rule becomes a sensitivity |
| 4 | Missing CI, missing rule when an encoder is infeasible, pair family not declared | §7.5: CI-based conclusion labels (adding the label `NOT ENOUGH EVIDENCE`), priority order, a family of 7 pairs, flag ⚑ when the calib↔test reversal disagrees |
| 5 | RQ-1 is computed on tier A (seen) but the threshold is used for unseen | §7.6: declare the scope of validity |
| 6 | Embedding invariance (original–transformed cosine) also depends on the scale | §5.3: cosine is descriptive only **within** an encoder; cross-encoder comparison by self-retrieval@1 after transformation (scale-independent) |
| 7 | The Δ baseline (tier B) is ambiguous between A(i) and A(ii) | §4.2: A(ii) is the baseline; §5.3: Δ paired on the same D(ii) |
| 8 | Denominator after de-duplication not declared | §4.2: D(ii) |
| 9 | Cosine threshold chosen blindly; union-find may chain | §4.1: unlabelled histogram before SEAL (§9.2), θ_cos + sensitivity, report the component distribution |
| 10 | Wilson assumes independence, whereas images of the same id are correlated | §4.6: bootstrap by identity is the **primary** CI; Wilson is reference only |
| 11 | "Exact match 81.4%" is a rounded number; the gallery hash of the publication run is missing | §9.2 (`ref_814.txt`: k/N + hash), §9.3 step 1; a fallback exists when it cannot be traced |
| 12 | *(found while writing v3)* self-retrieval = 100% can be wrong when there are pixel-identical images (score ties) or when the S score has a text component | §4.0: success = reaching the maximum score (ties accepted), measured on pure-image cosine; the S cell is report-only, not a stopping condition |
| 13 | Mechanical errors: ~10 broken cross-references; 73.328 → 73.348; "AURC" without a definition; the names `H1`/`H1'` easily confused; "hue in RGB" | Renumbering + `check_xrefs.py` (§9.2); §5.3 (73.348); §7.2 (AURC); §5.1 (names G/L/X, hue in HSV) |
| 14 | Claim scope: SigLIP2-S has no tier C figure; SigLIP2-I can be paired with CLIP-B-I | §0.2; §6.4 |
| 15 | Comparing S with I has no rule at tier A/B; w_img is not optimised so the claim must be limited | §4.7 |
| 16 | *(found additionally)* v2 dropped the "Known deviations" section and the WebP/GIF convention of v1 | §10 (restored + extended) |

**Patches after review #2 (14/11 — only fixed references/clarifications, did NOT change parameters):** (i) §7.3 splits the 2 steps T\*/τ\* and points to the correct source (`spec_v2.md` + `benchmark_checklist.md` §0 + paper §4 — prereg_v6 does **not** contain this procedure); (ii) the corresponding §9.2 checklist item; (iii) §4.0 states explicitly that the stopping condition applies to the **4 pure cosine measurements**, the case S cell is report-only.

**Fixed 16/11 (project owner):** §9.1 confirms 6 parameters (pHash ≤ 4 · c_min 10% · ε_p 2pp · tolerance 1e-5/1e-3 · seed 20261022 · split 76/75); **θ_cos = 0.98 is tentative**, formally fixed after `dedup_hist.txt`. **Pending item #3 closed:** C-case I is **included in the paper** — an exploratory supplementary table, **reported regardless of the result**. Two wording patches: removed the patch note in §7.3 (the trace is kept in this log line), §7.2 distinguishes AURC/E-AURC.

## 0. Research question (RQ-1) and claim scope

### 0.1 RQ-1

> **"Are the scores of the 4 encoders comparable with each other?"** — answered **by measurement**, not by rule.

An encoder pair is `COMPARABLE` only when it satisfies **both** (a) operational equivalence at p* (§7.3) and (b) threshold transferability between the two encoders (§7.4). Parameters declared in advance: **p\* = 0.95 · δ\* = 5 percentage points** (p\* = 1 − α of ab4; not optimised afterwards). The conclusion for each pair is one of four labels (§7.5): `COMPARABLE` / `NOT COMPARABLE` / `NOT ENOUGH EVIDENCE` / `FAILS p*` — all four can occur.

### 0.2 Claim scope

- **Primary (tier C) is CLIP-B/32 case S** (ab4: 98/169 = 58.0%). The production system uses SigLIP2 but **SigLIP2-S has no figure at tier C** (§3.3) → the paper **does not extrapolate** the ab4 figure to SigLIP2.
- Tier A/B and RQ-1 are **diagnostic** on the 1,667 gallery (seen/LOO): not accuracy, not extrapolated to unseen (§7.6).
- Comparing S with I (§4.7, §6.3) tests **one configuration point w_img = 0.5 (not optimised)**; it does not answer the question "does text help".

## 1. Context & scope

A benchmark of 3 tiers of increasing difficulty to separate 3 kinds of failure: **plumbing → invariance → generalisation** (§4–§6), each tier measured under **2 score cases** (§2). Tier C is the paper's primary (ab4 protocol unchanged); A/B are diagnostic; §7 answers RQ-1.

## 2. Two score cases (declaration)

| Symbol | Definition | Source |
|---|---|---|
| **S** (system) | `use_text=True, w_img=0.5` for SigLIP2 + CLIP-B/32 (score = 0.5·image cosine + 0.5·text cosine); DINOv2 + CLIP-L image-only | `visual_lab.py:594` `use_text=(_enc not in ("dinov2","clipl14"))`, `classify_visual.py:96-103` |
| **I** (image-only) | `use_text=False` for all 4 encoders → score = pure image cosine | same code, flag `use_text=False` |

- **No index rebuild** — `use_text` is a query parameter; the index is unchanged (1,667 vectors/encoder).
- `w_img = 0.5` is kept as is, **not optimised** (optimising after seeing the numbers = a violation of R2).
- Case S: **comparing raw scores between blend and pure is forbidden**; the per-tier Δ can be compared within the same encoder. Case I: raw-score comparison only at a **descriptive** level (§7.6); thresholds remain separate per encoder.

## 3. Cell matrix — stating which cells exist

### 3.1 Tier A/B: 8 cells, **6 distinct cells**

| Encoder | Case S | Case I | Status |
|---|---|---|---|
| SigLIP2 | cell 1 | cell 2 | independent |
| CLIP-B/32 | cell 3 | cell 4 | independent |
| DINOv2 | cell 5 | cell 5 | **S ≡ I bit-exact — deterministic check** |
| CLIP-L/14 | cell 6 | cell 6 | **S ≡ I bit-exact — deterministic check** |

- **Deterministic check:** embeddings are computed **once** per encoder and cached (record the cache hash); run the query path with both flags and compare the **entire result array (path + score)** — they must match bit-exact. A mismatch = a bug → **stop, write a log**; it is not "independent evidence". There are 6 independent cells + 2 check cells.

### 3.2 Number of measurements

- **Tier A:** 1,667 × 8 = **13,336** (10,002 independent + 3,334 deterministic checks), for each of the A(i)/A(ii) variants of §4.2.
- **Tier B:** 1,667 × 11 transformations = **18,337 queries** × 8 cells = **146,696** (110,022 independent + 36,674 deterministic checks).
- **Tier B embedding metric (§5.3):** 1,667 × 11 × 4 encoders = **73,348 pairs** (cases not separated).

### 3.3 Tier C: 4 cells, only 3 exist

| Encoder | Case S | Case I |
|---|---|---|
| CLIP-B/32 | **ab4 — ALREADY CLOSED** (1 set of numbers, not re-measured) | new run — exploratory |
| SigLIP2 | **DOES NOT EXIST** (never run, no speculation) | new run — exploratory |
| DINOv2 / CLIP-L | no tier C (neither S nor I) — out of scope, declared in advance | |

- "Report every cell" at tier C = report **exactly the 3 existing cells** + state clearly that 1 cell does not exist; no invented numbers for the empty cell.

## 4. Tier A — Seen / self-retrieval + leave-one-out (sanity, NOT the main benchmark)

### 4.0 Self-retrieval (BEFORE leave-one-out — checks the ID–vector match and the preprocessing path)

- Each image is **re-embedded from file** using exactly the preprocessing path used when building the index, then queried against the **full index, with nothing removed**. Score = **pure image cosine** (for all 4 encoders).
- **Success** if no candidate has a score greater than the image's own score beyond the tolerance **1e-5 (fp32 index) / 1e-3 (fp16 index)** [FIXED 16/11; declare the dtype in `env_capture.txt`]. **Ties are accepted** (pixel-identical images), but the number of tie cases must be reported.
- **Stopping condition:** the **4 pure image cosine measurements** (exactly 1 per encoder — corresponding to case I; for DINOv2/CLIP-L the two cases coincide) must reach 1,667/1,667; a shortfall → a meta/vector matching bug or a preprocessing bug → **stop**.
- Case S cells (score with a text component) are **report-only**, not a stopping condition: the text component can legitimately let another candidate surpass the image itself.
- This is a check metric, not a recognition result.

### 4.1 Universe & near-duplicate audit

- Production gallery 13/11 (**1,667 images / 151 ids / 8 categories**) + the 4 current indexes.
- **Step 0 (unlabelled, done BEFORE SEAL, §9.2):** for each image compute the largest SigLIP2 image–image cosine to another image → write the histogram to `dedup_hist.txt`. This step **does not read the top-1 or labels**; the project owner uses it to confirm θ_cos.
- **Duplicate edges (declared in advance):** (a) pHash Hamming ≤ **4** [FIXED 16/11] within the same category; (b) SigLIP2 image cosine ≥ **θ_cos = 0.98** [**FIXED 17/11** — see `dedup_hist.txt`: no elbow, ≥0.98 = 112 images (6.7%) → 0.98 is conservative; sensitivity: **0.95**]. Merge connected components by union-find. **A single merge** is used for all 4 encoders (fairness across encoders).
- **Report the component distribution:** number of components, maximum size, P99, number of images belonging to components of ≥ 2 images, number of components containing > 1 identity (near-duplicates with different labels — merge is still kept). If any component has **> 20 images** (exceeding the maximum n_c) → record a deviation (not an error), keep the declared merge, and additionally report a sensitivity using only pHash edges.
- Pin the gallery/index hash of the 81.4% publication run (§9.2) to separate "mismatch due to the gallery" from "mismatch due to a bug".

### 4.2 Variants & denominators (declared in advance — decides the 81.4% match and the tier B baseline)

| Variant | Removed from the index when querying image x | Denominator |
|---|---|---|
| **A(i)** raw LOO | only x itself | D(i) = images x whose identity still has ≥ 1 other image in the gallery (= n_c ≥ 2) |
| **A(ii)** LOO + de-duplication — **primary diagnostic** | x and every image in the same duplicate component as x (θ_cos = 0.98) | D(ii) = images x whose identity still has ≥ 1 image **after** removing x and the component of x; D(ii) ⊆ D(i) |
| **A(ii-s)** sensitivity | as A(ii) with θ_cos = 0.95 | D(ii-s) defined analogously |

- The **primary** denominator of each variant = the corresponding set D; images outside D are **excluded and counted** (n_c = 1; all same-id images lost after de-duplication) — report the count of each group.
- **"Count as wrong" sensitivity:** images outside D count as wrong, denominator = 1,667. Both sets of numbers must be printed.
- The 81.4% comparison is against **A(i)** of SigLIP2-S; record whichever denominator set matches (§9.3).
- **The tier B baseline and the RQ-1 universe = A(ii) on D(ii).** The difference (i) − (ii) = the inflation due to duplicates.

### 4.3 Top-k by **identity** (not by image)

- Primary: aggregate scores by identity using **max** (same convention as the paper) → top-1/top-3 identity.
- Image-level top-k is used only in self-retrieval (§4.0) — because an image-level top-3 may be 3 copies of 1 id, which is meaningless.

### 4.4 Full reporting

- **micro** (all of D) + **macro** (average by identity) + **8 categories** + **n_c strata** (<8 · 8–15 · 16–20), for A(i) and A(ii).
- Together with the n_c distribution (36 ids with < 8 images skew the result — this must be clearly visible in the table).

### 4.5 Ties & search

- **FAISS IndexFlatIP — exact search, no IVF/PQ** (verify by code before running; record the verification result).
- Ties between images: sort by `(−score, path)`; ties between identities after the max aggregation: sort by `(−score, identity_id)` — deterministic, reproducible.

### 4.6 Statistics

- **Primary CI** for all proportions and all differences: **cluster bootstrap with 2,000 resamples by identity (151 clusters)**, 95% percentile CI, seed **20261022** [FIXED 16/11]. Each resample draws identities with replacement and recomputes **all** quantities on the same selected set (so that differences are paired).
- **Wilson 95%** is a **reference** number only (it assumes independent images, hence is narrower than reality — noted right in the table).
- Bootstrap by source image: sensitivity, reported alongside.

### 4.7 Comparing S with I (only SigLIP2 and CLIP-B/32 — the two encoders with S ≠ I)

- Difference top-1(I) − top-1(S) on D(ii): **paired** — McNemar exact + paired bootstrap CI by identity.
- At tier B: compare Δ(S) with Δ(I) of the same encoder (the difference of two Δ), bootstrap CI by identity; **descriptive**.
- Claim limited per §0.2: only the configuration w_img = 0.5.

## 5. Tier B — Transformed (invariance, diagnostic)

### 5.1 Transformations — 11 of them, deterministic, no RNG, 3 groups

| Group | # | Transformation | Status |
|---|---|---|---|
| Geometric | G1 | hflip | canonical |
| | G2 | vflip | **stress** — physically implausible, separate row |
| Colour — light | L1 | brightness ×0.8 | canonical |
| | L2 | brightness ×1.2 | canonical |
| | L3 | saturation ×0.8 | canonical |
| | L4 | contrast ×0.8 | canonical |
| | L5 | hue +0.05 of the colour wheel (HSV, = +18°) | canonical |
| Colour — heavy | X1 | brightness ×0.5 | **stress** |
| | X2 | brightness ×1.5 | **stress** |
| | X3 | saturation ×0.4 | **stress** |
| | X4 | contrast ×0.5 | **stress** |

- **Canonical** = G1, L1–L5 (6 transformations). **Stress** = G2, X1–X4 (5 transformations) — **reported separately**, not merged into the canonical figure. The heavy set was added because the light set is expected to give Δ ≈ 0, unable to distinguish the encoders (declared in advance).
- **A single library** for all transformations; brightness/contrast/saturation follow the definitions of that library; hue is shifted in HSV. Work on decoded 8-bit RGB images, clipped to [0, 255], rounded to 8-bit. The transformation-generating code is `make_transforms.py` (sha256 pinned in §8; library + version recorded in `env_capture.txt`).
- Transformed queries are saved as **PNG (lossless)** — **no JPEG re-encoding** (avoids compression noise).

### 5.2 Procedure

- Query the transformed image T(x) against the **unchanged** index (no index augmentation); remove x **and the entire duplicate component of x** (exactly the removal of A(ii), §4.2).
- The denominator **= D(ii)** of the source image x, **shared across all 11 transformations** → Δ paired on the same set.

### 5.3 Metrics (reusing embeddings — cheap)

1. **Δ top-1** = tier B top-1 (transformation T) − **A(ii)** top-1, same cell, same D(ii). Cluster bootstrap by identity (§4.6); **pointwise** 95% CI.
2. **Flip rate:** % of images in D(ii) whose top-1 identity of T(x) ≠ the top-1 identity of x, together with the number of **lost** cases (correct → wrong) and **regained** cases (wrong → correct). Δ top-1 = 0 can still hide two-way label changes — reporting this alongside is mandatory.
3. **Embedding invariance** (unlabelled; cases not separated since the embedding is pure image; 1,667 × 11 × 4 = **73,348 pairs**):
   - **3a. cosine(e(x), e(T(x)))** — mean, median, P5 by encoder × transformation. **Compared only WITHIN the same encoder** (cosine depends on the encoder's scale; cross-encoder comparison is forbidden).
   - **3b. self-retrieval@1 after transformation** (scale-independent → **cross-encoder comparison allowed**): query T(x) against the **full index** (x still in the index) by pure image cosine; success if the top-1 image belongs to the duplicate component of x (including x itself). Report % + bootstrap CI by identity. This measurement depends on the gallery but not on the scale.

### 5.4 Tier B conclusion rule

- **Descriptive, no p-value** — Holm is not needed; the CI is **pointwise, not simultaneous** (stated explicitly, not phrased as an overall test).
- **Invariance margin declared in advance: −2.0 percentage points** (applies to both the canonical and stress groups; many stress cells being `INVARIANCE LOST` is a normal expectation). Each cell gets one of three labels:
  - `INVARIANT` if the lower bound of the 95% CI of Δ ≥ −2.0pp;
  - `INVARIANCE LOST` if the upper bound of the 95% CI < −2.0pp;
  - `NOT ENOUGH EVIDENCE` if the CI crosses the margin.
  - Statistical non-significance ≠ invariance — this metric decides.
- Comparing Δ **between encoders**: only within the same case; cross-case comparison: only for the same encoder (§4.7). Tier B samples are **not** pooled into the tier C CI.

## 6. Tier C — Unseen (PRIMARY)

### 6.1 Existing cells

See §3.3 — 3 cells (ab4-S/CLIP-B already closed; new run I/CLIP-B; new run I/SigLIP2), 1 cell does not exist.

### 6.2 Protocol (unchanged)

- `snapshot_S2` (1,489 files / 1,482 images, read-only), seed 20261022, test stratum n_c ≥ 3 **n = 169**, 4 scores on the same prediction set, T* on calib, cluster bootstrap 2,000, Holm over 3 comparisons, CP one-sided 95% @α=5%, "infeasible" is not relaxed.
- **Case S (CLIP-B/32): keep the ab4 figure unchanged** — top-1 **98/169 = 58.0%** (Wilson 95% ≈ **[50.5; 65.2]**, reference), top-3 79.9%; **not re-measured, not replaced**.
- **Case I:** 2 new runs (CLIP-B-I, SigLIP2-I), same setup as ab4, **changing only the `use_text` flag**; separate run-ID `<GPU_SERVER>-<date>-<tag>`; labelled **exploratory** — **included in the paper as a supplementary table, reported regardless of the result (fixed 16/11, §9.2)**. Every deviation from case S must be explainable by code (only the flag differs) and recorded in the deviation log.

### 6.3 Comparing S with I — paired (not comparing 2 separate CIs)

- Only **CLIP-B/32** has both S + I. Same 169 queries → **McNemar exact** + **difference top-1(I) − top-1(S) with a 2,000 paired bootstrap CI** (by identity).
- **Power limit declared in advance:** n = 169 → differences below about **10 percentage points** (depending on the proportion of discordant pairs) are almost undetectable; if undetected → the correct conclusion is "not detectable", **not** written as "the same".

### 6.4 SigLIP2-I versus CLIP-B-I — cross-encoder pairing (exploratory)

- SigLIP2-I has no S counterpart, so report (i) standing alone and (ii) **paired with CLIP-B-I on the same 169 queries**: McNemar exact + paired bootstrap CI by identity (top-1; top-3 descriptive).
- Same power limit as §6.3. This is **not** an RQ-1 test (RQ-1 is only in §7) and is not turned into a claim about "which encoder is better" beyond the scope of the 169 queries.

## 7. RQ-1 — Comparing score scales with scale-independent metrics

### 7.1 Universe & calib/test split

- **Universe: A(ii) on D(ii)** (de-duplicated, primary denominator §4.2), **6 independent cells** (§3.1).
- **Split by identity** (avoids leakage between images of the same id): sort the 151 identities by name (UTF-8 byte by byte), `random.Random(20261022).shuffle(...)` [FIXED 16/11: the split method]; **the first 76 ids = calib, the last 75 ids = test**. A query belongs to the group of its identity. The index remains the full gallery (LOO + de-duplication as in §4.2).
- **Main direction:** τ* chosen on calib, evaluated on test. The **reverse direction** (calib ↔ test) is a sensitivity (§7.5).

### 7.2 Per-cell metrics (descriptive + ranking) — computed on all of D(ii), no τ needed

| # | Metric | How reported |
|---|---|---|
| 7.2.1 | **Distribution of the top-1 score: correct vs wrong** | median + Q1/Q3 + P10/P30/P50/P70/P90 for each group; KS distance between the 2 distributions |
| 7.2.2 | **AUROC of the top-1 score → correct/wrong** | point + cluster-bootstrap CI by identity |
| 7.2.3 | **Precision(τ), coverage(τ) curves and AURC** | table/plot on a grid of all distinct values of the top-1 score; **AURC = (1/n) Σₖ risk@k**, k = 1..n in order of decreasing score (risk@k = the proportion wrong among the k highest-scoring images; lower is better) |

- **Distinguishing two quantities (patch 16/11):** **AURC** in 7.2.3 is the **plain AURC** (uncompensated) — a rank metric comparing encoders; **E-AURC** (= AURC − AURC_optimal, per the code of the ab4 run) is a **different** quantity, appearing only in **step 1 of §7.3** when choosing **T\*** for MSP/energy. The two symbols stand in two different places and are **not interchangeable**.
- Cross-encoder conclusions use only the **rank metrics** (7.2.2, 7.2.3) **+** the criteria of §7.3–§7.5.

### 7.3 Criterion (a) — operational equivalence at p*

- **τ\*_e** of cell e: chosen on **calib**, score = the top-1 score by identity, **α = 1 − p\* = 5%**, in the **2 steps** of exactly the ab4 procedure:
  1. **T\* (temperature)** — only for temperature-dependent scores (MSP, energy): T\* = argmin E-AURC on calib. Source: `spec_v2.md` + `benchmark_checklist.md` §0. The absolute similarity / identity gap scores **have no temperature** → skip this step.
  2. **τ\* (threshold)** = the **smallest** threshold whose one-sided 95% Clopper–Pearson upper bound ≤ α on calib. Source: paper §4 / `spec_v2.md`. If no feasible τ exists ("infeasible", not relaxed) → the pair containing e is `FAILS p*`.
- On **test**: precision_e(τ\*_e) and coverage_e(τ\*_e), bootstrap CI by identity.
- **Coverage floor [FIXED 16/11] c_min = 10%:** coverage_e(τ\*_e) < c_min → considered operationally unusable → `FAILS p*`. (Blocks the trivial "comparable" case when both coverages are very small.)
- **D_cov** = coverage₁(τ\*₁) − coverage₂(τ\*₂), paired bootstrap by identity on test. (a) is **met** if the CI95%(D_cov) lies entirely within [−δ\*, +δ\*]; **failed** if the CI lies entirely outside [−δ\*, +δ\*]; otherwise **insufficient**.
- **Sensitivity S1:** the v2 rule (the smallest τ with empirical precision ≥ p\* on calib) run in parallel, reported alongside; **does not replace the primary**.
- Note: the bootstrap CI does **not** include the uncertainty of τ\* (chosen on calib); the calib ↔ test reversal (§7.5) is a compensating check.

### 7.4 Criterion (b) — threshold transfer

For a pair (e₁, e₂) consider both directions (1→2 and 2→1). Direction k→j: take the **exact value** τ\*_k (from the calib of k), apply it to the top-1 score of e_j **on test**, and compare e_j with e_k itself at the same τ\*_k on the **same queries**:

- D_prec = precision_j(τ\*_k) − precision_k(τ\*_k); D_cov = coverage_j(τ\*_k) − coverage_k(τ\*_k); paired bootstrap by identity.
- That direction is **met** if CI(D_prec) ⊂ [−ε_p, +ε_p] **and** CI(D_cov) ⊂ [−δ\*, +δ\*], with **ε_p = 2.0 percentage points** [FIXED 16/11]; **failed** if either CI lies entirely outside the margin; otherwise **insufficient**.
- (b) is **met** if both directions are met; **failed** if either direction failed; otherwise **insufficient**.
- This is a **test by measurement**, not an assumption that "scores are comparable"; it applies only to pairs of the same scale type (the pair family in §7.5).

### 7.5 Per-pair conclusion, pair family, sensitivity

- **Priority order of labels:**
  1. `FAILS p*` — one of the two encoders is infeasible or below c_min (§7.3);
  2. `NOT COMPARABLE` — (a) failed or (b) failed;
  3. `COMPARABLE` — (a) met **and** (b) met;
  4. `NOT ENOUGH EVIDENCE` — everything else.
- Each label carries a reason code (a/b) to show which criterion decided.
- **The pair family declared in advance = 7 pairs**: case I — 6 pairs among {SigLIP2-I, CLIP-B-I, DINOv2, CLIP-L}; case S — **1 pair** (SigLIP2-S, CLIP-B-S; same blend scale type, same w_img). Blend–pure pairs are **not** considered (different scale types, §2). Descriptive, **no p-value, no Holm**; pointwise CI.
- **Calib ↔ test reversal:** if the label differs from the main direction → flag **⚑**; cells with the ⚑ flag **may not be used as a claim**.

### 7.6 Scope of validity & prohibitions

- RQ-1 is valid on the **LOO of the production gallery (seen)**; it is **not extrapolated** to unseen/tier C (n = 169 is not enough to repeat §7.3–§7.4 there).
- Case S: raw cross-encoder score comparison is forbidden (unchanged); the §7.4 test is a **measurement**, not a raw comparison.
- Case I between encoders: the raw-score table **only stands side by side (descriptive)**, with no conclusion — "same scale type" ≠ "same scale of measurement" (the DINOv2 cosine distribution ≠ CLIP).

## 8. Pinning sources (sha256 — computed 14/11/2026)

| File | sha256 |
|---|---|
| `prereg_v6.md` | `3be5dea5be34f6fe78934e85a5fd88c1e51ba8e3151af8168c106b073f684c60` ✓ matches `prereg_v6.md.sha256` |
| `proposal_giam_nguong_20261109.md` | `3685087fdcfff6e9eda05523c111aeea4c53a250cab819921a879578efba9378` |
| `VISUAL_LAB_MOTA.md` (13/11 version — **a living document**, hash = exactly the version read) | `360ee7951ad428809d7c20f587cc55036b380fec7104b80d14bdec25a0459475` |

- If MOTA changes later → this hash is still the version that was used; if the content of MOTA §3 (81.4%) is edited, a new version must be recorded + the discrepancy explained.
- **Artefacts before running — SHA256 ALREADY PINNED (16–17/11)**, in the project's `artifacts_prereg_v7/` directory (4 txt) and `scripts/` directory (3 py) (internal paths, not included):

| Artefact | sha256 |
|---|---|
| `scripts/check_xrefs.py` | `cfb9eab1db5968526ba3f9e8b85754843b0abd45118f876879c985a175871d9f` |
| `scripts/make_transforms.py` | `e6f3b41afad9ca04eddcfc4025cc541f1e7d1f0f10727f17594bbd4fb2ebdd60` |
| `scripts/collect_artifacts_v7.py` | `477b20897db0babef09e3f0968db95aeb0e53b7c7d38034341440eae8a02731e` |
| `dedup_hist.txt` | `0cbb762fa7c87deb0dcbec9d502c21919f4c8dcc40ebc22037e1a66c0d9347a1` |
| `ref_814.txt` | `78279091fb3e886748374d53017942ee7d69504051d9d8133401612b740c40ff` |
| `gallery_hash_20261116.txt` | `0f26962f08d852ad90827a659295610a81a2a5be4e043616fc6b586e5fce89c1` |
| `env_capture.txt` | `b5871de1cde870b8ccc9dc340299bb01069586ea45d963617c390c318cb105b8` |

- θ_cos fixed at 0.98 (17/11) → no artefact is left pending creation; §9.2 is complete.

## 9. SEAL conditions & run order

### 9.1 Parameters — **FULLY FIXED (16–17/11; θ_cos fixed 17/11)**

| Parameter | Fixed value (16/11) | Basis / note |
|---|---|---|
| θ_cos (duplicate edge by cosine) | **0.98**; sensitivity 0.95 | ✅ **fixed 17/11** after viewing `dedup_hist.txt` (P50 0.923 · ≥0.98 = 112 images 6.7% · no elbow → 0.98 is conservative: removes only near-perfect twins) — §4.1 |
| pHash Hamming | ≤ 4 | ✅ fixed 16/11 (already declared in v2) |
| c_min (coverage floor) | 10% | ✅ fixed 16/11 — below this level the threshold keeps < 10% of queries, unusable operationally — §7.3 |
| ε_p (threshold-transfer precision margin) | 2.0 percentage points | ✅ fixed 16/11 — same margin as §5.4 — §7.4 |
| Self-retrieval tolerance | 1e-5 (fp32) / 1e-3 (fp16) | ✅ fixed 16/11 — numerical noise — §4.0 |
| Bootstrap and calib/test split seed | 20261022 | ✅ fixed 16/11 — same seed as ab4 — §4.6, §7.1 |
| Calib/test split method | shuffle identities with `random.Random(seed)`, 76 calib / 75 test | ✅ fixed 16/11 — §7.1 |

### 9.2 Things that must be done BEFORE running anything

- [x] **Pending item #3 — CLOSED (16/11):** tier C-case I is **included in the paper** as an **exploratory supplementary table**, **reported regardless of the result** (even when no difference from case S is detected).
- [x] **Unlabelled step 0 (COMPLETED 17/11):** `dedup_hist.txt` run 16/11 (top-1 not read), histogram viewed, **θ_cos = 0.98 FIXED** in §9.1.
- [x] **Confirm the T\*/τ\* procedure in §7.3 — CONFIRMED (17/11): APPLICABLE.** Cross-checked sources: `spec_v2.md` (CP one-sided α) + `benchmark_checklist.md` §0 (T\* = argmin-E-AURC on calib, frozen) + paper §4 (τ = the smallest threshold with CP ≤ α; risk = *accepted-and-wrong over accepted*, `content_v5.py` line Calibration rules). The tier A score = the top-1 score by identity (§7.3), with the same risk definition as the paper → **step 1 (T\*) N/A** because cosine-family scores (case S/I) have no temperature — §7.3 already declared "skip this step" for exactly this kind; **step 2 (one-sided 95% CP ≤ α)** applies in full. No difference → no need to note anything elsewhere in §7.3.
- [x] **`ref_814.txt` (16/11):** **k/N + hash COULD NOT BE TRACED** (grepped tools/ + the 12–14/11 files + visual_logs) → use the **fallback branch of this very item**: the 81.4% match in §9.3 step 1 becomes a reference check with |deviation| ≤ 0.5 percentage points.
- [x] On <GPU_SERVER>: **`gallery_hash_20261116.txt` (16/11)** = sha256 of the gallery file list (**1,667 images**, agg listing `08bda4fc…`) + the 4 index files. *(Renamed from …20261114 to match the true creation date; content generated 16/11.)*
- [x] **`env_capture.txt` (16/11):** python 3.10.12 · faiss 1.15.1 · torch 2.14.1+cu130 · PIL 12.3.0 · numpy 2.2.6 · torchvision 0.29.1; **4 indexes = IndexFlatIP (exact), fp32, ntotal 1,667** ✓; checkpoint + preprocessing extracted from the build scripts; **imagehash NOT yet installed** (needed for pHash §4.1a).
- [x] `make_transforms.py` (16/11): **selftest 11/11 PASS** on <GPU_SERVER> (deterministic, RGB, correct class per §5.1); sha256 pinned in §8 at SEAL time.
- [x] **`check_xrefs.py` (16/11): run on the version to be hashed → BROKEN REFS = 0.**
- [x] **Project owner signed (17/11: "ok 0.98 + seal")** → `sha256sum prereg_v7.md` → **SEALED** (hash recorded in `prereg_v7.md.sha256`).

### 9.3 Run order (no skipping)

1. **Tier A (smoke):** (a) self-retrieval §4.0 → (b) 2 bit-exact deterministic check cells → (c) report the duplicate audit §4.1 → (d) **A(i) SigLIP2-S matches `ref_814.txt`**: the number of images exactly k coincides at the corresponding denominator (D(i) or 1,667 — record which one matches); if only a rounded number is available, it matches when k/N rounded to 1 decimal = 81.4% and state clearly "matches at the rounding level". A mismatch → stop, record a deviation, await direction → (e) run A(ii) and A(ii-s).
2. **Tier B** (11 transformations × 8 cells) + the embedding metrics (§5.3).
3. **RQ-1** (§7) on the tier A numbers.
4. **Tier C-case I:** 2 exploratory runs + §6.3 + §6.4.

### 9.4 Reporting rules (acceptance conditions)

1. Report **every existing cell**; a non-existent cell is stated clearly as "does not exist"; bad cells are still reported.
2. Each table: run-ID + seed + env hash + date + machine (<GPU_SERVER>).
3. Numbers entering the paper: ab4 (C-CLIP-B-S) · the 2 C-case I runs (exploratory) · A/B with run-IDs.
4. A/B are diagnostic, C is primary; A/B are not upgraded to "accuracy".
5. Each conclusion uses exactly the labels of §7.5 / §5.4 / §6.3, without making them synonymous; ⚑-flagged cells are not used as claims.
6. Every CI states what it is clustered by (identity is primary); Wilson is only noted as "reference".

## 10. Known deviations (declared in advance, not errors) & input conventions

- Tier A/B universe = the production gallery 1,667 (13/11); tier C universe = `snapshot_S2` 1,482 — **two different universes by design**; absolute numbers are not compared between tier A/B and tier C (the A→C gap is partly due to duplicates and to different encoders: A uses SigLIP2, C primary uses CLIP-B/32).
- 36 identities with < 8 images skew the tier A/B n_c distribution relative to the snapshot — report the n_c distribution alongside (§4.4).
- Case I is a configuration **never run** on the system (the `use_text` decision is pending in MOTA §10) — running this v7 is exactly what resolves the pending item.
- SigLIP2-S has no tier C figure; DINOv2/CLIP-L have no tier C (§3.3).
- WebP query images → converted to JPEG first (MOTA §9.4); `.gif` files are out of scope (the index excludes them by design). Tier B transformations (§5.1) are generated from images that have passed through this conversion and only then saved as PNG.
