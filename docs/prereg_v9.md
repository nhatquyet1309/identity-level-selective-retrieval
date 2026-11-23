# prereg_v9 — Experiment P1+P2: the IDENTITY GAP contribution (4-score comparison + gap invariance)

> English translation of the original (Vietnamese) document; content, numbers and codes are unchanged. The sha256 recorded in `prereg_v9.md.sha256` is the hash of the original Vietnamese file, not of this translation.

- **Status: SEALED — 21/11/2026** (the project owner chose **"P1+P2 (+1 day)"** via a question on 21/11; hash in `prereg_v9.md.sha256`).
- **Depends on (not modified):** `prereg_v7.md` SEALED `e5080142…74d1d` · `prereg_v8_addendum.md` SEALED · `decision_log.md` entry R · the tier A fp32 numbers (D8) in `PREREGV7_MASTER_20261119.md`.
- **Question:** H1 identity gap — what does `m_id` (the top-1 − top-2 distance **within the same identity**, Algorithm 1 line 10) contribute relative to absolute similarity / MSP / energy? (Currently: insufficient at n=169 per ab4.)

## 1. P1 — Comparing 4 confidence scores on tier A (large n)

- **Universe:** tier A **A(ii)** fp32 — D(ii) = **1,663 queries**, 145 identities (θ_cos=0.98 de-duplicated), 6 cells but **primary encoder = SigLIP2-I** (v8); **secondary = CLIP-B-I** (reported alongside, not primary). The other 4 cells (S, DINOv2, CLIP-L) are **not run** (out of scope — declared in advance).
- **Split (locked, not re-chosen):** RQ-1 `split_hash = 77a64289…` — **73 identities calib / 72 test**, seed 20261022. T\* and every parameter choice **on calib only**; conclusions **on test only**.
- **4 scores — exactly the ab4/spec_v2 definitions (Sc1–Sc4), unchanged:**
  1. `sim` = absolute similarity (the top-1 identity score);
  2. `MSP` = max softmax probability with **T\* = argmin E-AURC on calib** (T grid identical to ab4: 0.01,0.02,0.05,0.1,0.5,1.0);
  3. `energy` = −E_T (own T\* by the same rule);
  4. `gap` = **m_id** (s₁ − s₂ within Y_c, s₂ = the second-best image **of the same identity**).
- **Data needed:** for each query × each encoder cell = a score vector **per identity** (145) + the top-2 images within the identity → **1 server pass** (tier A LOO, D(ii)).
- **Metrics (identical to ab4):** AURC / E-AURC / AUROC of each score on test · **primary: paired ΔE-AURC (gap − other score)**, cluster bootstrap **2,000 by identity, seed 20261022** · **Holm over 3 comparisons** · coverage@risk(5%) reported descriptively · **2-label conclusion as in ab4: `better` (CI excludes 0) or `insufficient` — NO new δ declared after seeing the numbers** (keeping the ab4 rule exactly).
- **Honesty:** every score × 2 encoders × calib/test is reported in full; if `gap` comes out `insufficient` → **report insufficient** (this is a result, not a failure).

## 2. P2 — Invariance of the gap versus absolute similarity (tier B)

- **Data:** 18,337 transformed queries (11 T × 1,667), **primary encoder SigLIP2-I** (1 server pass, reusing the phase-1 transforms).
- **Measured per (x, T):** Δm_id = m_id(T(x)) − m_id(x) · Δs₁ = s₁(T(x)) − s₁(x) (s₁/s₂ over images in Y_c as in P1).
- **Comparison (descriptive, declared in advance):** per **canonical (6 T)** and **stress (5 T)** group: median |Δm_id| vs median |Δs₁| · P95 · **median ratio** (|Δm_id|/|Δs₁|) + **identity cluster bootstrap 2,000/seed 20261022** for the difference of medians. **No p-value, no "better/worse" label** — conclusions of the form: "the gap shifts much less / equally / more (ratio + CI)".
- **Expectation (declared in advance, not a commitment):** s₁ and s₂ shift together under transforms → the gap may be more stable — **if the ratio CI < 1 → 'more stable'; contains 1 → 'no difference seen'; > 1 → 'less stable'** — all 3 outcomes are reported.

## 3. Output & rules

- Directory `results/prereg_v9_gap_20261121/`: scripts (local copy in `scripts/`), score-capture (server), `p1_results.json` + `report_P1.md`, `p2_results.json` + `report_P2.md`, `SHA256SUMS.txt`.
- **RNG:** state the family explicitly in every output (numpy PCG64 if run on the server — record `default_rng(20261022)`; or stdlib `random.Random` — record it). CI clustered by identity (145/141 clusters — state which).
- **Do not modify** v7/v8/tests/old deviations; new numbers go only into the v9 directory. A deviation is recorded only if the plan is departed from (e.g. data loss).
- **Relation to the paper:** P1 answers H1 directly (a `better`/`insufficient` conclusion for the gap on n=1,663 — stronger than n=169); P2 adds the angle "is the gap stable under transformations". Both are labelled for exactly what they are: a **new experiment (v9), separate from the registered ab4 run**.
