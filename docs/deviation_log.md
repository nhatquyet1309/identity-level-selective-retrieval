# Deviation Log (v5 scope)

> This is an English translation of the original Vietnamese log; numbers use English thousands separators (commas).

Scope note: v5 narrows experimental scope to LNNS LNCS-style paper under user decision before experiments. Changes below are intentional pre-experiment scope decisions, not items that were "never considered".

Removed from v5 scope (were in spec v1):
- B2+ refit branch
- B3 Tip-Adapter cache
- H2, H3 hypotheses
- ΔRecall(N)
- 3 insertion policies
- noisy-oracle epsilon
- dHash as growth component
- confirmation stream (branch)

Impact:
- Only H1 + transfer C3 remain; comparison has 4 score components (Sc1-Sc4).
- Hypotheses and experiments limited accordingly; v5 documents design state, no runs.

Related draft: paper_ares_v5. spec_v1 retained unchanged as history; spec_v2 describes v5 evaluation plan.
---

## 2026-10-21 — Jetson Orin NX is back: first measurement of Block A (cost)

Context: the Jetson is reachable via Tailscale; for the first time there are cost measurements on the **target hardware**.

Deviations / notes relative to the specification (`results_evidence_spec.md` §12.3):

1. **`tegrastats` on this Orin NX does NOT print `VDD_IN`.** Spec §12.3 row A7 says to read
   `tegrastats` — wrong. Fixed: read the **INA3221** at `/sys/class/hwmon/hwmon3`
   (`in1_input` mV × `curr1_input` mA), which **requires root** → `sudo env HOME=<REDACTED_HOME> python3 …`
   (keep `HOME`, otherwise `python3` cannot see `torch` → `ModuleNotFoundError`).
2. **The query set of the cost measurement is the gallery itself (in-domain), not the protocol's
   query set** (Block B has not been run). Therefore these numbers are **for the cost section only**;
   they must never be used as quality numbers. Block A is not fully closed.
3. **No confidence interval for latency.** Report median + p95 per §8.11; CIs are required
   only for quality numbers, not for cost numbers — but a CI also **must not be inferred**
   from median + p95.
4. **Power was measured while 4 <SERVICE> services were running in parallel** → `VDD_IN` includes the background share
   (~5.5 W background). Stated clearly in the report; the whole amount is not attributed to the method.
5. **`visual.index` had to be built on the Jetson** (<GPU_SERVER> does not have this file): 581 images → 581 vectors,
   69 variants, `IndexFlatIP` 512 dimensions. CLIP ViT-B/32 runs **on the GPU** (torch 2.1.0a0+nv23.06).
   Required installing `faiss-cpu` 1.8.0 + `clip` into the user-site of the system `python3` (the venv has
   `torch 2.4.1` **without CUDA**, so it cannot be used for GPU measurement).
6. **The measurement was repeated 2 times** to check repeatability: error < 1 % (A1 20.24/20.21 ms ·
   A2 0.0793/0.0792 ms · A3 28.63/28.75 ms · 34.49/34.29 q/s batch 1).

Outside the paper's scope (measured for the FPV system, recorded in §12.5.1–12.5.2):
- Whole-system concurrent load: YOLO + SLM + VLM + collector on the same machine.
- **NVDEC**: actually measured, conclusion **not used** for the relay stream (CPU decoding takes only 2.06 ms/frame,
  the bottleneck is inference at ~36 ms/frame and the GPU being 95 % occupied by the LLM).

---

## 2026-10-22 — First run of Block B on the holdout set (current gallery)

Context: the user asked to "finish block A" and "send complete results of block A and B".
The gallery is **not yet finalised** → the results below are a **measurement on the current gallery**, not final results.

Design decisions (declared before measuring, fixed by seed 20261022):

1. **Holdout split**: 30 % of each identity is held out as the query set, and each identity always keeps ≥ 1 image
   in the index → index of 427 images, holdout set of 154 images (CALIB 62 / TEST 92).
   Reason: if queries take the very images in the index then top-1 is always correct → **there are no errors
   from which to compute AURC/risk-coverage**; there must be images that are *not* in the index.
2. **`coarse_source = oracle`**: candidates `Y_c` = the identities of the same coarse class (the category
   folder of the gallery). Gallery crops do **not** pass through the YOLO detector, so there is no
   true class label. The result does **not** describe the 2-stage path (detector → coarse → gallery).
3. **Only one query domain** (the gallery). Two-way threshold transfer A→B **cannot be
   evaluated** → no transfer conclusion may be written.
4. **No viewpoint labels** → the paper's "viewpoint shift" framing has **not** been verified
   with numbers; if kept in the title, this limitation must be disclosed.
5. **Stratum `n_c = 2` is empty** (every coarse class has ≥ 2 identities in the index) → the identity
   Sc2 ≡ Sc4 **cannot** be verified on real data (section 8.7); this part can only be
   verified by a self-test on synthetic data.
6. **Choice of T\***: argmin E-AURC on CALIB. With 62 calibration queries, this rule picks a very
   small T (0.01–0.02) — exactly the degenerate region where Sc3→Sc1 and Sc2→Sc4, i.e. the scores coincide.
   This is a real limitation of the current design (the calibration set is too small to choose T stably), it is
   **reported**, and not hidden by choosing T post hoc on the test set.

Implementation bugs found and fixed **before** the final run (via the self-test `--self-test`):

7. **Negative E-AURC**: the optimal-AURC formula was written wrongly (the error count grew in the opposite direction). Fixed to
   `risk(k' = k+1+j) = (1+j)/(k+1+j)`. Verified: E-AURC = 0 under a perfect ranking.
8. **AUROC reversed**: it was returning P(score(error) > score(correct)). Fixed to
   P(score(correct) > score(error)) = 1 − U (higher value = more confident). Verified: AUROC = 1 when every
   correct query has a higher score, 0 when reversed, 0.5 when indistinguishable.
9. **Clopper–Pearson wrong**: the wrong side of the beta function and the wrong bisection direction were used → the upper bound came out as 0.0051
   instead of 0.2589. Fixed; this directly affects the choice of τ_α (all earlier calibration numbers were
   worthless and were re-measured).

Test evidence: `paper_ares/logs/jetson/20261022-ab/selftest.txt` (19/19 PASS).

---

## 2026-10-26 — Recount of the gallery before the benchmark (checklist step 1)

Context: the gallery was expanded (tank/drone) + FAISS rebuilt on <GPU_SERVER> (25/10).
Ran a read-only count on both machines, raw log at
`results/gallery-recount-20261026/recount.md`.

Deviations / notes relative to the specification:

1. **The two galleries have diverged, with different indexes** — <GPU_SERVER>: 1082 images/100 identities
   (`visual.index` ntotal 1076 = 1082 − 6 gif, sha `f33d3f8e…`, rebuilt 25/10);
   Jetson: 557 images/66 identities (`visual.index` ntotal 551, sha `36e25f22…`,
   synced version 22/10). Not yet re-synced — as the user intended ("sync later once the review is satisfactory").
   Consequence: **all old Block A+B numbers (run `jetson-20261022-ab`) are stale**; they must
   not be entered in the paper; a rerun is only valuable after syncing + rebuilding the index
   on the Jetson + verifying byte-identical per the SYNC.md procedure of 22/10.
2. **helicopter in the opposite direction**: the Jetson (172 images) has more than <GPU_SERVER> (165 images) →
   the sync must merge in both directions by content hash; overwriting in one direction is forbidden.
3. **tau-5% feasibility is reversed depending on the machine**: the <GPU_SERVER> gallery gives a holdout of ~283
   queries (calib ~113 ≥ 59 → feasible in sample size); the current Jetson gallery gives a
   holdout of ~140 (calib ~56 → certainly infeasible, same as the old run).
   → After the sync, the final calib size must be recomputed from the merged gallery.
4. **1-image identities**: <GPU_SERVER> still has 5 (`drone/orlan_10`, `drone/shahed_136`,
   `drone/zala_lancet`, `ship/uss_gerald_r_ford`, `tank/t80`); the Jetson still has 9
   (old list in the DRAFT). Adding tank/drone removed most singles.
5. **ship is still 3 images/2 identities on both machines** — holdout_est = 0, this stratum
   contributes no queries; if ship is kept in the paper this must be declared.
6. Parallel static check: `s_y = max` by identity is still correct in
   `ares_deploy_jetson/tools/measure_retrieval_quality.py:16`
   (max aggregation, `n_c < 2` rejected separately) — no implementation change detected.

## 2026-10-26 — Sync gallery <GPU_SERVER> → Jetson + rebuild index (after step 1)

Per the user's order ("sync right away, <GPU_SERVER> is superior in every respect"). Procedure:

1. Back up the old Jetson gallery: tarball `<REDACTED_HOME>/gallery_backup_pre_sync_20261026.tar.gz`
   (283 MB) + keep the directory `visual_gallery_old_20261022` (delete after the A+B rerun is stable).
2. Pack <GPU_SERVER> (`tar czf`, 619 MB) → intermediate → Jetson; unpack 1082 files.
3. Verify rel→sha256 against the <GPU_SERVER> manifest: **MATCH 1082/1082, 0 missing/extra/mismatch**.
4. Rebuild `visual.index` on the Jetson (`tools/build_visual_index.py` + env
   `VISUAL_GALLERY`/`VISUAL_OUT` pointing to `<SERVICE>/data/` — if the env is not set the script
   looks in the wrong place, `ares_deploy_jetson/data/visual_gallery`): **ntotal 1076** = 1082 − 6 gif.
5. Copy the Jetson index+meta → <GPU_SERVER>: **byte-identical on both machines, sha `5bcb31d4…`**
   (different from the sha of the build on <GPU_SERVER>, `f33d3f8e…` — exactly as SYNC.md warned about
   floating-point non-determinism between sm_120 and sm_87; canonical for the paper = the Jetson version).

Consequence: the gallery 1082/100 is now unified; holdout_est ~283 (calib ~113);
the 5 one-image identities + ship with 0 queries stay as recorded at step 1.

## 2026-10-26 — Rerun of Block B on <GPU_SERVER> (user's decision)

The user decided: the paper does not state the edge device explicitly → run the benchmark on <GPU_SERVER>.
Recording the consequences per the rule "do not mix numbers from two devices in one table":

1. New run: `5070-20261026-ab` (out: `<SERVICE>/data/runs/5070-20261026-ab`),
   same protocol + seed `20261022`, frac 0.3, calib-frac 0.4, bootstrap 2000,
   script `tools/measure_retrieval_quality.py` (diff local == the Jetson version of 22/10:
   IDENTICAL). <GPU_SERVER> GPU sm_120, torch 2.14.1+cu130, faiss 1.15.1, CLIP ViT-B/32.
2. The index used for the rerun is the canonical version built on the Jetson (sha `5bcb31d4…`,
   copied byte-identical to <GPU_SERVER>) — but the query embeddings are computed on <GPU_SERVER>;
   the floating-point difference between sm_120/sm_87 is known (max|Δ|≈1.5e-3, SYNC.md 22/10).
3. All numbers of run `jetson-20261022-ab` (DRAFT_RESULTS_A_B.md) are officially stale for the
   paper; kept only as a process reference, not to be entered in the manuscript.

## 2026-10-26 — Results of the Block B rerun `5070-20261026-ab` (gallery 1076/100)

Raw log: the run dir on <GPU_SERVER>
(`<SERVICE>/data/visual_logs/runs/5070-20261026-ab/`:
`predictions.jsonl`, `metrics.json`, `split.json`, `report_B_quality.md`,
`index/`, `holdout/`); copies of `metrics.json` + `report_B_quality.md` +
`split.json` were pulled to `results/5070-20261026-ab/`.

Results (stratum `n_c>=3`, TEST n=169, CALIB n=114):

1. top-1 **52.7 %** (down from 69.4 % in the old run — the gallery grew 551→1076 images,
   66→100 identities so the problem is harder; this is a real number, not a measurement error),
   top-3 73.4 %, `r_same` **37.9 %** (down from 51.8 % — grouping by identity still
   makes sense but is weaker).
2. Main comparison (T\* chosen on CALIB: Sc2 T=0.02, Sc3 T=0.01):
   ΔE-AURC Sc1 +0.0087 CI [−0.048; +0.076] p=0.752;
   Sc2 −0.0005 CI [−0.027; +0.024] p=0.997;
   Sc3 +0.0224 CI [−0.038; +0.092] p=0.473;
   Holm over all 3 = 1.0 → **not enough evidence** (H1 not yet supported — same as the old run).
3. **τ_α is infeasible at both α = 1 % and 5 % for every score** — but the reason has changed:
   in the old run it was because calib was too small (57); this run has calib 114 (enough for the sample size 59) yet there is still no
   threshold reaching the CP bound → the constraint lies in the **error rate**, not the sample size.
4. Exploratory (not a conclusion): at T = 0.1/0.5/1.0, Sc4 is better than Sc2 and Sc3
   with CIs excluding 0 — the direction of the effect **depends on temperature**, while the calib rule
   (argmin E-AURC on a small calib) picks exactly the T region where the scores coincide.
5. Correctness checks: `m_img = min(s1−a, m_id)` 0 violations; stratum `n_c==2` has only 1 query;
   `n_c<2` = 0; the metric self-test 19/19 is retained from the old run (script IDENTICAL).
6. Citation constraints kept from the report: `coarse_source = oracle`,
   a single query domain, no viewpoint labels → no transfer/viewpoint-shift conclusion.

## 2026-10-26 — Results of Block A `5070-20261026-acost` (on <GPU_SERVER>, same holdout as Block B)

Raw log: the run dir on <GPU_SERVER>
(`<SERVICE>/data/visual_logs/runs/5070-20261026-acost/`:
`enc_ms/search_ms/e2e_ms/throughput/ram/gpu_mem/power/startup/summary.json`);
a copy of `summary.json` was pulled to `results/5070-20261026-acost/`.
Script IDENTICAL to the Jetson version; the `run_id` in the summary carries the prefix `jetson-…`, an
artifact of the script — the real device is recorded in `hardware` (NVIDIA RTX 50-series GPU, sm_120,
torch 2.14.1+cu130, faiss 1.15.1), canonical Jetson index (ntotal 1076).

| # | Metric | median | p95 | Notes |
|---|---|---|---|---|
| A1 | Encode CLIP batch 1 (GPU) | **1.93 ms** | 1.98 ms | CPU preprocess 3.71 ms (separate) |
| A2 | FAISS search k=50, 1076 vec | **0.032 ms** | 0.035 ms | `IndexFlatIP` 512 dimensions |
| A3 | E2E image→identity+decision | **5.80 ms** | 11.98 ms | accept 90.8 %; τ_margin 0.02 · τ_score 0.45 |
| A4 | Throughput batch 1 / 8 | **139.3 / 154.7 q/s** | — | 7.18 / 6.46 ms/query, measured over 60 s continuous |
| A5 | RAM peak (VmHWM) | **2602 MB** | — | tegrastats is not available on <GPU_SERVER> → `{}` |
| A6 | GPU: max_allocated / used | **496 / 11845 MB** | — | total 15833 MB (VRAM of <GPU_SERVER>) |
| A7 | Power | **n/a** | — | no INA3221 + no root on <GPU_SERVER> — recorded as n/a, not estimated |
| A8 | Cold start → first result | **2.78 s** | — | import 1.21 s · model load 1.35 s · first query 44 ms |

Compared with the old Jetson Block A (DRAFT version): enc 20.4→1.9 ms, e2e 28.8→5.8 ms,
batch-1 35.4→139.3 q/s — the difference is due to the different GPU (sm_120 vs sm_87), exactly as
SYNC.md warned; **it is forbidden to place these two columns side by side in the same paper table**.
For the cost section "on the target device" per spec §12, it is still up to the user to decide whether to keep any sentence
from this <GPU_SERVER> table or whether the Jetson must be re-measured (especially A7 power).

## 2026-10-26 — Gallery dedup + clean rerun `5070-20261026-ab2` / `-ab2-cost`

Internal dHash check of the gallery (from the sha256 manifest): 1082 files / 1080 unique hashes —
2 pairs that are byte-for-byte identical: `aircraft/f18_hornet/confirmed_014+015.jpg`,
`aircraft/h6/confirmed_001+010.jpg`. Checking `split.json` of run `-ab`:
for the f18 pair both are in the INDEX (harmless); for the h6 pair one is a QUERY and one is in the INDEX
→ exactly 1 query is guaranteed a top-1 hit (+0.35 pp at most — small but it violates the principle).

Handling (following the precedent of SYNC.md 22/10: keep the file with the newest mtime):
deleted `f18_hornet/confirmed_014.jpg` (kept 015) and `h6/confirmed_001.jpg` (kept 010)
on <GPU_SERVER> → gallery **1080 files**; deleted the 2 corresponding files on the Jetson; rebuilt the Jetson
index (**ntotal 1074** = 1080 − 6 gif); copied to <GPU_SERVER> → byte-identical
(sha `5463b35b…`). Rerun with the same protocol/seed → run `5070-20261026-ab2` (+ Block A
`5070-20261026-ab2-cost` on exactly the new holdout); artifacts pulled to
`results/5070-20261026-ab2[/-cost]/`.

Clean rerun results (stratum `n_c>=3`, TEST n=169, CALIB n=114):
top-1 **52.7 %** (identical to run `-ab` to 4 digits), top-3 74.0 %, r_same 37.3 %;
T\* Sc2=0.01/Sc3=0.01; paired Holm over all 3 = 1.0 → **not enough evidence**;
τ_α infeasible for every score/α (reason: error rate, not sample size);
sensitivity T≥0.1 Sc4 is better (exploratory, as in the previous run).
## 2026-10-26 — Implementation BUG: scores not restricted to Yc (fixed + rerun `-ab3`)

While writing the Fig.2 script (recomputing top-1 from `predictions.jsonl`) it was found that:
the `predictions.jsonl` of run `-ab2` gives a recomputed top-1 of **0.592** ≠ the **0.527** reported by the run.
Cause: `measure_retrieval_quality.py` built `s_y` from the **global top-50**
(all categories), and `y1/correct/m_id` were computed on that; `Yc` was used only to count `n_c`.
Per spec §4: every score must be computed **within Yc** (`M(c)`), `m_id` = s1 −
the max competitor in `Yc\{y1}`.

Fixed (local + deploy <GPU_SERVER> + Jetson, self-test still PASS):
full-ranking search (`index.ntotal`, exact — cheap at this size) → max by identity
→ **restricted to Yc**; `m_id=None` when `n_c<2`; `unfiltered_top_identity` =
argmax over the whole gallery + a real `conflict` (previously hardcoded 0).
Rerun with the same protocol/seed → run `5070-20261026-ab3` (replaces `-ab2`;
the `-ab/-ab2` numbers are kept as a process trace, not entered in the paper).
Consequence: the numbers in §3/§4/Fig.2/B-curves below are the old (global) numbers — awaiting `-ab3`,
after which the report + curves are redone (baselines B1/B2 are unchanged because they do not use Yc).

## 2026-10-26 — Rerun `-ab4` with the mapping M approved by the user (artillery/air_defense/missile = vehicle)

Protocol: Yc = the union of the gallery categories under the same YOLO coarse class
(Vehicle → {vehicle, artillery, air_defense, missile}; the rest 1:1).
Artifacts: `results/5070-20261026-ab4/`.

Results (TEST n=169 stratum n_c≥3, CALIB n=114):
the n_c distribution changed completely: {36: 94, 19: 57, 15: 36, 10: 45, 2: 1, 18: 51}
(94 queries fall in the vehicle-group stratum of 36 identities).
top-1 **58.0 %** (vs 59.2 % with same-category Yc — opening Yc barely changes top-1),
r_same 40.8 %; all 3 CIs of the main comparison contain 0, Holm = 1.0 →
**not enough evidence**; τ infeasible for every score/α.
→ The mapping M does not rescue H1, but it is the correct classification decision for E2E-1
(detector coarse → gallery). Fig.2 in the manuscript is still from `-ab3`
(the reference code.py reads the `ab3curves` CSV); whether to build Fig.2 from `-ab4` —
awaits the user's decision (Yc semantics changed → the curves will differ).

## 2026-10-27 — Replacing the CLIP ViT-B/32 core with google/siglip2-so400m-patch14-384

User decision: use SigLIP2 so400m (an external recommendation). Important finding
during the PoC on <GPU_SERVER>:
  * The checkpoint `google/siglip2-so400m-patch14-384` has a stub config
    (`model_type: siglip`, the vision/text sub-configs are complete but the arch label is v1).
    `AutoModel` resolves correctly to **SiglipModel** (not Siglip2Model —
    forcing Siglip2Model causes a patch_embed size-mismatch). The PoC with 4.57 loads + runs OK.
  * PoC <GPU_SERVER> (fp16, <GPU_SERVER>): encode batch1 **17.3 ms/image**, VRAM 2330 MB,
    dim **1152**, self-sim ~1.0; naive zero-shot on 12 images only 0.083 (= chance —
    crude prompts, not a conclusion about the model).
  * The Jetson (py3.8 + torch 2.1 nv23.06) does NOT run stock transformers:
    4.48.x Requires-Python>=3.9 (pip 20.0.2 refuses) + tokenizers>=0.21
    has no cp38 wheel. The solution, already proven by a smoke test:
    vendor transformers==4.48.3 + safetensors==0.5.3 (cp38 aarch64) +
    huggingface_hub==0.25.2 (py>=3.8 OK), patch the python pin 3.9→3.8 in the
    vendored copy, a stub module `tokenizers` (vision-only, every call raises),
    a stub dist-info for `regex` (missing metadata), use only the vision tower
    (SiglipVisionModel + SiglipImageProcessor — the tokenizer is not touched).
    Smoke: import OK, build so400m-shape, encode CUDA **420 ms** (random weights).
  * The Jetson CANNOT download directly from HF (tried: ~1.8 KB/s) → transfer the weights
    <GPU_SERVER>→Win→Jetson (model.safetensors 4.3 GB + config + preprocessor).
    Destination: `<REDACTED_HOME>/models_siglip2/so400m-patch14-384/`.
  * Known limitation: the text tower stays on <GPU_SERVER> (B1 zero-shot + text-fusion);
    the Jetson runs only vision. Changing the encoder = a new arm of the paper (R5b):
    the index must be rebuilt (d=1152), Block B rerun + recalibration — NOT DONE YET,
    waiting for the final proof on the Jetson before the swap into the pipeline is fixed.

## 2026-10-26 — Results of the spec-correct rerun `5070-20261026-ab3` (+ curves/baselines)

Yc-restricted protocol, same seed/frac. Artifacts pulled to
`results/5070-20261026-ab3[/]` +
`5070-20261026-ab3curves/` (fig2 CSV+PNG+meta, baselines.json, prompts.json;
the `.npy` features are kept on <GPU_SERVER> for later use by test_sim_runner).

- Stratum `n_c>=3` TEST n=169: top-1 **59.2 %** (up from 52.7 % global —
  the right direction when restricting Yc), top-3 74.0 %, r_same 37.3 %→**40.8 %**;
  Sc4 E-AURC 0.1179 [0.075; 0.167].
- Main comparison (T\* Sc2=0.01/Sc3=0.01): Sc1 +0.014 CI [−0.053; +0.090] p=0.683;
  Sc2 −0.0016 CI [−0.010; +0.007] p=0.729; Sc3 +0.026 CI [−0.046; +0.108]
  p=0.493; Holm = 1.0 for all 3 → **not enough evidence** (H1 not yet supported).
- τ_α infeasible for every score/α (56 "infeasible" lines in the report) — still due to
  the error rate, not sample size.
- Sensitivity T≥0.1 Sc4 is better (exploratory, as in the earlier runs).
- **The conflict flag is meaningful for the first time**: 40/284 queries have the global top-1 outside
  Yc (`conflict=1`) — material for the §7 ablation (not yet analysed).
- Baselines on the same TEST holdout (n=170): **B1 zero-shot top-1 31.2 %
  [22.2; 41.1], top-3 47.6 %** (3 prompts/viewpoint, closed 100-way);
  **B2 linear probe (train calib n=114) top-1 4.1 % [0.6; 9.0]**
  (train top-1 on calib only 37.7 % — 114 samples/100 classes are not enough to learn; this is a real number,
  showing that retrieval contributes clearly). Baseline AURC is not compared against retrieval.
- Fig.2: `fig2_risk_coverage.png` rendered from the CSV, endpoint risk 0.408 =
  1 − the run's top-1 — consistent, no longer off as in the old global version.

## 2026-10-26 — Block A on the Jetson with the latest index (`jetson-20261026-acost`)

Per the user's order ("measure on the Jetson, use the latest index").
Index: the canonical Jetson build (ntotal **1074**, sha `5463b35b…`);
queries: the holdout of run `-ab2` (284 images) rebuilt on the Jetson from `split.json`
(translating the path prefix <GPU_SERVER>→Jetson, copied 284/284 files).
Ran as root (`sudo`, keeping HOME) to read the INA3221; a copy of `summary.json` was pulled to
`results/jetson-20261026-acost/`.

| # | Metric | median | p95 | Notes |
|---|---|---|---|---|
| A1 | Encode CLIP batch 1 (GPU) | **19.62 ms** | 20.01 ms | CPU preprocess 9.15 ms (separate) |
| A2 | FAISS search k=50, 1074 vec | **0.143 ms** | 0.146 ms | `IndexFlatIP` 512 dimensions |
| A3 | E2E image→identity+decision | **29.85 ms** | 43.94 ms | accept 91 % |
| A4 | Throughput batch 1 / 8 | **31.0 / 64.7 q/s** | — | 32.24 / 15.45 ms/query, measured over 60 s continuous |
| A5 | RAM peak (VmHWM) | **4825 MB** | — | |
| A6 | GPU method max_allocated | **471.8 MB** | — | used shared SoC 13531/15503 MB |
| A7 | Power VDD_IN batch 1 / 8 | **12.26 / 13.37 W** (avg) | 12.59 / 14.40 W | includes the background of the <SERVICE> services; min 11.69 W |
| A8 | Cold start → first result | **9.89 s** | — | import 2.31 s · model 5.30 s · first query 1.07 s |

Compared with the old Jetson Block A (run 22/10, index 402 vec): e2e 28.8→29.9 ms,
batch-1 35.4→31.0 q/s — the same order of magnitude; the difference is due to the larger index
(1074 vs 402 vec) and a different query set. A7 increased (10.16→12.26 W) — includes the background;
the whole amount is not attributed to the method (stated clearly as in the old run).


## RUN deviations — 5070-20261026-ab3 (seed 20261022) — approved 26/10/2026
Deviations between the actual run and spec_v2/the 26/10 edit. Recorded honestly, the spec is not retroactively edited.

| # | Spec | Actual run | Reason | Impact / how recorded in the paper |
|---|---|---|---|---|
| R-D1 | Split by scenario/trajectory (1a/D6) | Stratified holdout of 30% of the images of each identity + calib 40% (identities appear on both sides) | The web gallery has no scenario/trajectory labels | Cluster = identity; there is no per-scenario step. Recorded in Setup |
| R-D2 | Choose T on a separate tuning split, separate from calibration (D4) | T* = argmin E-AURC ON CALIB (T_star_rule in metrics.json, "declared in advance, chosen once") | The run was exactly/just before the 26/10 edit, there is no separate tuning split | State clearly in Results: T* chosen on calibration. Do not call it a tuning split |
| R-D3 | Calibration unit = trajectory, CP on 1 query/cluster (D3) | CP computed on queries (n_calib=114), bootstrap cluster=identity | No trajectory labels | The CP bound is DESCRIPTIVE (query-level). Every threshold is infeasible so no conclusion is affected |
| R-D4 | Bootstrap 10 000 (spec) | 2000 replicates | The current runner | Report exactly 2000 in every CI table |
| R-D5 | Tier-B ablation chain gap → +n_c<2 → +conflict (C3) | Not yet in the run (coarse_source=oracle, the report has no ablation) | Not yet run | Keep [TBD], do not describe results |
| R-D6 | Primary metric coverage@risk(5%) (D7(i)) | metrics.json has only AURC/E-AURC/AUROC | The runner does not compute it yet | [TBD] or derive with a source note from fig2_risk_coverage.csv (ab3curves) — only when it is clearly established that the csv is the test set |
| R-D7 | Cost corresponding to gallery 1074 | Cost Jetson-22/10 measured on index 402 (gallery 551); jetson-20261026-acost needs its ntotal checked | A different measurement session from the quality one | If cost-22/10 is used, "index 402 vectors" must be stated together with the numbers |
| R-D8 | run_id on the correct machine | the summary.json of <GPU_SERVER>-acost records "jetson-20261026-..." | script copy | A traceability note, the result files are not changed |

## P0-4 — Fixing primary H1 = ΔE-AURC + the pre-registered wording rule (Director's decision, 26/10/2026)

Context: two pre-run files phrase the primary differently — `hypotheses.md` (06/10) §2.7 makes AURC/E-AURC primary for H1; `spec_v2.md` (19/10) D7(i) makes coverage@risk(5%) primary. Applied per the Director's decision:

1. **H1 primary = ΔE-AURC** (paired, n_c≥3, Holm over 3 comparisons) — evidence `hypotheses.md` §2.7; matches every comparison that was run + the Abstract. coverage@risk(5%) = the main operational/calibration report (spec_v2, 19/10), in descriptive form (R-D6). The text states both readings.
2. **Both are insufficient evidence on this run** → no selective advantage comes from the interpretation applied (stated clearly in §4 Metrics).
3. **Pre-registered/pre-specified wording rule:** only items traceable to a pre-run file (hypotheses 06/10, spec 08/10, spec_v2 19/10 — see protocol_lock.md) may be called "pre-registered", with the source `(file, date)` given in place; every item from the statistical edit of 26/10 (tie-breaking, secondary, D1–D7, Fig caption) is NOT traceable → call it "pre-specified in this manuscript". The Abstract changes "The study is pre-registered." → "The evaluation protocol was fixed before the runs (hypotheses 6 Oct, specification v2 19 Oct; see protocol_lock) and deviations are logged."

## R-D9 — Mapping M changed between ab3 and ab4 + fixing the main run ab4 (26/10/2026)

1. **Mapping M changed:** run ab3: Yc = the same gallery category (8 classes, n_c hist {2:1, 9:27, 10:45, 13:37, 14:30, 15:36, 18:51, 19:57}); run ab4: Yc = the union of the gallery categories under the same YOLO coarse class (Vehicle -> {vehicle, artillery, air_defense, missile}; the rest 1:1) -> 6 coarse candidate classes, n_c hist {2:1, 10:45, 15:36, 18:51, 19:57, 36:94}.
2. **split.json is byte-identical between ab3/ab4** (same seed 20261022, protocol, gallery) -> the only difference is the coarse class derivation; baselines B1/B2 do not depend on the mapping so their numbers are kept, with only a same-split note added.
3. **Authorisation:** the user fixed run ab4 as the main run on 26/10/2026; every number in manuscript v5 is read from results/5070-20261026-ab4/{metrics.json, report_B_quality.md}.
4. **ab4 curves are not yet on the machine** (the ab4 dir has only metrics/report/split, no predictions.jsonl or curves CSV) -> Fig.2 in the manuscript is a placeholder [TBD: risk-coverage curves of run 5070-20261026-ab4 — pending curves CSV], the ab3 PNG is NOT used (to avoid mixing runs); the files figures/fig2_risk_coverage*.png/pdf are kept unchanged as history.
## 2026-10-26 — VLM crawl of a new test set `sample_test` (user request, STOPPED)

Purpose: a query set independent of the gallery to measure recognition accuracy (filling the
still-empty Test-Real slot — but they are same-domain web images, NOT a UAV viewpoint shift; this will be
recorded as such, not called sim-to-real).

Protocol (declared in advance):
  * Quota 5 images/identity × 100 gallery identities (= 500 images).
    Output: `<SERVICE>/data/visual_logs/sample_test/<variant>/` (placed under
    visual_logs because `data/` belongs to uid 1002 and the user <USER> cannot create new directories).
  * Spec reused from the existing crawl module (tank/drone NEW+TOPUP+LABELS+EXIST;
    airdefense/artillery NEW+INSERT+CATEGORIES+EXIST_ALIASES;
    aircraft IDENTS) + a generic fallback (34 identities: old heli/ship/vehicle/aircraft).
  * 6 gates/candidate: sha-gallery → dHash hamming>10 → CLIP cosine<0.98
    (canonical Jetson index) → VLM P_IDENT (alias+exclude) → QUALITY GOOD →
    VERIFY2 YES (Qwen3-VL-8B :8003).
  * JSONL manifest (title/query/license/artist/sha/dhash/cosine/vlm/view/verify2);
    idempotent (skip titles already processed + identities that already have 5); VLM dead → wait 60s ×30.
  * Script: `tools/crawl_sample_test.py` (new). NO indexing, the gallery is NOT touched.
  * Operational incident 26/10: the plink launch commands timed out because a background job
    held the stdin of the remote shell session → the worker was thought dead, launched again several times →
    at one point 2 workers ran at once (increased 429s + duplicate VLM calls). The old PID was killed
    (357518, fast-rate code), keeping 1 new worker (gentle rate: search pause 4s,
    MAX_CAND 300). Lesson: launching in the background via plink must use `</dev/null`
    + `setsid`; verify with `ps` rather than guessing from a timeout.
  * After enough have been collected: run CLIP to check the top score (per the user's order: via the Visual Lab web).
  * Not circular: the Qwen VLM serves only as the crawl judge, it is not in the measured pipeline
    (YOLO→CLIP+FAISS).
  * Diagnosis of low yield (user: "the results are terrible", 26/10): dissecting the manifest of 1117 rows
    → the gates are CORRECT (the pool is all S-75/S-200/Buk/Strela/S-300 lookalikes of Kub; VERIFY2 correctly catches
    the VLM1 hallucinations Tunguska≠Kub, Arrow2≠Arrow3, S-400≠S-300).
    The problem is the POOL: generic queries hit junk Commons disambiguation pages
    (h6 → cake/mortar/map/moto; su34 → Su-27/35; f15 → F-16/model).
    Round 2 has been prepared: `ROUND2_EXTRA` (specific operator/museum queries +
    probed categories: Xian H-6, Sukhoi Su-34, F-15 Eagle, Sukhoi Su-25...),
    activated by `SAMPLE_ROUND2=1` after round 1 finishes sweeping.
    The rate is already gentle (pause 4s, MAX_CAND 300) after the 429 storm.

## 2026-10-26 — Top score check via the Visual Lab web (user asked to stop the crawl)

User decision: stop the sample_test crawl, test with the gallery images themselves via the web.
Ran `tools/web_topcheck.py` (new): POST each gallery image to
`POST /api/classify` (category=auto, the real deploy path: CLIP+FAISS+text fusion
w_img=0.5), comparing top-1 with the folder label. Run `web-topcheck-v1`:
**resubstitution acc = 0.5344 (miss 501/1076)** — abnormally low.
Direct control (bypassing the web, `tools/direct_topcheck.py` new):
CLIP + canonical index → **acc = 1.0000 (0 miss/1076)**.
Conclusion: the index + embeddings are perfect; all 501 misses are at the web layer —
(1) a stale index in the web's RAM (the uvicorn process has been running since 22/10, before the rebuild of
25–26/10) and/or (2) text-fusion flipping a correct image-sim result.
Artifacts of both runs were pulled to `results/web-topcheck-v1/` and
`direct-topcheck-v1/` (jsonl + report.json + run.log).


(Supplement: the Block A <GPU_SERVER> note that was overwritten by mistake in the previous edit — restored verbatim:)
Block A rerun (`-ab2-cost`, <GPU_SERVER>): enc 1.96 ms · search 0.032 ms ·
e2e 5.89 ms (accept 88 %) · batch-1 138.7 / batch-8 154.8 q/s · RAM 2595 MB ·
A7 n/a — matches the previous run (error <2 %), showing that the measurement is repeatable on the same machine.
(Block A does not depend on scoring so it does not need to be rerun after the Yc fix.)

## 2026-10-27 — SigLIP2-so400m on the <GPU_SERVER> web (user: replace the CLIP core to test on the web)

User decision: replace CLIP ViT-B/32 with google/siglip2-so400m-patch14-384
on the Visual Lab web (without touching the Jetson for now).
Implementation (keeping the CLIP path unchanged as the default):
  * `tools/siglip_engine.py` (new): an adapter with the CLIP surface (preprocess/encode_image/
    encode_text/tokenize, fp16 cuda) + `tools/build_siglip_index.py` (new):
    a SigLIP index of **1348 vectors d=1152** + meta with the same schema (`visual_siglip.*`,
    the canonical CLIP files are NOT touched).
  * `classify_visual.classify`: the text-fusion branch per engine (the text-emb cache is
    keyed by engine, to avoid mixing CLIP/SigLIP) + `visual_lab.get_engine`: env
    `VISUAL_ENCODER=siglip2` → load SigLIP (default clip).
  * The web runs under systemd `ares-vlab`; used `systemctl set-environment`
    (transient — a reboot reverts to CLIP) + restart. The old uvicorn had been running since 22/10
    (stale RAM index) → this restart also cures the staleness.
Test results on the web (`/api/classify`, category=auto):
  * Smoke (4 images): t72/f16/ins_vikrant/ababil all CORRECT (old CLIP wrong on 2/4).
  * Full resubstitution `web-topcheck-siglip-v1` (1355 images): acc **0.9432**,
    miss 77 — the confused pairs are all genuinely hard (t90↔t72 ×11, il76↔a50 ×5, b1b↔tu160...).
  * Web-bypass control `direct-siglip-v1` (image-only): acc **0.9840**
    (22/1379 miss, all hard pairs: t55→m60, m1→challenger1, stryker→bradley...).
    → text-fusion flips only a small part; the misses are mainly = images the user added AFTER
    building the index (the gallery is a moving target: 1082 → ~1379 files in 2 days).
  * Artifacts: `results/web-topcheck-siglip-v1/`, `results/direct-siglip-v1/`;
    Fig.2/checklist: Fig.2 is currently still CLIP (run -ab3); the Block B rerun with
    SigLIP (a real comparison vs CLIP 59.2 %) WAITS for the gallery to be frozen — not run
    while the gallery is still gaining ~100 files/day.

## 2026-10-27 — Fixing the web UI per encoder (user: "haven't you fixed the web UI yet")

The web hardcoded "CLIP ViT-B/32" in the subtitle/description/pill/section. Made dynamic:
`index()` replaces the strings according to the running `VISUAL_ENCODER` (SigLIP2-so400m (1152-d)
+ "recognition with SigLIP2 + FAISS" + "Cross-check of SigLIP2 and VLM" + "SigLIP top-1").
Deployed + restarted ares-vlab, curl confirmed the new text, smoke classify still correct
(t72/f16/ins_vikrant). The public zrok URL serves the same web so it is visible immediately.

---

# prereg_v7 — Tier A (smoke) — 2 batches, 17–18/11/2026

Run dir: `results/preregv7_tierA_20261117/` (batch 1: `tierA_smoke.py`,
batch 2: `scripts/tierA_v2.py` + `tierA_v2_probe.py` + `tierA_v2_w50check.py` +
`make_env_txt.py`). Machine: <GPU_SERVER>, venv `<PYTHON_ENV>`. Env + hashes of the 4 indexes
at the time of measurement: `results/preregv7_tierA_20261117/env.txt`.

Batch 1 **stopped correctly at gate (a)** of §4.0 (CLIP-B 1666/1667). The Director made 3 decisions
(recorded below as D1, D2, and the instruction to run (d) before/after (c)). **The SEALED prereg is not
modified** — every item below is a deviation of the *eval script* or a data finding.

## D1 — (decision iii) L2 normalisation at eval time for ALL 4 INDEXES — 17/11/2026

**Root cause (not random noise).** `build_visual_index.py` uses
`clip.load("ViT-B/32")`; `encode_image` returns **float16**, and the build L2-normalizes **in
fp16** and only then casts to float32 when adding to FAISS. Therefore the stored vectors are **not unit**:
measured on 1,667 vectors — norm min **0.9994866** · max **1.0004998**, **1,620/1,667**
vectors deviate from unit by > 1e-5. `IndexFlatIP` returns `IP = ‖s‖·cos`, so **IP ≠ cosine**,
exactly as feared. The failing case of batch 1: the near-duplicate pair `helicopter/ah6_little_bird`
(`…_f73a5bd7` vs `…_3aa1b71d`) has `‖s_other‖ = 1.00014` → `IP(other) = 0.999881 >
IP(self) = 0.999502` even though `cos = 0.9999902`; the residual is **not** a meta-matching bug.

**Handling (as decided).** Renorm L2 **only in the eval script, only in
RAM**: renorm both the query (re-embedded from file, batch 1 cache) and the stored vectors, then search
with an `IndexFlatIP` **built in RAM** from the renormed copy (exact — still `IndexFlat`, no
IVF/PQ). **The index file is not written back.**

**Result (a) after renorm — 4/4 PASS @1e-5:**

| encoder | batch 1 (IP, fp16 build) | batch 2 (renorm eval) | max_gap batch 2 |
|---|---|---|---|
| siglip2 | 1667/1667 · max_gap 1.19e-07 | **1667/1667** | 1.19e-07 |
| clip | **1666/1667** · max_gap 3.79e-04 (STOP) | **1667/1667** | **1.79e-07** |
| dinov2 | 1667/1667 · 5.96e-08 | **1667/1667** | 5.96e-08 |
| clipl14 | 1667/1667 · 5.96e-08 | **1667/1667** | 5.96e-08 |

- **The tolerance stays 1e-5** (not changed to 1e-3): the index is fp32 per
  `env_capture.txt`; after renorm `self_cos` min = 0.99999964 (maxdev 4.8e-07) so
  1e-5 is still about 20× wider than the real error. The batch 1 figure (max_gap 2.4e-07) was reproduced
  independently at the diagnostic step.
- **The hashes of the 4 indexes before/after did NOT change**, matching `gallery_hash_20261116.txt`:
  `6377a376…afef` (siglip) · `3a9fa459…7226` (visual) · `0b8ae986…4518` (dinov2) ·
  `a8f2c98c…9731` (clipl14). Checked 2 times (start/end of stage a) + once after all of batch 2.
- **Secondary effect measured:** case S of clip changes 768→**767**/1667 in the reference
  LOO version (exactly 1 image changes because a score tie is broken in the other direction); case S self-retrieval
  identity top-1 **does not change** (siglip2 1525/1667, clip 1626/1667).
- **Scope:** the renorm belongs to the *eval script*, it is **not** a system change —
  production still runs IP on the stored vectors as before. Hence the numbers here are **true
  cosine numbers per §2**, not numbers of the production path; the difference between the two paths is
  ≤ 5e-4 on the image score.

## D2 — (decision b) Encoders without a text tower: STRUCTURAL + DETERMINISTIC evidence

Applies to DINOv2 and CLIP-L (the 2 "S ≡ I" cells of §3.1). The two flags are **not** compared empirically because
`use_text=True` cannot exist for these 2 encoders; instead:

1. **Structure.** `visual_lab.py:594` → `use_text=(_enc not in ("dinov2","clipl14"))`
   ⇒ **always False** for these 2 encoders, so cell S and cell I are **the same call**.
   The adapter is `_HFAdapter` (`visual_lab.py:494`) — it has `encode_image`, **no**
   `encode_text` (`hasattr = False`). Calling `classify(..., use_text=True)` directly in a
   **clean process** (empty `_TEXT_EMB` cache) →
   `AttributeError: '_HFAdapter' object has no attribute 'encode_text'`
   (identical for both dinov2 and clipl14).
   *Side note (found during self-checking):* if run in a process that **has already** loaded CLIP-B, the
   module-level text cache `_TEXT_EMB` (key `("clip", tuple(variants))`) returns
   a **512-d** embedding, and the call dies at `ValueError: matmul … size 512 is different
   from 768` **instead of** AttributeError. The same conclusion (cannot run), a different
   message — so the primary evidence was taken in a clean process
   (`tierA_v2_probe.py`, writing `bitexact_probe_fresh.json`).
2. **Deterministic.** Ran the pure path (`use_text=False`) 2 times on **1,667 queries**, hashing
   with SHA-256 the whole result array (path + variant|category + `float.hex()` score,
   top-5): dinov2 `f6e5c3bc…ce1fcb` ×2 · clipl14 `f39a89df…cc56f4` ×2 → **identical**.
   Cell S under the exact production expression uses the same call so its hash **equals** that of cell I.
3. **The flag is not dead (sanity).** 50 images with clip/siglip: `use_text` False vs True give
   **different** hashes (`differ=True`) ⇒ the flag takes effect where a text tower exists, so the "S≡I"
   of DINOv2/CLIP-L is due to structure, not to the flag being ignored.

**How to record in the paper/report:** "an empirical comparison of the 2 flags is NOT feasible for encoders
lacking a text tower — replaced by structural + deterministic evidence".

## D3 — 145 identities (meta) vs 151 (directories) — FINDING, not yet decided

1. `find … -mindepth 2 -maxdepth 2 -type d` (excluding `_quarantine`, `_review`) = **151**
   identity directories. Of these, **6 directories are EMPTY (0 image files)**:
   `artillery/2s4_tyulpan` · `drone/fpv_kamikaze_aze` · `helicopter/aw249` ·
   `tank/t72b3` · `vehicle/fmtv` · `vehicle/fuchs`.
2. The **145** directories with images = exactly **145 unique variants** across all 4 `*_meta.pkl`;
   1,667 images match 1,667 vectors in each index; `meta ∖ folder = ∅`; no discrepancy in the number of images
   per identity; there are **no** duplicate variant names across 2 categories.
3. ⇒ "**151**" in §4.1/§4.6 is the number of **directories**; the real evaluation universe is **145
   identities**. The prereg is already SEALED → **not modified**, a deviation is recorded + the Director is informed.
4. Computed consequence: **4 one-image identities** (`btr80`, `ec130`, `switchblade_600`,
   `z19`) → **D(i) = 1663** images (4 images outside D); strata **30 ids <8 images · 79 ids 8–15 ·
   36 ids 16–20** (total 145).
5. **The §4.6 bootstrap (2,000 resamples, seed 20261022) runs on 145 clusters**, not 151
   — the 6 empty directories have no images so they cannot be clusters. State this clearly in every CI table.

## D4 — Run order: (a) → (d)(e) → (c); the full (c) is still pending

§9.3 orders (a) → (b) → (c) → (d) → (e). In practice: batch 1 ran (a) and then **stopped at gate (a)**
(did not run (b)(c)(d)(e)); batch 2 ran (a)-renorm → (b) → the identity investigation (part
of (c): counting the universe) → (d) → (e). **The full near-duplicate audit of (c)** (pHash
Hamming ≤ 4 + cosine ≥ 0.98, union-find, component distribution) has **NOT been run** — even though
`imagehash 4.3.2` has been on <GPU_SERVER> since after batch 1.

The reported numbers are not affected: (e) reports **A(i)** — which removes only the image itself and does **not** use
duplicate merging; A(ii)/A(ii-s) and the tier B baseline **are still waiting for (c)**.

## D5 — ref_814 (81.4 %) matches CASE I, NOT case S — finding, awaiting direction

`ref_814.txt` has only a rounded number (81.4 %) and does **not** state whether it is image-level or
identity-level. Result (d) — A(i), D(i)=1663:

| SigLIP2 cell | identity-level | image-level | all-1667 | deviation vs 81.4 |
|---|---|---|---|---|
| **S** (`use_text=True, w_img=0.5`) | 999/1663 = **60.07 %** | 999/1663 = 60.07 % | 999/1667 = 59.93 % | **−21.33 pp** |
| **I** (`use_text=False`) | 1357/1663 = **81.60 %** | 1357/1663 = 81.60 % | **1357/1667 = 81.4037 %** | **+0.20 pp** (D(i)) / **+0.004 pp** (1667) |

1. **The two interpretations, image-level vs identity-level, do NOT differ** for A(i): because aggregating
   with `max` by identity makes the top-1 identity exactly the identity of the image with the highest score
   ⇒ the two ways of counting coincide (999 and 1357 in both ways). What decides is **S or I**.
2. **81.4 % matches case I**, closest at the denominator **1,667** (81.4037 %, deviation 0.004 pp);
   it also meets the ±0.5 pp threshold at D(i) (81.60 %). Case S deviates by **−21.3 pp**.
3. **Cross-check of the 2 remaining numbers of MOTA §3** (same run, same renorm, case I):
   DINOv2 **57.05 %** (MOTA 57.0 %) · CLIP-L **69.17 %** (MOTA 69.2 %) — all three
   published numbers match **case I, denominator 1,667** within ±0.05 pp. So MOTA §3 measured the **pure
   image path**, not the blend configuration running in production.
4. **Nothing is decided yet.** Potential consequences (need the Director): (i) the "81.4 %" in the paper must
   be stated clearly as **image-only LOO**, not configuration S; (ii) the S↔I gap of ~21 pp on this
   gallery is a larger finding than anticipated by §4.7 (w_img=0.5 not optimised) and
   **must not** be used to optimise w_img post hoc (a violation of R2).

## D6 — The two groups of encoder meta differ in IMAGE ORDER (147/1667) — 18/11/2026 (batch 3)

**Found while building the canonical matrix for A(ii).** `visual_siglip_meta.pkl` and
`visual_meta.pkl` have the **same** image order; `visual_dinov2_meta.pkl` and
`visual_clipl14_meta.pkl` have a **different** order from those two files at **147/1667 positions**.

Measured, not speculated:

1. **The same set of images** (`set(path)` equal, 1,667 elements) and an **identical identity sequence
   at every position** (`[m["variant"]]` of the 4 metas coincide completely). Example of a differing position:
   `air_defense/s300/s300_001.png` (siglip2) ↔ `air_defense/s300/confirmed_019.jpg`
   (dinov2) — two **different** images, the same identity `s300`.
2. **Why batch 2 still has the right numbers:** batch 2 used the siglip2 `meta` for all 6 cells. For `dinov2`/`clipl14`,
   row i is a different image but of the **same identity**, so top-1 compared with `own[i]` is still a comparison with the
   **correct identity of that image**; counted over all images ⇒ k/N, macro, 8 categories, 3 strata
   are **invariant**. Re-measured in batch 3 by computing **exactly the batch 2 way** (no permutation) and comparing with the
   permuted version: **0/1667 rows** have a different `correct` flag, for **both** dinov2 and clipl14. ⇒ the batch 2 `loo_Ai.csv`
   has **no** value drift; the numbers already reported need no correction.
3. **Why it still has to be handled:** the exclusion masks of A(ii)/A(ii-s)/tier B exclude by the **component of the
   specific image** (not by identity). If the encoder order is kept, the mask points to the **wrong image**
   (excluding an image of another identity / missing the twin image of x itself). This is a **silent** bug,
   the result still produces numbers, so it must be recorded as a deviation.

**Handling (batch 3).** Build the canonical order = `path-sorted` of the siglip2 meta; for each
encoder compute `perm[i] = the position of canonical image i in that encoder's meta`, then permute **both
rows and columns**: `M_can[i,j] = M_enc[perm[i], perm[j]]` (the score depends on both query and candidate).
`perm` is printed in `stage_ii.log` / `_meta.order_check` of `loo_summary_Aii.json`.

**Verification that the permutation does not change the result:** A(i) recomputed in the same script (after the permutation)
coincides **exactly** with the batch 2 numbers in all 6 cells — k/N, N, macro, 8 categories, 3 strata
(`_meta.Ai_verify_vs_loo_summary`). The hashes of the 4 indexes match the pins in `gallery_hash_20261116.txt`
(`6377a376…` · `3a9fa459…` · `0b8ae986…` · `a8f2c98c…`).

**Scope of impact:** only the **row ↔ image assignment** for the 2 encoders `dinov2`/`clipl14`.
Not affecting (a) self-retrieval (query and index have the same order), (b), (d), (e)/A(i).
Applied to: A(ii), A(ii-s) (and will be applied to tier B, RQ-1).

## D7 — Check for components > 20 images: NOT triggered; plus D(ii) = D(i) — 18/11/2026 (batch 3)

§4.1 requires: if any component has **> 20 images** (exceeding the maximum n_c) then record a deviation (not an
error), keep the declared merge, and **additionally report a sensitivity using only pHash edges**.

**Check result (measured, `audit_dedup.json`):**

1. **No component has > 20 images at either threshold.** θ=0.98: 1,596 components, **max 17**
   (`p8_poseidon`, n_c = 20), P99 = 2. θ=0.95: 1,354 components, **max 18** (the same identity
   `p8_poseidon`), P99 = 5. ⇒ **D7 is not triggered**; the merge stays as declared.
2. **The pHash-edge-only sensitivity is still reported** (per the letter of §4.1, though it was not required to be triggered):
   **1,652 components, max 2, 15 pairs / 30 images**. This means most of the merging comes from cosine edges
   (of 71 merges, 3 are from pHash, 11 from pairs identical on both, 57 cosine only).
3. **Denominator consequence (recorded because §4.2 says D(ii) ⊆ D(i)):** 118 images @0.98 (465 @0.95) lie in
   components of ≥ 2, but **no identity loses all its images** after removing the components
   (`newly_out = 0`, `identities_fully_excluded = []`). Therefore **D(ii) = D(ii-s) = D(i) = 1,663**
   images; outside D there remain only the **4 one-image identities**. Methodological consequence: the difference (i) − (ii) is measured on the **same
   denominator**, so the difference is due to **k** and not to a change of denominator.
4. **Data note (not a deviation, needs the Director):** the only multi-label component
   @0.98 is `tank/leopard_2/leopard_2_6d43d6c8.jpg` ↔ `tank/type_96/confirmed_011.jpg` —
   **different bytes** (`6d43d6c8…` vs `9ba61590…`) but **identical pHash**
   (`ea94b9a7542e81f8`, Hamming 0) and cosine ≥ 0.98. §4.1 keeps the merge; fixing the labels/gallery
   is **not** within the scope of this batch.

## D8 — Root fix of the fp16 normalize + rebuild `visual.index` in fp32 + rerun of the 2 CLIP cells — 18/11/2026 (batch Z)

Director's decision 18/11. Batch X died before the final write; batch Z records it in full here. **The SEALED prereg is not modified.** D8 is an implementation bug fix on the BUILD path (not a protocol change), so the old numbers are preserved and replaced by numbers consistent with every tier now running.

**(i) Root cause and patch.** `tools/build_visual_index.py` (L38–66) uses `clip.load("ViT-B/32")`, and `encode_image` returns **float16**; the build L2-normalizes **in fp16** and only then casts to float32 on `add` to FAISS ⇒ the stored vectors are **not unit** (measured norm 0.99949–1.00050; 1,620/1,667 deviate by > 1e-5). `IndexFlatIP` returns `IP = ‖s‖·cos ≠ cos` — exactly the cause recorded in D1 (batch 1 stopped at gate (a), CLIP 1666/1667). D1 was only a temporary patch in the eval script (RAM renorm); **D8 patches at the root**: `model.float()` before encode and `e.float()` before normalize (`_patched_build_visual_index.py` locally; the version deployed on the server). Production does not change the eval renorm path.

**(ii) Rebuild 09:39** with `prep/rebuild_visual_fp32.py` (dry, atomic `.tmp`+`os.replace`). Stats: n=1667 · dim=512 · norm min 0.999999896 · max 1.000000134 · **max|norm−1| = 1.335e-07** (tol 1e-6) · `meta_unchanged=true` (meta count 1667). The new `visual.index` has sha **`31b8f9d2…8dc0`**; backup `.pre_fp32_20261118` sha **`3a9fa459…7226`** (the old pinned version, still intact); `visual_meta.pkl` is **unchanged** `a7fe1b94…0e52`; the 3 other indexes match the pins `6377a376…` (siglip) · `0b8ae986…` (dinov2) · `a8f2c98c…` (clipl14). **`gallery_hash_20261116.txt` is NOT modified** — the pinned version is still traceable through the backup; the hash file is a historical pin ledger, kept unchanged for audit. Evidence: `rebuild_visual_fp32.json`/`.log`.

**(iii) Verify a–e** (`/tmp/prereg_v7/verify_fp32.json` + `verify_e_refined.json`; copies in `results/preregv7_tierA_20261117/`):
- (a) fp32 norm: max|dev| = 1.335e-07 ≤ 1e-6 → **PASS**; 0 rows deviate by > 1e-6.
- (b) hash: `meta_same` · `new≠pin` · `backup==pin` · the 3 other indexes `match_pin` → **PASS**.
- (c) fp32 self-retrieval (query re-embedded with `model.float()`, **no renorm**) **1667/1667, max_gap = 0** → **PASS** (compare the batch 1 fp16 version at 1666/1667).
- (d) Δscore vs the old version: max 1.3e-3 · mean 1.7e-4 · `n_top1_identity_changed = 0` (1 top1-path changed within the near-duplicate pair `ah6_little_bird`, same identity → harmless).
- (e) **correct pipeline** (`verify_e_refined`, query = the **standard** fp16 `clip.load` + D1 renorm in float64 — keeping the path that produced the old numbers): `control_old` (run on the old index/backup) **0 flips (802/802 A(i), 766/766 A(ii))** ⇒ the pipeline does not change the old numbers; `rebuild_new` A(i) **5 flips → k = 801 (−1)**, A(ii) **5 flips → k = 765 (−1)**, each cell has 5 new flips relative to the control, control-only = 0.
- **The first verification pass (43–45 flips) IS AN ARTIFACT of a WRONG query pipeline — NOT cited.** `control_old = 0 flips` itself proves it: when the standard fp16 query + D1 renorm is used on the old index the old numbers are reproduced exactly; only the rebuild causes 5 flips. Every conclusion in (iii) is taken from `verify_e_refined`, not from the first pass.
- Real impact: **5 rows / −1 correct in each cell (−0.06 pp)** — small, explained (5 images near the threshold change top-1 due to fp16→fp32), does not change the encoder ranking.

**(iv) Instruction.** Rerun the **2 CLIP cells (`clip_S`, `clip_I`)** on the fp32 index for consistency with tier B/C, which run fp32 embeddings. The other 4 cells (`siglip2_S/I`, `dinov2`, `clipl14`) use 3 unchanged indexes → must be **bit-identical** (assert before/after). The old numbers (A(i)/A(ii)/A(ii-s), self-retrieval, ref814) are **archived** as `*.pre_fp32.*` — **not lost**, kept as a process trace. Unchanged: seed 20261022 · 145 identities · 2,000 bootstrap · Wilson · θ=0.98/0.95 · pHash≤4. §9/§13 of `report.md` state the new index + backup + D8 clearly.

## D4b — RQ-1 + §4.7 run in parallel with tier B phase 2 — 19/11/2026

RQ-1 + §4.7 ran in parallel with tier B phase 2 (order to add manpower 19/11) instead of after it — a deviation from the order of §9.3 (B→RQ-1). The RQ-1 data depends only on tier A (already closed), not on B → the results are not affected; recorded to audit the order. Output: `results/rq1_20261119/`; `results/preregv7_tierB_20261118/` was not touched.

## D9 — Tier B phase 2: the query script did NOT mask duplicate components → flip/Δ/verdict invalid — 19/11/2026

**Finding.** Ran `tierB_phase2_run.py` (sha `9ddba81e…9073`) on the D8 fp32 index (`visual.index` `31b8f9d2…8dc0`, unchanged before and after) for the full 110,022 rows / 66 pairs × 1,667 — all the count self-checks PASS. But `compute_cell()` computes `Simg = Xq_r @ Xs_r.T` and then `identity_top1_with_path(S, meta, ids)` over the **entire gallery**, **without the masking step** for x and the duplicate component of x that §5.2 + §4.2 (A(ii)) require.

**Measured evidence** (`tierB_top1.csv`, `siglip2_S`/`G1`): **1,483/1,667** rows have `top1_path == src_path` (the query image itself), **1,497/1,667** have a `top1_path` in the same θ=0.98 component. Consequence: the mean `correct` is ~89%, the false positive Δ relative to A(ii) is **+18.76 … +53.98 pp** in 12 cell × group combinations, and flips are mainly "regained" cases (for example `siglip2_S/G1`: 495 regained / 0 lost). The true A(ii) does exactly what `tierA_v3.ii_top1` does (`sub[:, mask] = -np.inf` over the whole component) — there is no equivalent logic in the phase 2 script.

**Handling.** The script is not fixed on my own (a prohibition in the task assignment), the prereg/tests are not modified. The flip table (§5.3.2), Δ+CI+verdict (§5.3.1/§5.4) in `results/preregv7_tierB_20261118/report.md` are marked **INVALID** and may not be cited in the paper. Only self-ret@1 (§5.3.3b) is **valid** because that item deliberately queries the full index (x still in the index) with pure image cosine; cell S reuses the `top1_path` of cell I (the cell S top-1 is the 0.5/0.5 blend). Self-ret@1 figures: siglip2 99.65% · CLIP-B 95.14% · dinov2 95.34% · CLIP-L/14 99.12% (pooling 11 transformations, 18,337 pairs/cell, bootstrap CI 2,000/seed 20261022).

**The Director must decide.** The phase 2 query must be rerun with the same A(ii) mask (masking the columns of every image in the same θ=0.98 component as the query's src before max-pooling by identity), generate a replacement CSV, then recompute flip/Δ/verdict. This is an implementation bug fix, not a change of protocol/prereg — similar to D8. No correct script has been approved in this batch yet.

**Minor note (not an error):** D(ii) after filtering `in_D_Aii=1` is **1,663 images / 141 identities** (4 one-image identities fall out of D(ii)); the task said "145", which is the total identities of D(i). The actual number of bootstrap clusters for Δ is 141; for self-ret (whole gallery) it is 145.


## D10 — Tier C case I: ab4 is image-only; universe = the ab4 split; §6.3 replaced by §6.4 (SigLIP2-I vs CLIP-B-I) — 19/11/2026

**Scope.** Record and correct the labels/report for tier C (case I). **The SEALED prereg is not modified** (every adjustment is recorded here). The old ab4 run is not touched, other batches' files are not touched.

**(i) ab4 is case I (image-only), not "ab4-S" in the text-fusion sense.**
- **Code evidence:** `tools/measure_retrieval_quality.py` (the ab4 entrypoint, `<REDACTED_PATH>/tools/measure_retrieval_quality.py`) has **no** parameter/variable `use_text`, `w_img`, loads no text encoder/prompt, and mixes no text score. The model is hardcoded as `clip.load("ViT-B/32")`, `encoder_id="clip-vit-b32"`. The score `s_y(q)` is the **max image-image cosine** per identity in `Y_c` (CLIP ViT-B/32 image features), and Sc1..Sc4 are computed from `s_y`/the pure image margin (see `measure_retrieval_quality.py`: build image embeddings, FAISS IndexFlatIP, `s_y` = max sim by identity in `Y_c`, no text branch). The ab4 run `5070-20261026-ab4` (`metrics.json`/`predictions.jsonl`/`split.json`, 790 index / 284 queries, test `n_c>=3` n=169, top-1 98/169 = 58.0%) matches the protocol/output of this runner.
- **Label adjustment:** change the cell label **"ab4-S"** to **"ab4 (case I)"** in every tier C table/report. §6.2 of the prereg was written in advance based on the assumption of a web-blend (text-fusion) — **wrong in practice** relative to the ab4 runner/run; recorded here (prereg not modified).

**(ii) The tier C universe = the saved ab4 split (paired), not `snapshot_S2`.**
- Use the ab4 **saved split itself**: `runs/5070-20261026-ab4/split.json` — **790 index / 284 queries**, test `n_c>=3` **n=169 (the same sample)** — instead of `snapshot_S2`.
- **Reason:** the paired comparison (McNemar) with ab4 **98/169** requires the **same query/test set**; `snapshot_S2` (1,483/133) is a different set/gallery (different size/identities/split, see recon plan C) so it cannot be paired directly. Every tier C case I run (CLIP-B-I, SigLIP2-I) loads the split from the ab4 `split.json` (`--split-from`) to keep the index/query/calib/test unchanged.

**(iii) §6.3 (CLIP-B S vs I) is infeasible at tier C → replaced by §6.4 (SigLIP2-I vs CLIP-B-I).**
- **Infeasible:** tier C (per the ab4 split, image-only runner) **has no S cell** (there is no text-fusion in this run/protocol), so the §6.3 comparison "CLIP-B S vs I" cannot be performed at tier C.
- **Replacement (the main tier C paired test):** §6.4 compares the pair **SigLIP2-I vs CLIP-B-I** on the **same 169 queries** of test `n_c>=3` (paired McNemar exact + difference + bootstrap by identity, 2000, seed 20261022).
- **The S-vs-I question:** not answered at tier C; answered by **tier A §4.7** — SigLIP2 S−I **+22.0 pp** (p=1.3e-74) · CLIP-B S−I **+1.6 pp** (p=0.15) (per the tier A results already recorded).

## D9-part-2 — A(ii) mask bug fixed, tier B phase 2 rerun — 19/11/2026

**Fix (approved by the Director).** Fixed `prep/tierB_phase2_run.py` (old sha `9ddba81e…9073` → new sha
`5ad11a26abe98ab9119bd8de814a4b0a246f109b77c3effbaf5b1f66857dae4b`; the local version at
`scripts/prep/tierB_phase2_run.py` is in sync). `compute_cell()` now calls
`masked_top1_with_path(S, meta, ids, q_src, comp_map)`: it reads `dedup_components.json` (`_meta.theta_cos = 0.98`,
n=1667, sha `e2f07af2…5d2c`), and for each query x takes `src_path` from the manifest and masks **every gallery image
in the same `component_098` as x (including x)** with `S2[mask] = -np.inf` BEFORE the identity max-pool on the
already-blended score matrix (cell S, `w_img = 0.5`) or the pure cosine (cell I) — exactly `tierA_v3.ii_top1`. Different from
tierA: components of 1 image are not dropped because the query is T(x), so x is always present in the gallery.

**CSV schema.** The old column names `top1_identity/top1_path/score/correct` are kept but are now **A(ii)-masked**
(used for flip/Δ per §5.3.1–5.3.2); added `full_identity/full_path/full_score` = the whole gallery, **unmasked**
(used for self-ret §5.3.3b, where x must remain in the index). Unit test of the mask function on the server: `MASK_UNIT_OK`
(full `['A','B']` → masked `['B','A']`, correct path, `n_masked_cells=3`).

**Rerun.** PID 28108, 08:28→09:52, out `/tmp/prereg_v7/tierB_phase2_masked/`. Server-side after the run:
`n_rows = 110,022` (6 × 18,337), 66 (cell,transform) pairs × 1,667, **`mask_violations = 0`**
(every masked `top1_path` is NOT in the same component as x), **`full_same_siglip2_S_G1 = 1,497/1,667`**
exactly the number measured in D9 (tolerance 5%, actual deviation 0%), `ok/mask_ok/full_same_ok = true/true/true`.
`visual.index` before and after is `31b8f9d21462cbb52864e0924466e8bc41585bd1c5a67350be5fae3198ff8dc0`
(READ only). The job ended on the command `# tierB_phase2_run done` (the server restart at 14:24 only cut the process that
polled the remote shell, without affecting the files already written).

**Local recomputation.** `compute_part2.py` run on `tierB_top1_masked.csv` (sha `8657f03f…a430`):
`anomalies = []`, D(ii) = 1,663 images / 141 identities (D9: 145 identities in D(i), 4 fall out of D(ii)).
Self-ret is taken from the `full_*` columns (cell S uses the `full_path` of the corresponding cell I because §5.3.3b requires pure images);
**assert reproduces the old full-gallery version exactly**, pct + CI95 + n in all 6 cells: `selfret_assert.ok = true`
(99.65 / 95.14 / 95.34 / 99.12). The harness `tierB_part2.py` is NOT modified; 26/26 local tests pass.

**New results (replacing the INVALID table):** canonical Δ near 0 (5/6 cells `INVARIANT`, `clip_S` = −1.60 pp
`NOT ENOUGH EVIDENCE`); stress loses invariance in 4/6 cells (`clip_S −8.62`, `clip_I −9.16`, `dinov2 −5.88`,
`clipl14 −4.99` pp → `INVARIANCE LOST`), the 2 siglip2 cells `NOT ENOUGH EVIDENCE` (−2.45 / −2.08 pp) — exactly the
expectation of §5.1 that the heavy set separates the encoders. The INVALID table of 19/11 is kept in `report.md` section 6 and
`tierB_top1.fullgallery.INVALID.csv` + `tierB_part2_metrics.fullgallery.INVALID.json` (renamed only, not deleted).

**Operational note:** at pull time, the new schema file had the same basename `tierB_top1_columns.json` and so overwrote
the 711-byte trace version; it was restored byte-exact (sha `d07ad62e…8383fe`) and the new version was renamed to
`tierB_top1_masked_columns.json` (sha `77dc1ae0…967766`).


## D11 — P-a (accepted by the Director): the tier C universe narrowed 790→619, §6.4 pairing on 160 — 19/11/2026

**Scope.** Tier C case I (ab4 split), option P-a. **The SEALED prereg is not modified**. The original `measure_retrieval_quality.py` is not modified (md5 `585ba429…` kept), the old ab4 run/prereg/other batches' files are not touched, nothing deleted.

**(i) The tier C universe narrowed 790 → 619 index images (93 identities).**
- Reason: **171/790 source images of the ab4 split index are lost and cannot be recovered** (scanned 29,757 files across all of <SERVICE> + `_quarantine/**` + backup/`snapshot_S2` by basename+relpath + `<REDACTED_HOME>`, `<REDACTED_PATH>`, `/tmp`, `build_expand_aliases.py`: **120 are lost outright** — of which 99 are `Screenshot 2026-09-28...png` deleted by hand, 15 hash-named; **51** are basename collisions with the wrong location/identity; `_quarantine/trim_20261111` and the other quarantine branches do not contain a version with the correct relpath; the ab4 run stored only vectors+meta, **did not store the original index images** → a reproducibility lesson: the source images or a snapshot keyed by `gallery_hash` must be stored/pinned).
- The index is narrowed to **619 images** resolvable with the correct relpath (orig 493 + `snapshot_S2` 126) / **93 identities**. 7 identities are lost entirely: `ababil`, `bmp3`, `fmtv`, `fpv_kamikaze_aze`, `karrar_tank`, `lav25`, `t72b3`; 38 identities lose images.

**(ii) Paired test 169 → 160.**
- Of the 169 test `n_c>=3` queries of ab4 (full 790), **160** queries still have GT in `Y_c` and `n_c>=3` on the 619 universe (measured on the ab4 split, index 619; 265/284 queries in total still have GT in `Y_c`). The §6.4 paired McNemar is performed on exactly these **160** queries.

**(iii) BOTH encoders are rerun on 619 (fair, no mixing).**
- Rerun **CLIP-B-I** and **SigLIP2-I** on the same 619 universe (build the index from 619 images, re-encode both — the old ab4 index vectors are not used for CLIP-B, to follow the same path as SigLIP2). **Run A full 98/169 = 58.0% (repro 0 pp, `5070-261119-ab4ci-clipb`, full 790) is kept as a separate reference point**, **not compared directly** with the 619-universe results.

**(iv) §6.4.** Paired McNemar (exact binomial two-sided, difference + paired bootstrap by identity cluster, 2,000, seed 20261022) on exactly 160 queries (same qid across the two 619 runs).

## D12 — Fixing the Sc4 definition error in prereg_v9 (§1): intra-image → inter-identity gap in Y_c — 21/11/2026

**Scope.** prereg_v9 (SEALED, `prereg_v9.md`, `2e0246c9…`). Only stats + report + deviation are fixed. **No re-capture** (pass1/pass2 npz kept unchanged), the prereg/tests/Results batch files are not modified, no commit.

**The error.** v9 §1 describes Sc4 as `gap = m_id (s₁ − s₂ in Y_c, s₂ = the second-best image **of the same identity**)` — an intra-image reading (image-level within the same identity). This is a **descriptive error by the director**. The deciding clause in v9 is still correct: “exactly the ab4/spec_v2 definition (Sc1–Sc4), unchanged”.

**Code evidence (ab4).** `tools/measure_retrieval_quality.py` L486-496: `s_all[v]=max score by identity` → `s_y={y:s_all[y] for y in Y_c}` → `sv=sorted(s_y.values(), reverse=True)` → `s1=sv[0]; s2=sv[1]; m_id=s1-s2` = **the gap between two IDENTITIES in Y_c** (inter-identity, max-pool). L349-356: “Sc2≡Sc4 when n_c=2 (MSP = sigmoid(m/T))”.

**The fix.** Recompute the P1+P2 stats from the existing npz (`pass1_capture.npz`, `pass2_capture.npz`). Sc4 P1 = `sv[0]-sv[1]` on the max-pooled identity scores in `Y_c`, with `Y_c` built per the ab4 `COARSE_MAP` L450-465 (Airplane/Tank/Vehicle{vehicle,artillery,air_defense,missile}/Helicopter/Drone/Ship). Sanity n_c==2 (MSP(T*) vs Sc4): siglip2_I n=10, rank_equal_full=True, pairwise 45/45 concordance=1.0; clip_I n=10, rank_equal_full=False, pairwise 37/45 concordance=0.822 (the difference is because Sc2 MSP in P1 computes the softmax over the 145 global identities, while Sc4/Y_c is restricted; ab4 L349-356 assumes MSP over Y_c).

**The old (intra-image) results are kept as a trace.** `p1_results.intra_image_SUPERSEDED.json`, `report_P1.intra_image_SUPERSEDED.md`, `p2_results.intra_image_SUPERSEDED.json`, `report_P2.intra_image_SUPERSEDED.md` (not deleted). The new results are written to `p1_results.json`, `report_P1.md`, `p2_results.json`, `report_P2.md`; `SHA256SUMS.txt` is updated and includes the SUPERSEDED files.

**P2 impact.** `pass2_capture.npz` stores only the global top-2 identity scalars per transform, with no 145-identity vector; P2 was rerun but the P2 `m_id` is still the global identity top-2 (not Y_c-restricted). The P2 numbers do not change relative to before D12; a recapture is needed if a Y_c-restricted P2 is wanted.

## D12-part-2 — Recapture pass2b (id_scores_all) + P2 Y_c-restricted restats — 21/11/2026

**Scope.** prereg_v9 (SEALED). Supplementary P2 capture to allow computing the Y_c-restricted m_id (D12). The prereg/tests/Results batch files are not modified; the old `pass1_capture.npz`, `pass2_capture.npz` are not touched.

**Reason.** `pass2_capture.npz` stores only the global top-2 identity scalars; the Y_c-restricted inter-identity gap per D12/ab4 L486-496 cannot be computed without a recapture.

**Capture.** Modified `prereg_v9_capture.py` to add the stage `pass2b`: stores `id_scores_all` of shape (1667,11,145) float32 + `b_id_scores` (1667,145) along with the old scalars; new output `pass2b_capture.npz` + `pass2b_meta.json` (the pass2_* files are not overwritten). Run on <GPU_SERVER> with `setsid nohup`, finished in 1902.3s (~31.7 minutes), 18,337 queries (1667×11). Sanity: the global scalar `s1_id-s2_id` in pass2b matches the recomputed vector and matches the old pass2 for 18,337/18,337 cells (mismatch 0).

**P2 Y_c-restricted restats.** m_id = sv[0]-sv[1] in Y_c (ab4 COARSE_MAP L450-465); s1 keeps the global top1 identity score (absolute). Identity cluster bootstrap 2000/seed 20261022, the 3 prereg v9 statuses. Group results: canonical_6T 1.0249 [0.9804,1.0787] `NO DIFFERENCE SEEN` (status changed: NO); stress_5T 0.8775 [0.8300,0.9246] `MORE STABLE` (status changed: NO). Per-transform: L3 changes from `NO DIFFERENCE SEEN` (global 1.0810 [0.9944,1.1882]) to `LESS STABLE` (Y_c 1.1104 [1.0225,1.2198]); the other transforms do not change status (X1 near the margin: Y_c CI upper 1.0008).

**Trace.** `pass2_capture.npz/meta.json` are kept unchanged. `p2_results.global_scalar_preD12part2.json` + `p2_results_global_sensitivity.json` keep the global numbers; `p2b_results.json` + `report_P2.md` are the main Y_c-restricted results.

## D13 — Sensitivity: MSP/energy computed over the candidate set Y_c (does not change the sealed v9 results) — 21/11/2026

**Scope.** prereg_v9 (SEALED) and `results/prereg_v9_gap_20261121/*` stay unchanged. A sensitivity analysis on the same `pass1_capture.npz`; specification `spec_D13.md`; script `scripts/d13_yc_sensitivity.py`; results `results/D13_Yc_sensitivity_20261121/`; figure `fig2_rc_v9.png`.

**Reason.** In v9, MSP and energy were computed over the **global 145 identities** (`msp_at_T`, `energy_at_T` take the whole score row), whereas Table 1 of the paper defines MSP over Y_c. Only gap uses Y_c (D12). The definition mismatch is disclosed here.

**Method.** (1) MSP/energy are recomputed in Y_c with the **same sealed T\* = 0.01** (all 4 T\* are at the lower edge of the grid), **T is not re-chosen**. (2) `sim` and `correct` (the global prediction) stay global. (3) Y_c is built from `pass2b_meta.json` because `pass1_meta.json` lacks the category of 4 identities (`btr80`, `ec130`, `switchblade_600`, `z19`). (4) The n_c=2 stratum is only the Ship group (2 identities) = 6 test queries (0.7%); stratum (b) n_c>=3 has 826 queries. (5) The v9 bootstrap uses `hash((enc, other))` as the auxiliary seed, which is randomised per process so the v9 CI/p are not reproducible bit by bit (MC error ~0.001-0.002 in the CI; tail p fluctuates considerably); D13 uses the deterministic `zlib.crc32(f"{eid}|{other}")`.

**Results.** Reproducing the v9 point estimates (all_832/global) deviates by 0 (threshold 1e-6), CI deviates by at most 0.0019, the v9 labels match, the re-chosen T\* matches. Y_c labels versus global: **no label changes** (both encoders, both strata); see `results/D13_Yc_sensitivity_20261121/report.md`.


## D14 — Split table and terminology (ab4, Tier C 619, v9 RQ-1) — 22/11/2026

**Scope.** Descriptive post hoc analysis (C1 revision v9). Reads sealed data only; no sealed file is modified. Script `scripts/v9rev_split.py`; results `results/v9rev_split_20261122/results.json`.

**Findings.**
- ab4: 790 index images/100 identities; 284 query images/95 identities (95 ⊂ 100); calibration 114 images/66 identities, test 170 images/80 identities, 51 identities in common (split by image). It is "unseen query images of known identities", NOT identity-disjoint between calibration/test.
- Tier C (619): index 619 images/93 identities (171 images lost; 7 identities lost entirely); paired test 169 -> 160 (9 queries dropped), 73 identity clusters; still held-out query images of identities already in the index.
- v9 RQ-1: gallery 1667 images/145 identities; calibration 73 identities/831 rows, test 72 identities/832 rows; 0 identities in common (identity-disjoint holds only for calibration vs test).

**Adjustment.** The v8 label "Tier C (unseen identities)" is dropped; use "held-out query images" (see spec_v9_revision.md [T1]-[T6]). No new numbers: every number is recomputed from split.json/predictions/pass1_meta.

## D15 — Y_c sensitivity: 4 scores on the same candidate set, two label regimes (post hoc)

- Date: 2026-11-22. Script: `scripts/v9rev_yc.py` (Python311, numpy). Specification: `spec_v9_revision.md` (internal spec, not included) section 5 [C2].
- Data: `pass1_capture.npz`, `pass1_meta.json`, `pass2b_meta.json` (Y_c with the full 145 identities), `p1_results.json` (sealed T*, not re-chosen). Sealed data is not changed.
- s1, MSP, energy, gap are all computed in Y_c. Main regime `yc_argmax` (label = argmax in Y_c); `global_argmax` (the sealed label) matches D13 to within 1e-9.
- Δ E-AURC = E(gap) - E(other), identity-cluster bootstrap CI (B=2000, seed [20261022, crc32(tag)]); a CI containing 0 -> label `insufficient`.
- Number of cases where the global label differs from the Y_c label: SigLIP2-I 23, CLIP-B-I 233 (out of 1663); number of cases whose correct/wrong status differs: 9 and 67.
- Descriptive post hoc analysis; no conclusion that gap is better than MSP.

## D16 — Image-level gap m_img vs identity gap m_id (post hoc)

- Date: 2026-11-22. Script: `scripts/v9rev_imggap.py` (Python311, numpy). Specification: `spec_v9_revision.md` (internal spec, not included) section 5 [C3].
- Data: ab4 `predictions.jsonl` (columns m_id, m_img, top2_same_identity), v9 `pass1_capture.npz` (s1_img, s2_img_same_id, id_scores). Sealed data is not changed.
- Δ E-AURC = E(m_img) - E(m_id), identity-cluster bootstrap (B=2000). ab4_test n=169, ab4_calib n=114; top2_same_identity over the 284 queries = 111.
- Descriptive post hoc analysis; the design or conclusions of the paper are not adjusted.

## D17 — Descriptive ablation of the conflict gate (post hoc)

- Date: 2026-11-22. Script: `scripts/v9rev_gate.py` (Python311, numpy). Specification: `spec_v9_revision.md` (internal spec, not included) section 5 [C4].
- Data: ab4 `predictions.jsonl`; v9 `pass1_capture.npz` + Y_c + sealed T*. Sealed data is not changed.
- Gate: conflict -> score = -inf. ab4_test n=169 (conflict 20), ab4_calib n=114 (conflict 17).
- Only a descriptive ablation; no certification (certified) conclusion.

## D18 — One-sided Clopper–Pearson upper bound by threshold (ab4 calibration, v9 calibration) — 22/11/2026

**Scope.** Descriptive post hoc analysis (C5 revision v9). Reads sealed data only (ab4 predictions, v9 pass1 + sealed T*); T is not re-chosen. Script `scripts/v9rev_cp.py`; results `results/v9rev_cp_20261122/results.json`.

**Method.** Each distinct score value is a threshold row (accept when score >= tau, ties in the same row); the one-sided 95% CP upper bound of the error rate (k/n; = 1 if k = n). certified if the bound <= alpha; "observed risk above target" if k/n > alpha; otherwise "insufficient data". n required for 0 errors: 59 (5%), 299 (1%).

**Fixing the wrong v8 sentence.** The v8 sentence "Calibration size exceeds the bound, so the binding constraint is the error rate" is half right: ab4 has 114 calibration samples, the longest error-free prefix is 21 (< 59) so the 5% level is infeasible because of the error rate; the 1% level needs >= 299 error-free accepted samples > 114 so the constraint is a lack of data.

**Label.** Post hoc, descriptive; not a certification.

## D19 — Identity-cluster sign-flip permutation test (supplementing McNemar) — 22/11/2026

**Scope.** Descriptive post hoc analysis (C6 revision v9). Reads sealed data only; the McNemar already reported is kept unchanged. Script `scripts/v9rev_perm.py`; results `results/v9rev_perm_20261122/results.json`.

**Method.** d_i = correct_A − correct_B; T = sum d / n; flip the sign of a whole identity cluster with probability 1/2 (paired, two-sided); Monte Carlo B = 100000, seed 20261022, p = (1 + #{|T*| >= |T_obs|}) / (B + 1), the minimum resolvable p is 1/(B+1). Three comparisons: Tier C SigLIP2-I vs CLIP-B-I (160 queries, 73 clusters), Tier A SigLIP2 I vs S and CLIP I vs S (1663 rows, 141 clusters).

**Interpretation.** McNemar treats cases as independent; cases of the same identity are correlated so the McNemar p is anti-conservative. The cluster permutation p is a post hoc sensitivity check and does not replace the registered test.

## D20 - C7 Tier B: 11 query transforms, flip/rank-corr/decision-flip/score-variation (post hoc) - 20261122

- Scope: SigLIP2-I, pass2b_capture.npz (1667 queries x 11 transforms), post hoc, descriptive; sealed data is not changed.
- Metrics: top-1 flip, Spearman/Kendall tau-b per query (finite identities), decision flip s1 >= tau* = 0.960025787, score variation, rho = median|d gap_Yc| / median|d s1| (matches p2b_results.json to 1e-9).
- rho is a ratio of the magnitude of score variation, NOT evidence of decision stability; decision flip is the actual decision metric.
- Reproduction: scripts/v9rev_tierB.py.

## D21 — Diagnosing MSP vs identity gap on the same Y_c: the role of delta (post hoc, descriptive)

- Date: 22/11/2026. Script: `scripts/v10_diag_msp_gap.py` (Python311, numpy). Specification: `spec_v10_diag.md` (internal spec, not included).
- Data: `pass1_capture.npz`, `pass1_meta.json`, `pass2b_meta.json`, `p1_results.json` (read only). No remote shell, no embedding recomputation.
- Form: logit(MSP_T) = m_id/T - delta, delta = log(1 + sum_{j>=3} exp((s_j - s2)/T)); T = sealed T* = 0.01, NOT re-chosen. n_c=2 => delta = 0 => the gap ordering coincides with MSP.
- Descriptive post hoc analysis (no p-value, no hypothesis test); NO conclusion that gap is better than MSP or vice versa. Logistic/bin/threshold fitted only on calib; test is only applied.
- The actual n_c differs from the initial assumption: only {2, 19-22, 30, 52}, with no n_c of 3-4 or 5-18; the pre-declared n_c strata follow these four levels.
- n_img_top1/top2/true are the number of ORIGINAL gallery images of the identity (approximate): images masked as near-duplicate/LOO are not subtracted.
- Quick summary:
- siglip2_I (test): E-AURC gap-MSP all = +0.00178 [+0.00044, +0.00348]; nc_ge3 = +0.00181 [+0.00036, +0.00333]; discordant pairs: MSP correct 1794, gap correct 1014 (n_opposite=2808).
- clip_I (test): E-AURC gap-MSP all = +0.00856 [+0.00221, +0.01473]; nc_ge3 = +0.00929 [+0.00335, +0.01577]; discordant pairs: MSP correct 9540, gap correct 6123 (n_opposite=15663).

## D22 - Rerunning the P1 v9 bootstrap with a deterministic auxiliary seed (crc32)

**Reason.** `prereg_v9_stats.compute_p1` uses the auxiliary seed `hash((enc, other)) % 2**32`; the `hash()` of a string/tuple is randomised per process (PYTHONHASHSEED) so the P1 CI and p are not reproducible bit by bit (D13 item 5: CI deviates by at most 0.0019).

**Method.** A new script `scripts/prereg_v9_stats_det.py` (the sealed version is not modified), the same analysis (B = 2000, clusters by test identity, paired E-AURC, percentile [2.5; 97.5], two-sided p with +1 smoothing, Holm m = 3), changing only the auxiliary seed: `default_rng([20261022, zlib.crc32(f"{eid}|{other}".encode())])` (the same tag as D13, so it coincides bit by bit with D13 `all_832/global`).

**Scope.** Only the CI/p/Holm of P1 change. The point estimates (delta, E-AURC, metrics_test) and T\* do not change. The 6/6 labels do not change.

Maximum CI deviation |d| = 0.001908; maximum raw p deviation = 0.019990.

**Table of old -> new deviations.**

| Encoder | Comparison | CI lo old -> new | CI hi old -> new | p old -> new | Holm old -> new | Label old -> new |
|---|---|---|---|---|---|---|
| SigLIP2-I (siglip2_I) | gap_minus_sim | -0.050575 -> -0.051112 | -0.008614 -> -0.007012 | 0.001000 -> 0.008996 | 0.002999 -> 0.017991 | better -> better |
| SigLIP2-I (siglip2_I) | gap_minus_MSP | -0.000665 -> -0.000601 | +0.006585 -> +0.006336 | 0.117941 -> 0.118941 | 0.117941 -> 0.118941 | insufficient -> insufficient |
| SigLIP2-I (siglip2_I) | gap_minus_energy | -0.058336 -> -0.058587 | -0.009967 -> -0.010269 | 0.002999 -> 0.003998 | 0.005997 -> 0.011994 | better -> better |
| CLIP-B-I (clip_I) | gap_minus_sim | -0.135171 -> -0.134851 | -0.042403 -> -0.044311 | 0.001000 -> 0.001000 | 0.002999 -> 0.002999 | better -> better |
| CLIP-B-I (clip_I) | gap_minus_MSP | -0.016443 -> -0.015909 | +0.026107 -> +0.026274 | 0.746627 -> 0.766617 | 0.746627 -> 0.766617 | insufficient -> insufficient |
| CLIP-B-I (clip_I) | gap_minus_energy | -0.169259 -> -0.170756 | -0.069389 -> -0.070693 | 0.001000 -> 0.001000 | 0.002999 -> 0.002999 | better -> better |

**Reference.** D13 (`results/D13_Yc_sensitivity_20261121/results.json`, `all_832/global`); results: `results/prereg_v9_det_20261122/results.json`. The original `results/prereg_v9_gap_20261121/p1_results.json` is kept unchanged.

**New P1 sentence for the paper (Δk = E_k − E_gap; CI; Holm p):**

> +0.027 [+0.007, +0.051] (0.018); −0.003 [−0.006, +0.001] (0.119); +0.033 [+0.010, +0.059] (0.012); +0.090 [+0.044, +0.135] (0.003); −0.005 [−0.026, +0.016] (0.767); +0.118 [+0.071, +0.171] (0.003)

**Places to update (a later wording pass, not edited here).**
- `paper_ares_v9.docx` (the manuscript): the sentence "Against the gap, Δ with Holm p is ..." (6 cells of CI/Holm p); source `content_v9.py` (internal build script) line 657.
- `ESM_v9.docx`: the paragraph "Bootstrap seed ... hash() ..." is rewritten as the D22 deterministic rerun.
- `deviation_log.md` (the project's working copy): add the D22 entry from this file (D13 is not modified).
