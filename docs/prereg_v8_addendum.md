# prereg_v8 — ADDENDUM: Switching the paper's primary encoder according to the pre-declared rule

> English translation of the original (Vietnamese) document; content, numbers and codes are unchanged. The sha256 recorded in `prereg_v8_addendum.md.sha256` is the hash of the original Vietnamese file, not of this translation.

- **Status: SEALED — 21/11/2026** (the project owner signed "Signed — SEALED" on 21/11; hash in `prereg_v8_addendum.md.sha256`).
- **Date:** 20/11/2026 (drafted 21/11). **Depends on:** `prereg_v7.md` SEALED `e5080142fac7b6ec6eb4aefcf6a12faa77be934671c97bd77bb63442a8774d1d` (**not modified**) · `decision_log.md` (internal), entry **R (18/11)** · numbers quoted from `PREREGV7_MASTER_20261119.md`.
- **Scope:** changes only the **presentation direction of the paper** (which encoder is the primary one). **Does NOT** change any parameter, metric, seed, tolerance, or deviation of prereg_v7.

## 1. Rule fixed BEFORE the numbers were available (decision_log R, 18/11)

> CLIP (B/32) = **baseline**. Whichever embedding model the **benchmark proves to be better** → **revise the paper to make that model the primary one**, because the benchmark results show so. Order: **tier C (unseen) primary**, tier A/B diagnostics; cross-compare with scale-independent criteria. If the winner ≠ CLIP-B → a prereg amendment before running; ab4 keeps the baseline role and is not deleted.

## 2. Condition MET — 4/4 pieces of evidence favour SigLIP2-I

| Evidence | SigLIP2-I | CLIP-B-I (baseline) | Source |
|---|---|---|---|
| **Tier C — unseen, paired (PRIMARY per the rule)** | **129/160 = 80.6%** | 83/160 = 51.9% | §6.4: **+28.75pp · McNemar exact p = 1.7×10⁻⁹ · bootstrap CI [+20.5; +36.9]pp** (73 clusters, B=2000, seed 20261022), run `5070-261119-ab4ci619-{clipb,siglip2}` (D11) |
| Tier A — seen/LOO (diagnostic) | **81.6%** · AURC **0.086** · τ\* **the only feasible one** | 48.2% · AURC 0.401 · infeasible | `loo_summary*.json` fp32 (D8) |
| Tier B — invariance (diagnostic) | canonical **−0.11pp INVARIANT** · stress −2.08 (NOT ENOUGH EVIDENCE) · flip **2.6–7.9%** | canonical −1.07 · stress **−9.16 INVARIANCE LOST** · flip 10.5–22% | `tierB_part2_metrics.json` |
| §4.7 — S vs I (secondary) | I beats S by **+22.0pp, p = 1.3×10⁻⁷⁴** | +1.6pp, p = 0.15 | `rq1/mcnemar.json` |

## 3. Decision (at signing)

1. **The paper's primary encoder = SigLIP2-I (image only).** The Results/Abstract/Intro/Contributions sections are presented around it.
2. **CLIP-B/32 = reference baseline** — the ab4 figure (**98/169 = 58.0%**, repro 0pp) is kept unchanged as the registered run, **not deleted, not rewritten**; relabelled "baseline (case I)" per D10.
3. **Report every cell** per §9.4 of prereg_v7 — every cell of the 6-cell/2-case/3-tier grid appears (including weak cells), not only the winning ones.
4. **Selection honesty (nothing hidden):** the fact that the winner was chosen on the **same benchmark** will be published together with (i) the rule pre-declared on 18/11, (ii) all cells, (iii) the large paired-test effect (p=1.7×10⁻⁹) — no claim beyond the evidence.
5. **Unchanged:** prereg_v7 (D1–D11) · the ab4 protocol · the old paper numbers · baseline text B1 · H1 identity gap (**still insufficient at n=169** — experiment P1/P2 is separate and not part of this addendum).

## 4. Consequences: work to do after signing

- Write Results v6 around SigLIP2-I (using `PREREGV7_MASTER_20261119.md`) · relabel "case I/ab4 baseline" (D10) · new figures/tables (tier A/B/C + §6.4) · update Abstract/Contributions · compress to ≤12 pages.
- If the project owner chooses **P1/P2** (proving the gap) → prereg_v9 is separate and does not touch v8.

## 5. Seal

The hash is recorded in `prereg_v8_addendum.md.sha256` (computed on the version carrying the SEALED status above).
