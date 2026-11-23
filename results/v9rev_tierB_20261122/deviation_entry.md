## D20 - C7 Tier B: 11 query transform, flip/rank-corr/decision-flip/score-variation (post hoc) - 20261122

- Pham vi: SigLIP2-I, pass2b_capture.npz (1667 query x 11 transform), post hoc, mo ta; khong doi du lieu sealed.
- Chi so: top-1 flip, Spearman/Kendall tau-b theo query (identity huu han), decision flip s1 >= tau* = 0.960025787, score variation, rho = median|d gap_Yc| / median|d s1| (khop p2b_results.json 1e-9).
- rho la ti so do lon bien thien diem, KHONG phai bang chung on dinh quyết định; decision flip moi la chi so quyết định.
- Tai lap: scripts/v9rev_tierB.py.
