# Selection of Fig. 2 row 2

Row 2 of the qualitative figure was chosen on the outcome (hand-picked, post hoc) in three rounds.

Row 2 of Fig. 2 was chosen on the outcome (hand-picked) in three rounds that were run in this order: round 1 (Commons photographs, declared 2026-11-22T02:24:52Z), round 2 (held-out gallery images, leave-one-out, declared 2026-11-22T08:31:53Z) and round 3 (exact-box crop, declared 2026-11-22T11:03:30Z). Round 3 was declared after rounds 1 and 2 had failed, so the figure is post hoc and illustrative and is not part of the preregistered evaluation. The threshold 0.960 is unchanged in all three rounds (τ* from rq1_results.json). Box (b) of the composites is the public-domain MH-65 helicopter photograph (outside the gallery); it is refused in every round-3 composite, with s1 0.801, because s1 is below τ*. Its top-1 identity is aw101_merlin, so the refusal does not mean that the system recognises the helicopter.

## Round 1: Commons photographs

Five candidate photographs of test-split identities were fixed in order before the run. Four valid photographs were scored and all four were refused, with s1 between 0.781 and 0.901 (below 0.960). rank 5, uh60_black_hawk was excluded before inference (not a single clear aircraft (UH-60 cabin interior with crew member; checked by eye before inference)). Status: no_qualifying_candidate.


| Rank | Identity | s1 | Decision |
|---|---|---|---|
| 1 | p8_poseidon | 0.901 | refuse |
| 2 | kc135_stratotanker | 0.781 | refuse |
| 3 | ch47_chinook | 0.889 | refuse |
| 4 | p8_poseidon | 0.864 | refuse |

## Rounds 2 and 3: twelve leave-one-out candidates

Candidates are held-out gallery images (leave-one-out) of test-split identities, filtered in this order (images left after each step): 832 (split test), 375 (category aircraft helicopter), 244 (filename exclusion), 49 (accept pass1), 49 (top1 correct), 43 (not row1 identity), 12 (licence manifest). The licence step (CC0, public domain or CC BY) removed 31 of the 43 images; all 12 remaining candidates passed the eye check. The order is sorted by image id, not by score.

Round 2 used the box expanded by 15% (grey margin included): 12 of 12 candidates were refused in the composite (s1 between 0.901 and 0.946), with the correct top-1 identity in every case (12 of 12). Round 3 changed one factor, the crop: the exact box. It accepted 12 of 12 (s1 between 0.961 and 0.979), with the correct top-1 in 12 of 12. The composite images are the same in both rounds. s1 pass 1 is the leave-one-out score on the full image.


| Image id | Identity | s1 pass 1 | s1 round 2 (15% crop) | s1 round 3 (exact box) | Round 3 |
|---|---|---|---|---|---|
| a400m_atlas_bc50ef27.jpg | a400m_atlas | 0.972 | 0.923 | 0.971 | accept |
| b1b_276f7599.jpg | b1b | 0.974 | 0.917 | 0.976 | accept |
| b1b_30192fc4.jpg | b1b | 0.978 | 0.908 | 0.979 | accept |
| b1b_b22c6f15.jpg | b1b | 0.978 | 0.919 | 0.979 | accept |
| e3_sentry_39884534.jpg | e3_sentry | 0.964 | 0.927 | 0.963 | accept |
| e3_sentry_e5883adf.jpg | e3_sentry | 0.964 | 0.923 | 0.963 | accept |
| f18_hornet_5b7d2c06.jpg | f18_hornet | 0.967 | 0.906 | 0.966 | accept |
| f18_hornet_6c80ba6c.jpg | f18_hornet | 0.969 | 0.908 | 0.968 | accept |
| f18_hornet_6f9d84f9.jpg | f18_hornet | 0.963 | 0.901 | 0.962 | accept |
| f18_hornet_d0156697.jpg | f18_hornet | 0.969 | 0.922 | 0.968 | accept |
| su34_fullback_28e08df3.jpg | su34_fullback | 0.966 | 0.946 | 0.968 | accept |
| su34_fullback_3cacbd29.jpg | su34_fullback | 0.966 | 0.938 | 0.961 | accept |

Displayed in Fig. 2 box (a): a400m_atlas_bc50ef27.jpg (Ronnie Macdonald, CC BY 2.0; Commons File:Airbus_A400M_05.jpg), the first candidate in image-id order that is accepted with the correct top-1 identity in round 3 while box (b) is refused. Its s1 is 0.972 on the full image (pass 1), 0.923 in round 2 and 0.971 in round 3; scores in the figure are recomputed on the crops of the composite. The same images move from refuse to accept when the crop changes from the 15% expansion to the exact box, so τ* is sensitive to the crop; the grey margin and the resizing were not separated, so the cause is not isolated.

Accept decisions under leave-one-out may owe to near-duplicate or same-series photographs that remain in the gallery below the 0.98 merge threshold. The A-50 label error (section 10) belongs to the v8 row 2.

Fig. 2 row 2, thumbnails of the UH-60 and SA342 boxes (file title and Commons page without scheme, read from thumbs_override.json at build time): uh60_black_hawk: File:UH-60_2nd_Squadron,_2nd_Cavalry_Regiment_(cropped).jpg, commons.wikimedia.org/wiki/File:UH-60_2nd_Squadron,_2nd_Cavalry_Regiment_(cropped).jpg (Public domain; Gertrud Zach, US Army); sa342_gazelle: File:An_Egyptian_(French_built)_SA-342_Gazelle_helicopter_lands_near_a_Soviet_built_tank_during_Bright_Star_'82,_an_exercise_involving_troops_from_the_U.S.,_Egypt,_Sudan,_Somalia,_and_Oman_DF-ST-85-03622.jpg, commons.wikimedia.org/wiki/File:An_Egyptian_(French_built)_SA-342_Gazelle_helicopter_lands_near_a_Soviet_built_tank_during_Bright_Star_'82,_an_exercise_involving_troops_from_the_U.S.,_Egypt,_Sudan,_Somalia,_and_Oman_DF-ST-85-03622.jpg (Public domain; MSgt Don Sutherland).
