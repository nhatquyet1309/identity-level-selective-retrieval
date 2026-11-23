# Algorithm 1: Gate-and-Score

Gate-and-Score computes, for every detected box, one score per identity inside the candidate set Yc, takes the identity gap m = s1 - s2 as the confidence score and returns a decision set. The decision is ACCEPT when m reaches the threshold tau and the unfiltered nearest neighbour does not conflict with the mapped class; otherwise the output is REFUSE with its reason (undefined gap, conflict or m below tau). The index F and the gallery G are not modified. Yc is the set of gallery identities whose category is mapped from the detector class c by M.

```
     Require: image I; detector D; frozen encoder E; flat index F over gallery G = {(vi, yi)}; threshold τ; class-to-category map M
     Ensure: decision set R
  1  R ← ∅
  2  for all (b, c) ∈ D(I) do    // Stage 1
  3  z ← E(expand(b, 15 %)); q ← z / ‖z‖2    // crop, encode
  4  Yc ← { y : cat(y) ∈ M(c) }; nc ← |Yc|
  5  if nc < 2 then    // gap undefined
  6  R ← R ∪ {(b, REFUSE, undefined-gap)}; continue
  7  end if
  8  sy ← maxi : y_i = y ⟨q, vi⟩ for all y ∈ Yc    // per-identity max
  9  y1 ← arg maxy ∈ Y_c sy; s1 ← s(y1)
 10  s2 ← max over y ∈ Yc with y ≠ y1 of sy; m ← s1 − s2    // margin
 11  yall ← yi* with i* = arg maxi ∈ G ⟨q, vi⟩    // unfiltered NN
 12  conf ← 1[ cat(yall) ∉ M(c) ]
 13  if m ≥ τ and conf = 0 then    // gate
 14  R ← R ∪ {(b, ACCEPT, y1)}
 15  else
 16  R ← R ∪ {(b, REFUSE, conflict or m < τ)}
 17  end if
 18  end for
 19  return R    // F and G unchanged
```

Notes: s_y is the maximum cosine similarity over the gallery images of identity y; nc = |Yc| must be at least 2 for the gap to exist; the conflict flag is a descriptive gate (post hoc ablation, see results/v9rev_gate_20261122). Decisions are ACCEPT or REFUSE only; a person reviews every output.
