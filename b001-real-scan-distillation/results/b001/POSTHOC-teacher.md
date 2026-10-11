# B-001 post hoc (descriptive, not part of the rule): how close is REAL to the 2.4 µm teacher?

Computed in the cloud on 2026-10-11, after the verdict. The teacher is the canonical 2.4 µm model's published
prediction of each test segment, read through B-001's CT-only correction at every LR px. The pixels are B-001's
scoring pixels (CT-valid, supervised, teacher in bounds) on the same 49 windows.

| scroll | windows | S0 (stock ink_9um) | REAL (B-001) | teacher (2.4 µm map) |
| --- | --- | --- | --- | --- |
| PHerc0009B (8.64 µm) | 12 | 0.7349 | 0.8393 | 0.9275 |
| PHerc0500P2 | 12 | 0.7277 | 0.8444 | 0.9628 |
| PHerc0841 | 25 | 0.6921 | 0.7652 | 0.8739 |
| all | 49 | 0.7113 | 0.8027 | 0.9088 |

REAL closes about 46% of the gap between the stock 9 µm model and the 2.4 µm teacher (0.0914 of 0.1975). REAL is at
or above the teacher on 6 of 49 windows. Caveat: the human labels were likely drawn with the high-resolution
renders and predictions in view, which favours the teacher.

Also descriptive, from `summary.json`: S0 scored with grid-index (uncorrected) labels averages 0.7086 vs 0.7113 with
corrected labels. The 100-300 µm label misplacement changes window-level AUC only slightly here.
