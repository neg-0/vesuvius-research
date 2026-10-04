# V-013 fine depth sweep and label-free selection — mixed; safety clause FAILED

GPU sweep on the RTX 3080 (OfficeComputer, runs `20261004T073315-cac6d770a2`,
interrupted `20261004T075552-63773e953a`, final `20261004T095609-ce522a51c1`):
25 flattened depths (−48…+48 layers, ~9.6 µm steps) × 8 labelled PHerc0139
March-scan windows. w044 flattened depths match the reporter to 4 decimals.
Selection scored in the cloud from the committed ds8 maps
(`v013_select.py`, run `20261004T105802-2627bb2564`).

| Segment | R1 (primary) | offset | Reporter label-free | Δ | Oracle (fine) | As published | 2.399 µm ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| w030 | 0.9070 | −20 | 0.9072 | −0.000 | 0.9208 | 0.9233 | 0.9357 |
| w040 | 0.8172 | −4 | 0.8042 | +0.013 | 0.8172 | 0.8118 | 0.8724 |
| w041 | 0.7795 | +28 | 0.8019 | **−0.022** | 0.8460 | 0.8382 | 0.8882 |
| w043 | 0.7408 | −24 | 0.7719 | **−0.031** | 0.8138 | 0.7678 | 0.9205 |
| w044 | 0.8864 | −44 | 0.9027 | −0.016 | 0.9025 | 0.6958 | 0.9655 |
| w045 | 0.7880 | −32 | 0.7000 | +0.088 | 0.7880 | 0.6129 | 0.8395 |
| w039 | 0.7777 | −44 | 0.7824 | −0.005 | 0.7941 | 0.6434 | 0.7760 |
| w035 | 0.8418 | −12 | 0.7530 | +0.089 | 0.8418 | 0.7521 | 0.7818 |
| **Mean** | **0.8173** | | **0.8029** | +0.014 | **0.8405** | 0.7557 | 0.8725 |

Other registered rules (means): R2 mean-probability 0.8075, R3 per-pixel max
0.7806, R4 per-2 mm R1 0.7981.

**Verdict against the registered rule:** R1 beats 0.803 on the mean (0.8173),
but loses more than 0.02 on two already-aligned segments (w041 −0.022, w043
−0.031), so V-013 does not pass. The original CT-only predictor P1 was already
rejected.

**What is still informative:**
- Finer depth steps raise the label-oracle ceiling from 0.810 (reporter's 5
  windows) to 0.841; the gains are w045 and w035.
- Every segment's label-best offset is negative (−4 to −44 layers), including
  the five aligned ones: the March meshes sit systematically on the same side of
  the inked surface. This was observed after seeing all 8 segments, so it is a
  hypothesis for another scan, not a validated correction.
- w041's R1 pick (+28) is the only positive pick and the worst miss; a
  sign-constrained search would fix it here but is post hoc.

No submission, publication or contact. All 8 labelled segments on this scan
have now been used for selection; there is no untouched test set.
