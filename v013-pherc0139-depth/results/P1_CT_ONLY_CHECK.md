# V-013 cheap check — CT-only predictor P1: REJECTED by its registered rule

Run `20261004T024342-f42aeee500` (scoring); CT renders/flattening run
`20261004T013736-affcfd3907` (8 segments, CT only, no ink model).

| Set | Scorable sub-windows | P1 matches | Constant-0 matches |
|---|---:|---:|---:|
| All 8 segments | 30 | 16 | 13 |
| 7 held-out (excl. design segment w044) | 26 | 12 | 13 |

The registered pass rule needed P1 to beat constant 0 on the held-out set; it
did not (12 vs 13), so P1 is rejected and no ink pass is run for it.

What it did get right: w044 4/4 (predicted −35 to −40 layers; label best −40,
the edge of the reporter's search range) and the direction of w045's tilt
(predicted −17, −21, −27, −33 vs label best +20, 0, −20, −40). It moved
aligned segments off 0 wrongly (w035 sub-window 3 predicted +46; w040 sub 3 −22).

Caveats: label-best depths come from 5 windows 20 layers apart and several
sub-windows have near-tied AUCs, so the match metric is coarse. All 8 labelled
segments have now been seen; any revised predictor tuned on them has no
independent test set on this scan.

Payload: [2026-10-04-v013-p1-check.json](2026-10-04-v013-p1-check.json).
