# V-016 arm P control: not a valid comparison (2 of 3 folds diverged)

One RTX 3080, 2026-10-05. The driver prints `VERDICT ... 'verdict': 'FAIL'` for arm P (mean fine-tuned 0.298,
verdict run `20261005T180742-e7a21327c7`). That number is a numerical failure, not a measurement of the published mesh.

| Fold | Held-out | P zero-shot | P fine-tuned | Effective training |
|---|---|---|---|---|
| 0 | w013 / w018 | 0.6078 / 0.4860 | 0.0000 / 0.0000 (NaN maps) | stopped updating near iteration 337 of 2000 |
| 1 | w023 / w028 | 0.6060 / 0.6139 | 0.7217 / 0.8917 | 2000 of 2000, 4 skipped micro-batches |
| 2 | w029 / w031 | 0.5998 / 0.5490 | 0.0000 / 0.1770 (NaN maps) | stopped updating near iteration 809 of 2000 |

What happened in folds 0 and 2: backbone activations blew up during fine-tuning. On a held-out tile, layer4's largest
activation is 3691 (fold 0) and 4196 (fold 2), against 8.7 for the canonical model, 15.9 for C fold 0 and 56 for
P fold 1. The weights are finite and an fp32 forward is finite, but under fp16 autocast the decoder overflows
(`decoder.depth_collapse` is the first non-finite module), so every later training step was skipped by the
Amendment 1 guard and inference (also fp16) returns NaN.

The one valid P fold does not show C ahead of P: fold 1 gives P 0.722 / 0.892 against C 0.843 / 0.852 on the same
segments (one each way). So V-016 gives no ink evidence yet that the refit mesh beats the published one.

Runs: prep `20261005T145533-a47fdf391a` (timed out), `20261005T164545-75ee89f1e7` (success); baseline w028 P
`20261005T170312-c279d6543d` (0.6139, equals V-014); fold 1 train `20261005T170930-3245c61944`, eval
`20261005T173427-8476c93a5f`; fold 0 train `20261005T175119-d5cb825601` (after four crashed attempts, Amendment 1a),
eval `20261005T180002-4a93800efe`; fold 2 train `20261005T180113-7f96d5073d`, eval `20261005T180635-a3d27fe4cb`.

A valid control needs a recipe that stays stable on both arms (for example a lower learning rate, or frozen
BatchNorm affine parameters, or fp32), applied to C and P alike. That changes the registered design, so it is a new
registered run, not a rerun.
