# V-016b result: the refit PHerc1667 mesh trains a better 1.129 µm ink model than the published one

**PASS (seed 0).** We fine-tuned the 2.4 µm canonical ink model with one fixed recipe on 1.129 µm renders, once
along the refit meshes (C) and once along the published meshes (P). The model trained along C reads held-out ink
better on all 6 segments, with a mean gain of +0.208 AUC.

This is ink-level evidence for [villa #1843](https://github.com/ScrollPrize/villa/issues/1843), the report that the
published PHerc1667 1.129 µm transform is wrong. Labels drawn on the 2.399 µm scan supervise the right papyrus only
when the 1.129 µm CT is placed by the refit mesh.

| Segment | C (refit) | P (published) | C − P |
| --- | --- | --- | --- |
| w013 | 0.948 | 0.661 | +0.287 |
| w018 | 0.911 | 0.809 | +0.102 |
| w023 | 0.833 | 0.569 | +0.264 |
| w028 | 0.840 | 0.551 | +0.289 |
| w029 | 0.879 | 0.791 | +0.088 |
| w031 | 0.743 | 0.525 | +0.218 |

Each value is the held-out AUC at the test window. The setup is 3 leave-a-pair-out folds, with the same patches,
loss, schedule, 2000 iterations and seed 0 for both arms. Evaluation is the same as V-014 and V-016.

## Validity

The rule was registered before training ([PROTOCOL.md](PROTOCOL.md)). It required all six trainings to skip at most
1% of steps and to produce no NaN maps. All six skipped 0 steps.

The only change from [V-016](../v016-cross-scan-ink/) is bfloat16 instead of float16, applied to both arms. In the
float16 attempt, 2 of the 3 P folds diverged; see `results/fp16-P-control-note.md`.

C under bf16 reproduces V-016: mean 0.859 against 0.864, and every segment is within 0.012.

## Supporting observations

- P barely learns. Its training loss over the last 200 iterations averages 0.60–0.63, against 0.33–0.38 for C.
- On w023, w028 and w031, P's best label alignment sits at or next to the ±48 px search limit (46–48 px). C's stays within ±4 px on every segment.
- On P's own terms, P fine-tuned beats P zero-shot on only 3 of 6 segments (mean 0.651 vs 0.577). C beats C zero-shot
  on 6 of 6 (mean 0.859 vs 0.564).

## Caveats

- **One seed per arm.** P is sensitive to numerical precision. Its fold 1 (w023/w028) scored 0.722 / 0.892 in the
  float16 run and 0.569 / 0.551 here, with the same seed and data. The direction is consistent on all 6 segments, but
  the size of the gap should not be quoted as stable yet. A seed-1 replication of both arms is registered (protocol
  Amendment 1) and will be added here.
- Six segments from one scroll and one scan. The labels were drawn on 2.399 µm.
- A model can partly learn around a misplacement. The gap therefore measures how much worse P supervision is, not the
  size of the transform error.

## Reproduce

[`code/v016_cross_scan_ink.py`](code/v016_cross_scan_ink.py) is the V-016 driver. It sits in the same `src` tree as
V-014 and uses V-014's meshes; see [../v016-cross-scan-ink/code/README.md](../v016-cross-scan-ink/code/README.md). For
this run set `V016_AMP=bf16 V016_TAG=_b`. Then, for each arm `C` and `P` and each fold 0–2, run `train <arm> <fold>`
followed by `eval <arm> <fold>`. Finish with `compare`.

| Step | Run ID |
| --- | --- |
| C fold 0 / 1 / 2 train | `20261005T185531-26f213b0a3`, `20261005T192027-786f7af268`, `20261005T194755-ed62f75931` |
| C fold 0 / 1 / 2 eval | `20261005T191856-4e65755303`, `20261005T194613-2a7b15630e`, `20261005T201802-0d54704461` |
| P fold 0 / 1 / 2 train | `20261005T201951-d7f4b473e5`, `20261005T205546-1ce5226b2b` (resumed after `20261005T204733-4810e56b01` hit a time limit), `20261005T211829-4bb0bc8e3a` |
| P fold 0 / 1 / 2 eval | `20261005T204541-04c0a5790c`, `20261005T211654-f95b339e82`, `20261005T214309-cf6ead4301` |
| verdict C / P, compare | `20261005T214450-e73f81e68a`, `20261005T214452-07b908e8a3`, `20261005T214454-b72bbc3faf` |
