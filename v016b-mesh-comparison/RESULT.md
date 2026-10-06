# V-016b result: the refit PHerc1667 mesh trains a better 1.129 µm ink model than the published one

**PASS, replicated with a second seed.** We fine-tuned the 2.4 µm canonical ink model with one fixed recipe on 1.129 µm renders, once
along the refit meshes (C) and once along the published meshes (P). The model trained along C reads held-out ink
better on all 6 segments in both seeds: 12 of 12 comparisons, with mean gains of +0.208 (seed 0, the registered
result) and +0.182 (seed 1).

This is ink-level evidence for [villa #1843](https://github.com/ScrollPrize/villa/issues/1843), the report that the
published PHerc1667 1.129 µm transform is wrong. Labels drawn on the 2.399 µm scan supervise the right papyrus only
when the 1.129 µm CT is placed by the refit mesh.

| Segment | C seed 0 | C seed 1 | P seed 0 | P seed 1 | C − P (s0 / s1) |
| --- | --- | --- | --- | --- | --- |
| w013 | 0.948 | 0.962 | 0.661 | 0.626 | +0.287 / +0.336 |
| w018 | 0.911 | 0.899 | 0.809 | 0.687 | +0.102 / +0.212 |
| w023 | 0.833 | 0.831 | 0.569 | 0.606 | +0.264 / +0.225 |
| w028 | 0.840 | 0.831 | 0.551 | 0.750 | +0.289 / +0.081 |
| w029 | 0.879 | 0.879 | 0.791 | 0.816 | +0.088 / +0.062 |
| w031 | 0.743 | 0.746 | 0.525 | 0.573 | +0.218 / +0.173 |
| **Mean** | **0.859** | **0.858** | **0.651** | **0.676** | **+0.208 / +0.182** |

Each value is the held-out AUC at the test window. The setup is 3 leave-a-pair-out folds, with the same patches,
loss, schedule, 2000 iterations and seed for both arms (seed 0, then seed 1 per protocol Amendment 1). Evaluation is the same as V-014 and V-016.

## Validity

The rule was registered before training ([PROTOCOL.md](PROTOCOL.md)). It required all six trainings to skip at most
1% of steps and to produce no NaN maps. All six skipped 0 steps, and so did all six seed-1 trainings.

The only change from [V-016](../v016-cross-scan-ink/) is bfloat16 instead of float16, applied to both arms. In the
float16 attempt, 2 of the 3 P folds diverged; see `results/fp16-P-control-note.md`.

C under bf16 reproduces V-016: mean 0.859 against 0.864, and every segment is within 0.012.

## Supporting observations

- C is stable across seeds: every segment is within 0.014. P is not: w028 moves by 0.20 between seeds.
- P learns less. Its training loss over the last 200 iterations averages 0.60–0.63 (seed 0) and 0.57–0.64 (seed 1),
  against 0.33–0.38 and 0.33–0.49 for C.
- On w023, w028 and w031, P's best label alignment sits at or next to the ±48 px search limit (46–48 px). C's stays within ±4 px on every segment.
- On P's own terms, P fine-tuned beats P zero-shot on 3 of 6 segments in seed 0 and 6 of 6 in seed 1 (means 0.651 and
  0.676 against 0.577). C beats C zero-shot on 6 of 6 in both seeds (means 0.859 and 0.858 against 0.564).

## Caveats

- **Two seeds.** The direction holds on all 12 segment comparisons, but P's per-segment numbers move by up to 0.20
  between seeds (and by more between float16 and bfloat16), so quote the gap as roughly 0.18–0.21, not as one number.
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
| seed 1: C fold 0 / 1 / 2 train | `20261006T000652-56656e5a42`, `20261006T003452-9e57ab8300`, `20261006T010443-a79f152a4a` |
| seed 1: C fold 0 / 1 / 2 eval | `20261006T003152-06c5a8cea2`, `20261006T010134-cf7ee9ddad`, `20261006T013011-e37c497a17` |
| seed 1: P fold 0 / 1 / 2 train | `20261006T013326-88b845e6f0`, `20261006T020656-f8ebfbc994` (resumed after `20261006T020215-0090a13051` hit a time limit), `20261006T023123-24b93713ca` |
| seed 1: P fold 0 / 1 / 2 eval | `20261006T015953-86c2c396ed`, `20261006T022857-f6cde22a2a`, `20261006T025629-fdb322d99e` |
| seed 1: verdict C / P, compare | `20261006T025842-ffb85080d4`, `20261006T025844-846405ddf1`, `20261006T025845-c10b1c61ff` |

Seed-1 payloads and training logs are in [`results/seed1/`](results/seed1/). For seed 1 set
`V016_AMP=bf16 V016_TAG=_b_s1 V016_SEED=1`.
