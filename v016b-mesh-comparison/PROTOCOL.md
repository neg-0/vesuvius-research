# V-016b — Does the refit mesh read ink better than the published one? (C vs P, one stable recipe)

**Status: REGISTERED 2026-10-05**, before any V-016b training run. It follows the V-016 arm P control, which was
not a valid comparison (`results/fp16-P-control-note.md`): 2 of 3 P folds diverged under fp16, and the one valid
P fold split one each way against C (w023 P 0.722 vs C 0.843; w028 P 0.892 vs C 0.852).

## Question

#1843 says the published PHerc1667 1.129 µm transform is wrong. V-014 showed the refit mesh (C) places the CT on
the labelled papyrus and the published mesh (P) does not, by CT similarity. V-016b asks whether that shows up in
ink: with the same fine-tune on both, does a model trained along C read held-out ink better than one trained along P?

## Design

Identical to V-016 (folds, patches, loss, sampling, optimiser, schedule, 2000 iterations, seed, evaluation), with
one change applied to **both** arms: training and inference autocast in **bfloat16** instead of float16
(`V016_AMP=bf16`; no gradient scaler). bfloat16 has float32's exponent range, which removes the fp16 decoder
overflow that killed P folds 0 and 2. Learning rate stays 2e-5, so C here is a direct rerun of the V-016 recipe.
Outputs go to separate checkpoint folders and payload names (`V016_TAG=_b`); nothing from V-016 is reused except the
renders.

## Rule (script `v016_cross_scan_ink.py compare`)

- **Valid** only if all 6 fold models train with at most 1% skipped micro-batches and every held-out AUC is above 0
  (no NaN maps). Otherwise **UNKNOWN**, and the result says which fold failed.
- **PASS** (ink evidence for #1843): C fine-tuned AUC above P fine-tuned AUC on at least 5 of 6 held-out segments and
  mean (C − P) ≥ 0.03.
- **FAIL** otherwise: no ink evidence that the refit mesh beats the published one at this resolution.

Secondary, descriptive: each arm's own V-016 verdict (mean ≥ 0.75 and above its zero-shot on ≥ 5/6), and whether
bf16 C reproduces V-016's C (mean 0.864).

## Limits

Six segments from one scroll and one scan; labels drawn on 2.399 µm. A fine-tuned model can partly learn around a
misplacement, so FAIL does not mean the meshes are equivalent, only that ink does not separate them here.

## Budget

One RTX 3080, about 3.5 h (6 trainings plus evals). Renders are already on disk. $0.

## Amendment 1 (2026-10-05, after the seed-0 verdict): seed replication

The seed-0 verdict above stands as the registered result. Both arms are rerun with seed 1, using the same bf16 recipe
(`V016_SEED=1 V016_TAG=_b_s1`). The rerun is descriptive and is reported next to seed 0. It changes nothing in the
registered verdict, with one exception: if seed 1's compare is not PASS, the public write-up must say the gap did not
replicate and give both seeds' numbers.
