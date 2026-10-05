# V-016 — Cross-scan ink: adapt the 2.4 µm model to the PHerc1667 1.129 µm scan

**Status: REGISTERED 2026-10-05T03:55Z**, before any V-016 render, baseline or training run. The design below and the driver are unchanged from the draft at commit bf9e49c. Run on one RTX 3080. The R arm (forgetting) and the P control are not part of this request.

## Why

V-014 showed two things on the 6 labelled PHerc1667 windows:

- The canonical 2.4 µm model reads 0.87–0.99 AUC on the 2.399 µm scan but only 0.49–0.63 on the 59 keV
  1.129 µm scan. This holds along both the published mesh and the refit mesh.
- Brightness matching does not close the gap.

The community digest (2026-10-04) lists ink models that work across scan resolutions as an open gap. The refit
mesh (C) puts the 1.129 µm CT on the labelled papyrus (V-014 placement PASS), so labels drawn on the 2.399 µm
volume can supervise the 1.129 µm scan along C.

## Hypothesis

Fine-tuning `scrollprize/ink_canonical_2um` on 1.129 µm renders along C lets it read ink on segments it has not
seen.

- **Folds:** 3, leaving one pair out each time: (w013, w018), (w023, w028), (w029, w031). Each fold trains on the
  other 4 segments' labelled areas, excluding their V-014 windows, and is tested on the V-014 windows of the
  held-out pair.
- **Primary:** mean held-out AUC at the centre window. It uses the same renderer, `eval_map` and ±48 ds8 px label
  shift as V-014.
- **Baseline:** the unmodified canonical model on the same held-out stacks (zero-shot), computed in the same run. It
  should reproduce V-014's C numbers (0.51–0.63), because the renderer and layers are identical.
- **Pass:** mean held-out AUC ≥ 0.75, and fine-tuned > zero-shot on at least 5 of 6 segments.
- **Fail:** mean < 0.75, or at most 4 of 6 segments above zero-shot.
- **UNKNOWN:** a fold does not finish within budget.

## Secondary (descriptive)

- **P-trained control:** the same fine-tune with renders along P, tested along P. If C-trained beats P-trained,
  that is ink evidence for #1843 that V-014 could not give.
- **Forgetting:** AUC of each fine-tuned model on R (2.399 µm). This checks whether the model still reads 2.4 µm.

## Limits

- 6 segments from one scroll and one scan.
- The labels were drawn on 2.399 µm, so residual misregistration of a few µm limits what any 1.129 µm model can
  score.
- Training choices are fixed here and are not tuned on test windows. Any change after a held-out window is scored
  makes the result exploratory.

## Fixed design (script `experiments/v016/v016_cross_scan_ink.py`)

- **Test stacks:** the V-014 windows (60 × 170 cells), rendered with 63 layers. Layers 0–61 equal V-014's offset-0
  window.
- **Training patches:** up to 4 per training segment, each 40 × 80 cells. They are the densest labelled patches
  that lie inside the 1.129 µm volume for both meshes and stay at least 10 cells clear of that segment's test window.
  Labels are full resolution, so 20 px per cell matches the render.
- **Loss:** BCE on the model's 1/4-resolution logits against 4×-pooled labels. It is masked to CT-valid pixels
  within 2 mm of ink, the same negative rule as `eval_map`.
- **Sampling:** 256 px tiles with at least 2% ink and at least 50% supervised pixels. Augmentation is rot90, flips
  and ±10% contrast with ±0.05 brightness.
- **Optimiser:** full fine-tune from the canonical checkpoint with frozen BatchNorm statistics. AdamW, lr 2e-5,
  weight decay 1e-4, 100-step warm-up then cosine, 2000 iterations, batch 2 (two accumulated micro-batches of 1), fp16 autocast, gradient clipping at
  1.0, seed 0. Activation checkpointing is used on the backbone.
- **Evaluation:** `infer_crop` tiling (256, stride 128), then `eval_map`, as in V-014.

## Budget and environment

- One RTX 3080 (10 GB).
- CT comes from the public bucket. The cache is deleted after the run.
- Nothing is posted or submitted. This folder is the public write-up.

## Amendment 1 (2026-10-05 06:45Z, before any informative held-out score)

Fold 0 of the driver as registered (train run `20261005T060447-305b165b7f`, eval `20261005T062717-18a8f589d5`)
did not produce a model: the loss became NaN from iteration 361, 502 of 978 weight tensors ended non-finite, and
the fine-tuned ink maps for w013 and w018 are all NaN (scored as AUC 0.0000, which carries no information about
those windows). The artifacts are kept in the local `v016/data/ckpt_C_fold0_run1_nan/`.

Cause: the training forward bypassed `RegressionModel.forward` and so skipped the model's
input `normalization` layer (BatchNorm3d), which inference applies. On a w028 training tile the two forwards differ
by 0.25 logits; with the layer added they are identical (difference 0.0). The model was therefore trained on inputs
it is not evaluated on.

Changes to the driver, made before rerunning fold 0 and applied to all folds:
- the training forward applies `net.normalization`, as inference does;
- a micro-batch with a non-finite loss is dropped, and the optimiser is not stepped on non-finite gradients;
  both are counted in `train_log.json` as `skipped_nonfinite`;
- training refuses to save non-finite weights.

Nothing else changes: folds, patches, loss, optimiser, schedule, iterations, seed, evaluation and the pass rule
are as registered. The zero-shot numbers seen so far (w013 0.5143, w018 0.5345, w028 0.6268) equal V-014's.
