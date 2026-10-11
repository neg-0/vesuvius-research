# B-001: real-scan distillation lifts the 9 µm ink model on unseen scrolls

**Question.** The organisers' 9 µm ink model (`scrollprize/ink_9um`, `hybrid_3d2d-seed42/step-075000`) learned
mostly from 2.4 µm renders pooled down to ~9.6 µm, and saw no real 8.64 µm (116 keV) data. Can it read better on
real low-resolution scans of scrolls it never saw, if we fine-tune it on real 9.36 µm and 8.64 µm surface volumes
of *other* scrolls, using the canonical 2.4 µm model's published prediction of the same segment as the teacher?

**Answer.** Yes. Preregistered, held out by scroll, scored against human ink labels:

| study | what | result |
| --- | --- | --- |
| B-001 | 3 training scrolls (PHerc0139, 0814, 0343P), 1,530 teacher-labelled tiles, seed 4500 | mean AUC 0.711 → 0.803, gain +0.0914, better on 44 of 49 windows: **PASS** |
| B-001r | same recipe, seed 4501 | +0.1035, 46 of 49, per-window gain correlation with B-001 0.94: **replicated** |
| B-001r | same tiles and labels, but simulated 9 µm input (2.4 µm surface volume pooled to the 9 µm grid) | real input better by +0.035 on 39 of 49: **real scans matter** |
| B-001s | all six paired scrolls, leave-one-scroll-out, 5,717 tiles | +0.0987 vs stock, but +0.001 over B-001: **no gain from more data** |
| controls | the fine-tuned models on layer-shuffled input | pooled AUC 0.54-0.58 (no depth order, no read) |

- Test: 49 windows (520 × 520 px, 4.9 mm) on three held-out scrolls, PHerc0841 (25), PHerc0009B (12, 8.64 µm) and
  PHerc0500P2 (12). Every held-out scroll improves.
- The gain is largest on the 8.64 µm scan type (PHerc0009B, +0.10 to +0.12). Four Grand Prize scrolls are 8.64 µm
  scans.
- For reference, the 2.4 µm teacher scores 0.909 on the same pixels (post hoc). The student closes about 46% of the
  gap between the stock 9 µm model and the 2.4 µm model, and more paired data does not close more of it.
- Compute: one rented RTX 4090 per study, about $5.40 in total.

## Side findings

- **Label placement.** Moving labels from a high-resolution scan to the low-resolution render of the same segment
  by grid index (low-res px (y, x) ← high-res px (k·y, k·x)) misplaces them by up to ~100-300 µm, and the error varies
  across a segment. A CT-only per-segment affine fit corrects it: NCC rises from ~0.1-0.7 to ~0.85-0.94, and 56 of 70
  segments fit (most PHerc0814 segments do not). At these window sizes the effect on AUC is small (0.709 vs 0.711),
  so this is a data-quality note, not an explanation of earlier 9 µm results.
- **First look at PHerc0800 (8.64 µm Grand Prize scroll).** On 4 public segments the fine-tuned model arranges its
  output into horizontal rows; the stock model does not. No letters are legible. A layer-shuffle control came out
  MIXED (its map correlates 0.37 with its own shuffled-input map, against 0.02-0.34 on the labelled test windows), so
  the rows may be fibre texture. **They are not claimed as text.**

## Files

- [`PROTOCOL.md`](PROTOCOL.md): the three registered specs (data, alignment correction, recipe, rules), committed
  before any model output.
- [`RESULT.md`](RESULT.md): rule lines, per-window AUC tables, the teacher comparison and the PHerc0800 first look.
- [`code/`](code/): the drivers that ran, and how to run them.
- `results/b001`, `results/b001r`, `results/b001s`: segment lists, CT placement fits, tile and window lists, packed
  teacher and human labels, per-window scores, training logs.

The fine-tuned checkpoints (273 MB each, sha256 in `RESULT.md`) are not in this repository; they are available on
request.

## Provenance

- Data: Vesuvius Challenge open-data bucket, read in place (surface volumes, meshes, human `inklabels` /
  `supervision`, and the canonical 2.4 µm model's published `new_canon_autoresearch_recipe` predictions). No raw CT
  is redistributed here.
- Base model: [`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um) `hybrid_3d2d-seed42/step-075000`
  (sha256 `e635558a…62a9cab`), trained and run through
  [ScrollPrize/villa](https://github.com/ScrollPrize/villa) `vesuvius/src/vesuvius/ink_detection` at `e0bbb8b`.
