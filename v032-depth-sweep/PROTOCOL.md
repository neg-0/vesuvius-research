# V-032: does a depth offset alone explain why the stock canonical model loses to mrg20736-1um on PHercParis4?

> **Reproducibility note.** `code/` holds this experiment's driver only. It imports helper modules from earlier
> experiments (V-030, V-020, V-016, V-014) that are not published in this repository yet, so it will not run here as is.
> The windows, per-window scores and the pre-registered rule are all in `results/` and this folder, so the numbers can be
> re-derived and checked without them. Queue and machine references in the text are internal run logistics.

Registered 2026-10-09 (cloud, before any V-032 score). Public-safe (PHercParis4 and PHerc0139 are outside the First
Letters list). Driver: `v032_run.py` (imports V-031's driver; nothing is trained). Windows:
`results/v032-depth-sweep/windows.json`, committed before any sweep.

## Why

V-031 (FAIL, pooled 100 of 212 windows) and V-031b found that the stock canonical 2.4 µm model, zero-shot on 1.129 µm
renders, beats the organisers' `mrg20736-1um` map on PHerc1667 (76 of 107, but the maps there were made on the misplaced
published mesh, #1843) and loses on PHercParis4 (24 of 105 windows, mean 0.861 vs 0.920) and PHerc0139 (25 of 85, mean
0.801 vs 0.857). Placement is not the cause on those two: step 0 was PLACEMENT_OK on every segment. The remaining
candidates the Stage 2 toolkit separates are (a) the mesh sits off the inked layer in depth (`score_depth_sweep.py`,
villa #1912), and (b) the model itself does not read the new scan (`finetune_on_scan.py`). V-031b's single descriptive
depth column (canonical at −34 layers on 7 PHerc0139 windows) was mixed (it rose on 2, fell on 5), so it neither
supports nor excludes (a). This item asks (a) directly on PHercParis4, the scroll with the clearest loss.

## Question and hypothesis

H (depth explains): on PHercParis4 windows where the canonical model lost to `mrg20736-1um`, moving the read window by
up to ±32 layers (±72 µm) along the render normal closes the gap, and does so specifically for loss windows, not just
because the best of nine offsets is always at least as good as offset 0.

The test is an **oracle upper bound**: the best offset per window is chosen with the labels. So FAIL is a robust
negative (even a label-chosen offset within ±72 µm cannot recover the gap, so depth is not the cause and adaptation, not
re-seating, is the route). PASS is only a weak positive (depth *can* explain it) and would be followed by a separate item
that chooses the offset without labels.

## Windows (fixed in `windows.json`, chosen from V-031's committed `scores.json` before any sweep)

- **L, loss set (12):** per PHercParis4 segment (6), the 2 windows with the lowest canonical − mrg AUC in `scores.json`.
- **W, control set (12):** per PHercParis4 segment, the 2 windows with the highest canonical − mrg. Same size, same
  segments; it measures how much the best-of-nine selection gains by chance (and from label noise) on windows that are
  already fine.
- **D, descriptive set (7):** every window of PHerc0139 w044 (5) and w045 (2), the segments of V-031b's depth column and
  of #1912.

Selection is deterministic (`choose`, sorted by (difference, window name)); `select` prints SELECTION SAME against the
committed file. `windows.json` also stores each window's V-031 values and the SHA-256 of `scores.json`. All 31 windows
come from V-031's committed plans (same cells, canvas origin, labels, ds8 pixels).

## Scoring (GPU machine)

- **Render:** the window along the V-031 step-2 mesh (`mesh_for_step2`; PHercParis4: published B per step 0; PHerc0139:
  the 2.399 µm mesh carried through the published transform) on the 1.129 µm volume, level 1, oversample 1 (2.258 µm px
  and layer step), **127 layers** = V-031's 63 plus 32 on each side. The centre 63 are exactly V-031's 63-layer render
  (same layer positions; odd counts), so offset 0 must reproduce V-031's canonical AUC. CT at uncarried pixels is set
  to 0.
- **Canonical:** `infer_crop` (tile 256, stride 128, bf16) on the 62 layers at stack centre + offset, for offsets
  **−32, −24, −16, −8, 0, +8, +16, +24, +32** layers (negative = towards −normal, V-031b's convention; ±32 layers =
  ±72.3 µm; V-031b's −34 is covered to within one step of −32).
- **mrg20736-1um:** V-031's cropped map, scored on the same pixels.
- **Pixels:** V-031's ds8 sets (positives carried label ≥ 0.5, negatives < 0.05 within 2 mm of ink, CT-valid and
  carried), computed from the offset-0 layers, identical for every offset and for mrg. AUC by `march_bench.auc`.
  Offsets are not combined and no label search or shift search is used.

## Pre-registered rule (fixed before any sweep score)

Per window: gain = (max over offsets of canonical AUC) − (canonical AUC at offset 0) ≥ 0; the best offset is the argmax
(ties to the smaller |offset|, then the more negative). Gap closed = best-offset AUC ≥ mrg AUC − 0.03.

**PASS** (depth can explain the PHercParis4 losses) iff both:
1. the gap is closed on at least ⌈0.5 × 12⌉ = **6 of the 12 loss windows**; and
2. median gain(L) − median gain(W) ≥ **0.05**.

**FAIL** if all 24 windows are scored with defined AUCs and the above does not hold. **UNKNOWN** if any planned L or W
window is unscored (budget cap) or has an undefined AUC (under 50 positives or 50 negatives; none expected, V-031 had
all defined). The PHerc0139 set D is **descriptive only** (n = 7 < 10): no verdict.

## Recorded, not in the rule

- Per window: AUC at all nine offsets, best offset, gain, gap closed, V-031's offset-0 value and the difference (REPRO).
- Whether the best offsets of a segment agree (a mesh depth shift would give one offset per segment) or scatter (a
  per-window effect that no single mesh correction would fix), listed by segment in RESULT.md prose.
- D: how many windows have their best offset ≤ −24 (the #1912 prediction is −34), against V-031b's −34 column.
- Windows below the toolkit's `--auc-ok 0.75` at offset 0 and at the best offset.

## Stop and drop conditions

- `--check` failing, `SELECTION DIFFERENT`, or a window's `CARRY … DIFFERENT` / `STOPPED_AT_CARRY` lines: stop and report.
- **REPRO FAIL** (the driver raises it): offset-0 canonical AUC differs from V-031's by more than 0.005, or the ds8
  pixel counts differ, on any of the first 3 windows: stop and report the line. Expected: exactly equal (same layers,
  same inference).
- `run` refuses unless `windows.json` is committed and unchanged in git.
- Budget: 4 h of GPU-machine time; unscored planned windows make the verdict UNKNOWN.

## Budget and environment

31 windows: render 127 layers ~2 min, nine inferences ~0.22 min each, ~4 min per window (~2.1 h), plus the carry
recompute the render needs for each of the 8 segments (~5 min per PHercParis4 segment, ~2 min per PHerc0139 segment,
~0.5 h on CPU). `select` prints the estimate (~2.6 h). One queue item under the 4 h cap. Disk: one 127-layer stack at a
time (deleted after scoring) plus up to ~0.5 GB of carry per segment. Nothing is trained or posted.

## Outputs (`results/v032-depth-sweep/`)

`windows.json` (committed first), `sweep.json` (a row per window), `summary.json` (RULE lines, evaluation, rows),
`RESULT.md` (written by `rule`; prose below its `<!-- prose -->` marker). No maps or renders are committed (ink maps
stay in `V031_DATA`).

## What each outcome means for the toolkit

- **FAIL:** depth is not the main cause of the PHercParis4 losses; the model gap (the toolkit's adaptation branch) is
  what remains. A V-016-style fine-tune on PHercParis4 is then the next registered item (not queued: it needs its own
  protocol, and V-030 showed the recipe lowers AUC where the stock model already reads, so it would be gated on the
  stock AUC).
- **PASS:** a label-free depth chooser (maximising agreement with the neighbouring layers or the organisers' map) is the
  next item. The October form's depth-sweep claim (#1912) stays as it is.

## Desktop run

From the repo root (Python from `venv-gpu`; reuse the villa checkout of items 44 to 46):

```sh
V=workspaces/vesuvius/villa-diag-7a5ba5a          # rev-parse must print 7a5ba5a01f1ff7070918f2143143c11c6f7268c5
export V031_VILLA=$PWD/$V V031_DATA=$PWD/workspaces/vesuvius/v032/data \
  V031_SRC=$PWD/workspaces/vesuvius/v013gpu/xscan/src XSCAN_CACHE=$PWD/workspaces/vesuvius/v032/cache \
  CANON_CKPT=$PWD/workspaces/vesuvius/models/r152_3ddec_v2_l5_epoch13.ckpt \
  V032_RESULTS=$PWD/projects/vesuvius/results/v032-depth-sweep
P=$PWD/workspaces/vesuvius/venv-gpu/bin/python
R() { python3 tools/research.py run vesuvius --hypothesis V-032 --seed 0 --timeout 6600 \
  --cwd projects/vesuvius/experiments/v032 --input v032_run.py --input ../v031/v031_run.py \
  --input ../v030/v030_run.py --input ../v020/v020_scan_adapt.py --input ../v016/v016_cross_scan_ink.py \
  --input ../v014/v014_transform_ink.py --input $V031_SRC/render_crop.py --input $V031_SRC/render_tifxyz.py \
  --input $V031_SRC/infer_crop.py --input $V031_SRC/march_bench.py -- $P v032_run.py "$@"; }
$P projects/vesuvius/experiments/v032/v032_run.py check     # last line must be CHECK OK
R select                                                   # must print SELECTION SAME, then the ESTIMATE line
R run --limit 1                                            # first window; must not print REPRO FAIL
R run                                                      # resumable; rerun the same line if cut off at 2 h
R rule; R status                                           # RULE / DESCR / REPRO lines, summary.json, RESULT.md
rm -rf $V031_DATA/*/carry $V031_DATA/*/labels.zarr         # after the item; keep sweep.json and the ink maps
```

`check` runs V-031's `check` (needs `V030_SRC` set from `V031_SRC`, as for items 44 to 46) and also verifies that
`windows.json` equals the selection rebuilt from the committed V-031 `scores.json`.
