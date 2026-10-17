# V-033: can a label-free statistic of the model's own ink maps choose the depth offset per window?

Registered 2026-10-09 (cloud), after V-032's results and before any V-033 statistic was computed. Public-safe. CPU only,
no GPU, no network, nothing rendered or inferred: it re-reads the nine ink maps per window that V-032 kept on the
desktop. Driver: `v033_chooser.py` (numpy only). Output: `results/v033-depth-chooser/`.

## Why (what V-032 showed, and what it did not)

V-032 (GPU-QUEUE item 47, FAIL on its registered rule): on the 12 PHercParis4 loss windows the canonical model, read at
the best of nine depth offsets (label-chosen), rises from mean AUC 0.7227 to 0.8358, against 0.9332 for the organisers'
`mrg20736-1um` map; on the 12 control windows it rises from 0.8966 to 0.9256. The gap closed on only 2 of 12 loss
windows (needed 6), but the loss set gains 0.100 more than the control (needed 0.05), so depth matters for the losses
without explaining all of them. On the 7 PHerc0139 w044/w045 windows the oracle reaches 0.8987 (mrg 0.8784).

Post-hoc analysis of V-032's committed `sweep.json` (cloud, 2026-10-09, exploratory, not registered) shows the gain is
**per window, not a constant to calibrate**:
- The best offsets scatter inside each segment (from −8 to +24 layers; e.g. the two loss windows of segment
  `P4_20231210121321` are best at −8 and +24).
- A leave-one-window-out segment offset (choose the offset with the best mean AUC over the segment's other windows,
  apply it to the held-out one) gains +0.014 on loss windows and +0.000 on controls. One offset for the whole scroll
  (leave-one-out) loses 0.040 and 0.013. A fixed +16 gains +0.019 on loss windows and loses 0.017 on controls.
- The loss windows' mean AUC is flat from −8 to +24 layers (0.73 to 0.74), where the controls peak sharply at 0 (0.897
  at 0, 0.878 at −8, 0.880 at +16): the loss windows read ink at no single clear depth.

So a mesh-wide depth correction does not work; only a per-window choice would. Whether that choice can be made
**without labels** is the practical question, and the one an unlabelled scroll needs answered. Context from the
community (Discord, 2026-09-25/29, unverified): a fixed label-free depth filter gave about +0.03 on PHerc0841 and +0.01
to +0.02 on held-out scrolls for the organisers' 9 µm model.

## Question and hypothesis

H: choosing, for each window, the depth offset that maximises a **label-free statistic of the canonical model's own ink
map** recovers at least half of the oracle gain on the PHercParis4 loss windows without hurting the control windows.

## Choosers (fixed now; computed only from the ink maps and the carried-coverage mask)

For each window and offset o in {−32, −24, −16, −8, 0, +8, +16, +24, +32} layers (2.258 µm each): M_o = the 8 × 8
block mean (ds8, the scoring resolution) of V-032's saved ink map; V = blocks whose 64 pixels are all carried (the
carried mask is mesh coverage, not label values). Statistics over M_o[V]:
- **std (primary):** the standard deviation of M_o over V. A read at the right depth should separate ink from papyrus
  more sharply.
- **top5:** the mean of the highest 5% of M_o over V.
- **mean:** the mean of M_o over V.

A chooser picks the offset with the highest value (ties to the smaller |offset|, then the more negative). A window with
fewer than 100 fully carried blocks keeps offset 0 (recorded). `stats` writes the choices to `stats.json` **without
opening sweep.json, any label, any AUC or any pixel set**; `rule` refuses to run unless `stats.json` is committed and
unchanged in git, so the choices are fixed before any AUC is joined to them.

## Pre-registered rule (primary chooser `std` only, on V-032's windows and AUCs)

Gain of a window = AUC at the chosen offset − AUC at offset 0, from V-032's committed `sweep.json` (the AUC at every
offset is already stored there). With L = the 12 PHercParis4 loss windows and W = the 12 control windows of V-032's
`windows.json`, and the oracle gain on L = mean(best-offset AUC − AUC at 0) = about +0.113:

**PASS** iff all three hold:
1. mean gain on L ≥ 0.5 × the oracle gain on L (about +0.057);
2. mean gain on W ≥ −0.01 (do no harm);
3. the chosen offset beats offset 0 on at least 8 of the 12 loss windows.

**FAIL** otherwise. **UNKNOWN** if any of the 24 maps is missing or an AUC is undefined (then `stats` or `rule` stops).

## Recorded, not in the rule

- The same three numbers for `top5` and `mean` (the multiplicity is why the rule names one primary).
- The chance reference: the mean gain of a random offset (mean over the nine offsets) on L, W and D.
- On the 7 PHerc0139 windows (D) and as a descriptive column: mean AUC at 0, at the primary choice, at the oracle, and
  the organisers' map. No verdict (n = 7).
- How often each chooser equals the oracle offset, and the chosen offsets by set.
- Reference points from the post-hoc analysis above: a segment-level leave-one-out offset (+0.014 on L), a fixed +16
  (+0.019 on L, −0.017 on W). A chooser that does not beat these adds nothing.

## What each outcome means

- **PASS:** a label-free per-window depth choice recovers real AUC on a scroll that needs it. The next items are to
  re-score all 105 PHercParis4 windows with it (V-031's comparison) and to fix the render depth of any fine-tune on it.
- **FAIL:** depth is a per-window effect that simple self-statistics cannot find. Then the toolkit's depth sweep stays a
  label-assisted diagnostic (as in the October form), and the next GPU item is the gated PHercParis4 fine-tune (which
  must be specced with the depth question in mind, since the fine-tune would learn on renders whose depth is locally off).

## Budget and environment

CPU only, a few minutes (31 windows × 9 maps of ~400 × 800). Needs the maps and carried masks left by V-032's run in
`V031_DATA` of item 47 (`workspaces/vesuvius/v032/data`), and the committed `results/v032-depth-sweep/{windows,sweep}.json`.
Stop and report if `check` fails or a map is missing; do not recompute maps.

## Desktop run

From the repo root (any Python 3 with numpy):

```sh
export V033_MAPS=$PWD/workspaces/vesuvius/v032/data \
  V032_RESULTS=$PWD/projects/vesuvius/results/v032-depth-sweep \
  V033_RESULTS=$PWD/projects/vesuvius/results/v033-depth-chooser
R() { python3 tools/research.py run vesuvius --hypothesis V-033 --seed 0 --timeout 1800 \
  --cwd projects/vesuvius/experiments/v033 --input v033_chooser.py -- python3 v033_chooser.py "$@"; }
python3 projects/vesuvius/experiments/v033/v033_chooser.py check     # last line must be CHECK OK
R stats                                                              # 31 STATS lines; writes stats.json
git add projects/vesuvius/results/v033-depth-chooser/stats.json && git commit -m "V-033 stats: chosen offsets"   # before rule
R rule                                                               # RULE / DESCR lines, summary.json, RESULT.md
```
