# V-031: canonical 2.4 µm model zero-shot vs the organisers' mrg20736-1um map on 1.129 µm scans

> **Reproducibility note.** `code/` holds this experiment's driver only. It imports helper modules from earlier
> experiments (V-030, V-020, V-016, V-014) that are not published in this repository yet, so it will not run here as is.
> The windows, per-window scores and the pre-registered rule are all in `results/` and this folder, so the numbers can be
> re-derived and checked without them. Queue and machine references in the text are internal run logistics.

**Status: REGISTERED 2026-10-08T02:10Z**, before any V-031 score: no ink map of the canonical model, no
`mrg20736-1um` value and no AUC has been computed for V-031. Before this text was committed, the driver's step 0 ran on
one segment (`1667_w031`) as a test (placement statistic only, CT vs CT; result in `step0/1667_w031.json`). Requested
by the project's roadmap. Public-safe: PHerc1667 and PHercParis4 are not First Letters targets. Desktop RTX 3080,
GPU-QUEUE item 44 (or 44/45 when split per scroll; see "Budget").

## Why

V-030 (PHerc0814, one segment) failed its own rule, but post hoc the stock canonical model, zero-shot on 1.129 µm
renders at 2.258 µm along the 1.129 µm mesh, beat the organisers' published `mrg20736-1um` prediction on 10 of 11
windows (mean AUC 0.963 vs 0.907). That was not a registered comparison and rests on one segment. V-031 registers it
on every other labelled segment that has a published `mrg20736-1um` prediction on the roadmap thread's list (bucket
scan of 2026-10-08).

## Question and hypothesis

On 1.129 µm scans, does the stock 2.4 µm canonical model (`scrollprize/ink_canonical_2um`,
`r152_3ddec_v2_l5_epoch13.ckpt`, zero-shot, no training), rendered at 2.258 µm along the 1.129 µm mesh as in V-030,
read held-out-style labelled windows as well as the organisers' published `mrg20736-1um` prediction of the same
segment, scored on identical pixels with no label search?

Hypothesis: yes. Canonical AUC ≥ mrg20736-1um AUC on at least 70% of scored windows and a mean paired difference
(canonical − mrg) ≥ 0, pooled over both scrolls.

## Data (open-data bucket `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/`)

Exact paths per segment (meshes, label zarr, mrg map, L1 surface volume, shapes, label date `20260918` everywhere) are
in [`results/v031-canonical-vs-mrg-1um/segments.json`](../../results/v031-canonical-vs-mrg-1um/segments.json),
written by `v031_run.py resolve` from the bucket listing. Pattern, with `SEG = <scroll>/segments/<segment>/`:

| what | PHerc1667 | PHercParis4 |
| --- | --- | --- |
| labelled scan (A) | `PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr` | `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr` |
| 1.129 µm scan (B) | `PHerc1667/volumes/20260323082859-1.129um-0.2m-59keV-masked.zarr` | `PHercParis4/volumes/20260608103018-1.129um-0.2m-78keV-masked.zarr` |
| transform (B → A voxels) | `<B>/transform.json` (6 landmarks; fixed volume `SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_1667_TA_0001_masked`) | `<B>/transform.json` (16 landmarks; fixed volume `2.4um_PHerc-Paris4_masked`, the 78 keV scan the labels are on) |
| mesh A | `SEG/mesh/<id>-on-20251217075048-2.399um.tifxyz/` | `SEG/mesh/<id>-on-20260411134726-2.4um.tifxyz/` |
| mesh B | `SEG/mesh/<id>-on-20260323082859-1.129um.tifxyz/` | `SEG/mesh/<id>-on-20260608103018-1.129um.tifxyz/` |
| labels (level 0, A canvas) | `SEG/ink-labels/2.399um-volume-20251217075048/20260918/inklabels.zarr/0/` | `SEG/ink-labels/2.4um-volume-20260411134726/20260918/inklabels.zarr/0/` |
| mrg20736-1um (L1 canvas = 10 × mesh B grid, uint8, tiled) | `SEG/ink-detection/PHerc1667-<id>-1.129um-0.22m-59keV-volume-20260323082859-L1-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif` | `SEG/ink-detection/PHercParis4-<id>-1.129um-0.23m-78keV-volume-20260608103018-L1-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif` |
| coverage (where B has data) | `SEG/surface-volumes/1.129um-0.22m-59keV-volume-20260323082859-L1.zarr/5/` | `SEG/surface-volumes/1.129um-0.23m-78keV-volume-20260608103018-L1.zarr/5/` |

Segments (12): PHerc1667 `20240304141531-w013_20240304141531_flatboi`, `20240304144031-w018_20240304144031_flatboi`,
`20240304161941-w023_20240304161941_flatboi`, `20251208130119-w028_20251208130119156_flatboi`,
`20251212185248-w029_20251212185248662_flatboi`, `20251223230000-w031_2025122323_flatboi` (short names
`1667_w013` … `1667_w031`); PHercParis4 `20230702185753`, `20231007101619`, `20231012184424`, `20231031143852`,
`20231106155351`, `20231210121321` (short names `P4_<id>`). Each has exactly one label set, one 1.129 µm mesh, one
`mrg20736-1um` map whose shape is 10 × its mesh-B grid, and one L1 surface volume (`resolve`, all `ok`).

**Not included: PHerc0139.** A full bucket scan for this registration (2026-10-08, every scroll) found 8 more labelled
segments with an `mrg20736-1um` map on a 1.129 µm scan: PHerc0139 w030, w035, w039, w040, w041, w043, w044, w045
(labels on 2.399 µm `20260102150214`, 1.129 µm scan `20260413113053`). The roadmap list did not have them; V-031
keeps to that list. They are added as the separate extension **V-031b** (section below, registered 2026-10-08 at the
roadmap thread's request); V-031's registered verdict does not change.

Code: `v031_run.py` (this folder) imports `../v030/v030_run.py` unchanged for the placement statistic, greedy crops,
the 3D carry (`carry_planes`, `local_scale`), `render_crop` call, pixel sets, AUC and bf16 inference, and V-020's
window chooser through it. Render tools from cross-scan-ink-transfer `df9db2f` (render_crop.py, render_tifxyz.py,
infer_crop.py, march_bench.py); villa `claude/project-thread-s6f6nh` at **7a5ba5a** for `optimized_inference`.

## Step 0: placement (CPU; V-030 step 0, per segment)

V-030's statistic and threshold, unchanged: 6 crops of 8 × 16 mesh-A cells chosen greedily by labelled-ink density,
each moved to mesh B through the published transform; B crop 16 × 32 B cells; reference A = the same B vertices
mapped by `M_pub` into the labelled volume, level 0; B rendered on the 1.129 µm volume, level 1, 61 layers, normals
σ = 2 B cells; NCC0 of the central 21 layers at zero shift (12 px margin), best NCC within ±20 layers / ±12 px
reported alongside. Crops use only A cells within one A cell (3D) of a "usable" B vertex: valid, inside the 1.129 µm
level-0 volume along the step-2 mesh, and where the 1.129 µm scan has data (level 5 of the organisers' L1 surface
volume, any layer non-zero; V-030 Amendment 1).

- **PHerc1667: the refit is required** (villa #1843; V-014 addendum, V-016, V-016b): step 2 renders along
  B' = `M_lsq⁻¹ · M_pub · B`, with `M_lsq` the least-squares fit to the 6 published landmarks (RMS 0.93 px vs 51.4 px
  for the published matrix). The segment is kept (**PLACEMENT_REFIT**) if NCC0(B') ≥ 0.4 on ≥ 5 of 6 crops, else
  **PLACEMENT_FAIL** (dropped). NCC0 of the published B is recorded alongside.
- **PHercParis4: V-030's rule.** **PLACEMENT_OK** if NCC0(B) ≥ 0.4 on ≥ 5/6; otherwise refit as above,
  **PLACEMENT_REFIT** if NCC0(B') ≥ 0.4 on ≥ 5/6 and NCC0(B') > NCC0(B) on ≥ 5/6; else **PLACEMENT_FAIL** (dropped).
- Descriptive: landmark RMS of both matrices, leave-one-out residuals, singular values.

## Step 1: carry the labels onto the 1.129 µm L1 canvas in 3D (CPU; V-030 step 1, per segment)

V-030's carry, unchanged in its maths (`v030_run.carry_planes`): for each B vertex x, q = `M_pub` · x (correct for B'
too, since `M_lsq` · B' = `M_pub` · B); nearest mesh-A vertex, local Jacobian solve for sub-cell (u, v); carried if
|δ| ≤ 1 cell and the normal residual ≤ 8 voxels; pixels by the renderer's corner-aligned upsample (10 px per B cell),
label at A pixel (round(20u), round(20v)).

Change needed for size (these meshes are 13–63 M vertices, V-030's was 1.3 M): the carry is computed only on the
**ROI**, the bounding box (+2 cells) of the usable B vertices, and per-cell ink and carried masks are computed in tiles
without holding the full canvas. Pixels outside the ROI cannot be rendered (no 1.129 µm data), so no window is lost.

- **CARRY_OK** if (1) median normal residual ≤ 4 voxels; (2) carried ink on usable, fully carried B cells ≥ 50% of
  the **in-scope** labelled ink on the A canvas (A cells whose vertices lie within one A cell, in 3D, of a usable B
  vertex; V-030 compared with the whole segment, which here would count ink the 1.129 µm scan never imaged); (3) the
  median local scale of the carry is within ±10% of 1.129 / um_A. Otherwise **CARRY_FAIL** (dropped).

## Windows (CPU, labels and mesh only; fixed in `windows.json` before any score)

V-030's held-out windows (Amendment 2), per kept segment, no training patches (nothing is trained): windows of
40 × 80 B cells (400 × 800 px, 0.90 × 1.81 mm); a cell counts when its 4 vertices are usable and all its pixels
are carried; 3 z bands per segment at the ink-weighted terciles of the step-2 mesh z over those cells (rounded to 10
voxels); per band, windows greedy by carried-ink density without overlap, all in-volume vertices in the band, ≥ 50%
of cells counted, up to 6, stopping when the next density is below 20% of the band's first or below 2%. So up to 18
windows per segment.

## Scoring (GPU machine)

- **Render:** each window along the step-2 mesh (B' for PHerc1667, B or B' for PHercParis4 per step 0) on the
  1.129 µm volume, level 1, oversample 1 (2.258 µm px and layer step), 63 layers (the model reads the centre 62),
  normals σ = 2 B cells. CT at uncarried pixels is set to 0. This is the organisers' L1 canvas, so the `mrg20736-1um`
  map is read on the same pixels.
- **Canonical:** `infer_crop` (tile 256, stride 128, bf16), as V-030's zero-shot arm.
- **mrg20736-1um:** the published map, uint8 / 255, cropped at the window's canvas origin (read tile by tile from the
  bucket).
- **Pixels (identical for both maps):** ds8; positives carried label ≥ 0.5, negatives < 0.05 within 2 mm of ink;
  CT-valid and carried only (V-030 `pixel_sets`). Offset 0, no label search. AUC by `march_bench.auc` (undefined
  under 50 positives or 50 negatives).

## Pre-registered rule (fixed before any score)

A window is **scored** when both AUCs are defined. For a set of scored windows (n):

- **PASS** if canonical AUC ≥ mrg20736-1um AUC on at least ⌈0.7 n⌉ windows **and** the mean paired difference
  (canonical − mrg) ≥ 0.
- **FAIL** otherwise.
- **UNKNOWN** if n < 10, or if any planned window of that set has not been scored (budget cap reached).

The rule is evaluated **pooled over both scrolls** and **per scroll**. **The pooled result is the verdict.** The
per-scroll results are reported alongside and do not change it. If every PHercParis4 segment is dropped at step 0 or
1, the pooled set is PHerc1667 alone and RESULT.md says so (and vice versa).

## Recorded, not in the rule

- Windows below `--auc-ok 0.75` (villa `scan_diagnosis`'s threshold, where the tool would advise adapting): listed for
  the canonical map and for `mrg20736-1um`.
- **Placement caveat, PHerc1667.** The organisers' `mrg20736-1um` maps were made along the published mesh B, which
  V-014/V-016b/#1843 found off the labelled papyrus (V-014 window medians 38–411 µm). So on PHerc1667 the comparison
  mixes model and placement. To separate them, an extra descriptive arm renders the 2 densest windows of each z band
  of each PHerc1667 segment (up to 36) along the published B as well and scores the canonical model on it, on the
  main arm's pixels (`canonical_pubmesh`). It runs after the main arm, within the 4 h cap.
- Neither model is guaranteed held out on these segments (the organisers' models may have trained on them); the
  windows are held-out-style, as in V-030.
- Per-window positive/negative pixel counts, step-0 crops and best shifts, carry residuals, ROI, in-scope ink.

## Stop and drop conditions

- A segment is **dropped**, with the reason recorded, if `resolve` finds a path missing, step 0 ends in
  PLACEMENT_FAIL (`step0/<seg>.json`), step 1 ends in CARRY_FAIL (`carry/<seg>.json`), or it yields no window.
- `--check` failing, `CARRY DIFFERENT` or `PLAN DIFFERENT` on the desktop: stop and report the line.
- `render`/`score` refuse to run unless `windows.json` is committed and unchanged in git.
- Budget: the GPU-machine stages of a queue item stop at 4 h; unscored planned windows make that scroll UNKNOWN.

## Budget and environment

- Cap 4 h per queue item on the RTX 3080. `plan` prints the estimate (render ~1 min and score ~0.25 min per window,
  from V-030's 0.75 min per 400 × 800 render); if the total exceeds 4 h the work is one queue item per scroll.
- Steps 0, 1 and the plan run in the cloud (`v031_run.py cloud`), reading meshes, label shards and CT crops into
  memory; their JSON outputs and `windows.json` are committed. The desktop recomputes the carry (needed for window
  labels) and checks `CARRY SAME` and `PLAN SAME`.
- No training. Nothing is posted. Public-safe.

## Outputs (`results/v031-canonical-vs-mrg-1um/`)

`segments.json` (resolved paths), `step0/<seg>.json`, `carry/<seg>.json`, `windows.json` (committed before any score),
`scores.json` (per window: canonical, mrg20736-1um, canonical_pubmesh for PHerc1667, pos/neg ds8 px),
`summary.json` (RULE and VERDICT lines, per scroll and pooled, below-0.75 lists), `RESULT.md` (written by `rule`,
prose added below its marker).

## Desktop run

From the repo root (Python from `venv-gpu`; reuse item 42's villa checkout):

```sh
V=workspaces/vesuvius/villa-diag-7a5ba5a          # rev-parse must print 7a5ba5a01f1ff7070918f2143143c11c6f7268c5
export V031_VILLA=$PWD/$V V031_DATA=$PWD/workspaces/vesuvius/v031/data \
  V031_SRC=$PWD/workspaces/vesuvius/v013gpu/xscan/src XSCAN_CACHE=$PWD/workspaces/vesuvius/v031/cache \
  CANON_CKPT=$PWD/workspaces/vesuvius/models/r152_3ddec_v2_l5_epoch13.ckpt \
  V031_RESULTS=$PWD/projects/vesuvius/results/v031-canonical-vs-mrg-1um
P=$PWD/workspaces/vesuvius/venv-gpu/bin/python
R() { python3 tools/research.py run vesuvius --hypothesis V-031 --seed 0 --timeout 6600 \
  --cwd projects/vesuvius/experiments/v031 --input v031_run.py --input ../v030/v030_run.py \
  --input ../v020/v020_scan_adapt.py --input ../v016/v016_cross_scan_ink.py --input ../v014/v014_transform_ink.py \
  --input $V031_SRC/render_crop.py --input $V031_SRC/render_tifxyz.py --input $V031_SRC/infer_crop.py \
  --input $V031_SRC/march_bench.py -- $P v031_run.py "$@"; }
$P projects/vesuvius/experiments/v031/v031_run.py --check     # last line must be CHECK OK
S=PHerc1667                       # the queue item names the scroll (or both, one after the other)
R step0 $S                        # prints the committed PLACEMENT lines (no recompute)
R carry $S                        # recomputes; every kept segment must print CARRY SAME and CARRY_OK
R plan $S                         # must print PLAN SAME and the ESTIMATE lines
R render --scroll $S              # main arm; resumable; rerun the same line if cut off
R score --scroll $S               # canonical vs mrg20736-1um per window; deletes each window's stack after
R render --scroll $S --extra; R score --scroll $S --extra   # PHerc1667 only, descriptive, only if under ~3 h so far
R rule; R status                  # RULE / VERDICT lines, summary.json, RESULT.md
rm -rf $V031_DATA/*/carry $V031_DATA/*/labels.zarr            # after the item; keep scores and maps
```

## Amendment 1 (2026-10-08T04:45Z, after steps 0, 1 and the plan, before any step-2 render or score)

- **Extra arm limited.** All 12 segments gave 18 windows (216 in all), so the full published-mesh arm on PHerc1667
  would have taken the PHerc1667 item past 4 h. The descriptive `canonical_pubmesh` arm now covers only the 2 densest
  windows of each z band of each PHerc1667 segment (36 windows, ~0.8 h). The registration text above (Recorded, not in
  the rule) was edited to say so; the rule is unchanged.
- **Memory and disk only, no change in result:** the carry runs `v030_run.carry_vertices`' per-vertex maths in row
  chunks of mesh B (selftest: identical to `v030_run.carry_planes`), and windows are rendered from per-window
  sub-meshes (window + 12 cells, more than render_crop's 8-cell normal margin) instead of whole ROI meshes. The first
  cloud attempt at `P4_20231007101619`'s carry was killed for memory (15 GB container) before this change.

## Steps 0 and 1 and the plan: results (cloud CPU, 2026-10-08 01:42–04:35Z; `v031_run.py cloud`)

All 12 segments are kept; none dropped.

- **PHerc1667, step 0: PLACEMENT_REFIT on 6/6 segments.** Refit B' NCC0 ≥ 0.4 on 6 of 6 crops everywhere (0.61–0.92);
  the published mesh reached 0.4 on 0–1 crop per segment (−0.25 to 0.79) and the refit beat it on every crop (36/36).
  This is #1843 again: published matrix landmark RMS 51.4 px at 2.399 µm, least-squares 0.93 px (leave-one-out max
  4.6 px). **The organisers' `mrg20736-1um` PHerc1667 maps were made along the published mesh.**
- **PHercParis4, step 0: PLACEMENT_OK on 6/6 segments.** Published NCC0 ≥ 0.4 on 6/6 crops for five segments and 5/6
  for `P4_20231012184424` (one crop 0.27). Landmark RMS 1.22 px at 2.4 µm for both matrices (16 landmarks, LOO max
  2.95 px), singular values 0.4709/0.4708/0.4707 (physical 0.4704). The transform's fixed volume is the labelled 78 keV
  scan, so the labels carry.
- **Step 1: CARRY_OK on 12/12.** Median normal residual 0.04–0.06 voxels; carried ink 99.5–100% of the in-scope ink
  (27–299 mm² per segment); local scale 0.4738/0.4675 (PHerc1667, target 0.4706) and 0.4712/0.4705 (PHercParis4,
  target 0.4704). ROIs 8–39 M B vertices.
- **Plan: 216 windows, 18 per segment (6 per z band everywhere), 108 per scroll**, 18.8–27.5 mm² of carried ink per
  segment in windows. `windows.json` committed with this text. ESTIMATE: PHerc1667 main ~2.2 h + extra ~0.8 h;
  PHercParis4 ~2.2 h; total ~5.2 h > 4 h, so two queue items (44: PHerc1667, 45: PHercParis4). The verdict (pooled) is
  computed by item 45's `rule`; item 44's `rule` reports PHerc1667 and a pooled UNKNOWN until PHercParis4 is scored.

Per segment: `results/v031-canonical-vs-mrg-1um/step0/<seg>.json`, `carry/<seg>.json`, `windows.json`.

**Smoke test of the render path (cloud, no score).** Window `1667_w028_r1715_c690` rendered along B' and along the
published B (34 s each); labels 81.8% ink, 100% carried and CT-valid, 4088 / 858 ds8 positives / negatives; mrg crop
400 × 800. The published-B render matches the organisers' own L1 surface volume of the segment at the same canvas
pixels (in-plane NCC 0.998 of 21-layer means), so the window grid is the `mrg20736-1um` canvas. No ink map was made and
no AUC computed; the files were deleted.

# V-031b: PHerc0139 extension (registered 2026-10-08T05:31Z, before any V-031b score)

Requested by the roadmap thread after V-031's registration: the 8 labelled PHerc0139 segments with a published
`mrg20736-1um` map on a 1.129 µm scan, under V-031's rule. Public-safe (PHerc0139 is outside the First Letters list).
GPU-QUEUE item 46, after 45.

**Data.** Labelled scan `PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr`; 1.129 µm scan
`PHerc0139/volumes/20260413113053-1.129um-0.2m-59keV-masked.zarr` with `transform.json` (12 landmarks, fixed volume
`SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_0139_TA3_0001_masked`); segments `20250108000005-w030_2025010818`,
`20260317000000-w035_2026031718`, `20260302000000-w039_2026030210`, `20250831000000-w040_2025083102`,
`20260108000000-w041_2026010816`, `20260112000000-w043_2026011217`, `20260115000000-w044_2026011522`,
`20260126000000-w045_2026012619` (short names `0139_w030` … `0139_w045`). Exact paths in `segments.json` (added by
`resolve PHerc0139`; the V-031 entries are unchanged). Labels `2.399um-volume-20260102150214/20260918`.

**Steps, windows, scoring: exactly V-031's** (step 0 with V-030's rule: published transform, refit only if it fails;
step 1; the V-030 windows; offset 0, identical pixels; no training). The windows are fixed in a separate committed file,
`results/v031-canonical-vs-mrg-1um/windows_v031b.json`, so V-031's `windows.json` is untouched.

**Rule and how it is reported.** The same rule (canonical ≥ mrg on ≥ ⌈0.7 n⌉ windows and mean difference ≥ 0;
UNKNOWN under 10 scored windows or with planned windows unscored), evaluated:
- per scroll, for PHerc0139 as for the other two;
- **V-031 registered verdict:** pooled over PHerc1667 + PHercParis4 only (`RULE pooled V-031`, `VERDICT` line),
  unchanged by V-031b;
- **V-031b extension:** pooled over all three scrolls (`RULE pooled V-031b extension`, `VERDICT_V031B_EXTENSION`
  line). It is not V-031's verdict.

**Descriptive depth column (villa #1912; V-013, V-021).** On w044 and w045 the canonical model read best 32 layers
below the PHerc0139 *March-scan* mesh (`20260319133554`, 2.403 µm): V-021 A1/A2 label best −32 (AUC 0.910 / 0.803 vs
0.661 / 0.635 at 0). Convention (cross-scan-ink-transfer `infer_crop.py --offsets`): the 62-layer window centred at
stack centre + offset; layers run along the render normal (cross(dv, du) of the mesh), so −32 is towards −normal.
Those renders were level 1, oversample 2 on the 2.403 µm scan, 2.403 µm per layer: −32 layers = −76.9 µm. Here a layer
is 2.258 µm, so −76.9 / 2.258 = −34.06 → **−34 layers**. The sign carries over: the March mesh maps onto the 2.399 µm
mesh by an affine fit with det +0.994 (same grid orientation, grid offset (2, 3) cells, mean residual 22 voxels ≈ 53 µm,
which is #1912's misplacement), and the 1.129 → 2.399 µm transform has det +0.104 (no mirror). To reach it, the w044
and w045 windows are rendered 63 + 2 × 34 = **131 layers**; their centre 63 layers sit at exactly the layer positions of
the 63-layer render, so the main scores are unchanged; the `canonical_depth-34` column reads the 62 layers shifted by
−34. Note: the 1.129 µm meshes of PHerc0139 are mesh A (2.399 µm) pushed through the published transform (carry
residual 0.04 voxels), not the March mesh, so the column asks whether the 2.399 µm mesh, and with it the organisers' 1 µm
map, also sits off the inked layer there.

**Budget.** 85 windows, ~1.9 h by the driver's estimate (7 deep renders included); one queue item under the 4 h cap.

## V-031b steps 0 and 1 and the plan: results (cloud CPU, 2026-10-08 04:48–05:30Z)

- **Transform:** landmark RMS 1.16 px at 2.399 µm for both matrices (LOO max 2.26 px), singular values
  0.4701/0.4701/0.4700 (physical 0.4706).
- **Step 0: PLACEMENT_OK on 7 segments** (published NCC0 ≥ 0.4 on 6 of 6 crops each, 0.72–0.93).
- **Dropped: `0139_w043`.** Its mesh has 3.3 M vertices where the 1.129 µm scan has data, but none of its 17.77 mm²
  of labelled ink lies there (in-scope ink 0.00 mm²), so there is no crop to test and no window
  (`step0/0139_w043.json`: `DROPPED`).
- **Step 1: CARRY_OK on 7/7** (median residual 0.03–0.04 voxels; carried ink 99–101% of in-scope ink; scale
  0.4703/0.4699).
- **Plan: 85 windows** — w030 18, w035 18, w040 18, w039 13, w041 11, w044 5, w045 2 (bands with too little ink yield
  fewer than 6). In-scope ink is small on w044 (4.7 mm²) and w045 (1.9 mm²), so the depth column has 7 windows.

Driver changes for V-031b (before any score; V-031's windows, step 0 and carry files unchanged): PHerc0139 entry in
`SCROLLS`, `resolve <scroll>` merges into `segments.json`, per-study windows files, deep renders and the depth column,
the two pooled lines in `rule`, and step 0 now records a segment with no in-scope labelled ink as `DROPPED` rather than
as a placement failure.

Desktop: as V-031's block with `S=PHerc0139` (no `--extra`).
