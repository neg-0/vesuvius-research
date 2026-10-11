# B-001 protocols

The three studies' registered specs, as committed before any of their model outputs (registration dates in each
status line). Internal run logistics (machine queue, pod setup) are omitted; nothing in the design or rules is changed.
Paths such as `experiments/b001/` and `results/b001/` are `code/b001/` and `results/b001/` here.

## B-001: real-scan distillation for the eligible ~9 µm scan types

**Status: REGISTERED 2026-10-10**, before any B-001 model output exists: no ink map of the stock or the fine-tuned model
has been computed on any B-001 window or tile. Before this text was committed, a CT-only alignment probe ran on four
segments (`experiments/b001/place_probe.py`, results quoted under "Why alignment"); it reads CT only, no labels,
teacher maps or models. No First Letters or Grand Prize scroll is opened.

### Question

The organisers' 9 µm ink model (`scrollprize/ink_9um`) was trained on 24 label sets of 2.4 µm renders pooled to
~9.6 µm (simulated 9 µm) and 5 of real 9.362 µm renders, all PHerc0139. No real 8.64 µm / 116 keV data, which is the
scan type of 4 Grand Prize scrolls and 11 First Letters scrolls. Does fine-tuning it on **real** 9.36 / 8.64 µm surface
volumes of other scrolls, with the canonical 2.4 µm model's published prediction of the same segment as the teacher,
read held-out scrolls' human-labelled ink better than the stock model?

Hypothesis: yes, by at least +0.05 mean window AUC (rule below).

### Data (open-data bucket, all published by the organisers)

For every segment scanned both ways the bucket has the mesh on both scans, the rendered surface volume (sv) on both
scans, and the canonical 2.4 µm model's prediction on the high-resolution canvas
(`ink-detection/*-new_canon_autoresearch_recipe-tile256-stride128.tif`, uint8, A canvas). `b001_data.py resolve` lists
them into `results/b001/segments.json`.

| role | scroll | low-res (LR) scan | high-res (HR) scan | segments with both meshes + teacher | human labels |
| --- | --- | --- | --- | --- | --- |
| train | PHerc0139 | 20250728140407 9.362 µm 113 keV | 20260102150214 2.399 µm | 38 | (not used) |
| train | PHerc0814 | 20250804134230 9.362 µm 113 keV | 20260309142202 2.399 µm | 19 | (not used) |
| train | PHerc0343P | 20250521134555 **8.64 µm 116 keV** | 20260304131111 2.215 µm | 8 | none |
| test | PHerc0841 | 20250821151531 9.366 µm 113 keV | 20260319124803 2.403 µm | 3 | 3 segments |
| test | PHerc0009B | 20250521125136 **8.64 µm 116 keV** | 20250820154339 2.401 µm | 18 | 1 segment |
| test | PHerc0500P2 | 20250820143440 9.362 µm 113 keV | 20250526151718 2.215 µm | 46 | 1 segment |

- **Hold-out is by scroll.** Nothing from a test scroll is trained on, including its unlabelled segments. The test
  scrolls are also outside ink_9um's four training scrolls (0139, 1667, Paris4, 0814; villa
  `aligned21_hybrid_3d2d.json`). Two training scrolls (0139, 0814) were in ink_9um's training; that is allowed, the
  question is about the held-out scrolls.
- **LR input** is the organisers' published LR sv (28 layers at 9.362/9.366 µm, 31 at 8.64 µm; level 0, 20 px per mesh
  cell, the renders First Letters users read). 31-layer stacks are cropped to the centre 28 (`[1:29]`), so the model's
  17-layer window has the same centre.
- **Teacher pseudo-labels** on train tiles: teacher ≥ 160 → ink, ≤ 50 → no ink, between → ignored (unsupervised).
  Fixed from the teacher's value range only (floor ~30, ink up to ~240 on PHerc0139 w035; no label or model output
  consulted).
- **Test labels:** the human `inklabels` / `supervision` zarrs (latest label date) on the HR canvas, level 0.

### Why alignment (and the correction step)

Our earlier 9 µm experiments (unpublished) moved HR labels to LR renders by grid index, LR px (y, x) ← HR px (k·y, k·x) with
k = LR µm / HR µm. The CT-only probe compared each LR sv with its HR sv pooled to the LR grid (layers 109 → 28 by
group mean, in-plane by bilinear resampling of HR level 2) over 4 crops per segment and searched shifts of ±16 px and
±3 layers. Normalised cross-correlation (NCC) at zero shift versus at the best shift:

| segment | NCC at zero shift | NCC at best shift | best in-plane shift (LR px, dy / dx) |
| --- | --- | --- | --- |
| PHerc0841 w00 | 0.24–0.53 | 0.82–0.91 | −8…−10 / −7…−8 (≈ 110 µm) |
| PHerc0139 w035 | 0.45–0.71 | 0.92–0.93 | −3…+6 / −5…+2 |
| PHerc0500P2 `20250825181859--1` | 0.26–0.74 | 0.85–0.92 | +1…+11 / −7…+6 (grows with y) |
| PHerc0009B `20250919125754` | 0.20–0.29 | 0.75–0.92 | −6…−12 / −8…−10 |

So grid-index transfer misplaces HR labels on the LR canvas by up to ~110 µm, about a stroke width, and the error
varies across a segment. B-001 therefore maps every HR label and teacher value through a **per-segment affine
correction fitted from CT alone** (`b001_data.py place`):

- 12 crops per segment, 192 × 192 LR px, at positions drawn with `default_rng(<segment index>)` among those whose
  192 px box is ≥ 98% CT-covered (LR sv level 4 mask), at least 384 px apart where possible.
- Per crop, NCC of LR layers 8–20 against the pooled HR stack over dz ∈ [−3, 3] layers, coarse in-plane search at 2×
  downsampling over ±20 px, then full-resolution refinement ±2 px around the coarse best (best dz ± 1).
- A crop is **informative** if its best NCC ≥ 0.5. The correction maps LR px p to HR px k·(p + s(p)), with
  s(p) = a + B·p a least-squares affine fit to the informative crops' best shifts (one trimming pass: drop crops with
  residual > 3 px, refit). dz = median of the informative crops' best dz.
- **PLACE_OK** if ≥ 8 informative crops remain and the fit's RMS residual ≤ 2.0 px. Otherwise **PLACE_FAIL** and the
  segment is not used (train or test). Descriptive: NCC at zero shift, the correction's size over the segment.
- dz is applied to the LR stack (rolled by dz layers, zero-filled) for every arm alike, before the 17-layer window.

### Data stages as run (2026-10-10, CT, teacher and human labels only; no model output)

- `place`: 56 of 70 segments PLACE_OK. PHerc0139 38/38, PHerc0343P 8/8, PHerc0814 5/19 (14 PLACE_FAIL, too few
  informative crops; excluded as the rule says), PHerc0841 3/3, PHerc0009B 1/1, PHerc0500P2 1/1 (only labelled test
  segments are placed). Median NCC at zero shift ~0.1–0.7, after correction ~0.85–0.94; max correction over a segment
  ~100–300 µm on PLACE_OK segments. Per-segment logs in `results/b001/place/`.
- `tiles`: 1,530 tiles (PHerc0139 1,140, PHerc0343P 240, PHerc0814 150). Scroll weighting keeps the three equal.
- `windows`: 49 windows (PHerc0841 25, PHerc0009B 12, PHerc0500P2 12), so n ≥ 20 is reachable.

### Train tiles (`b001_data.py tiles`, teacher and CT coverage only)

- Tile format (from our earlier 9 µm fine-tunes): 13 × 13-cell cores (260 × 260 px) on a 13-cell grid of the LR mesh, core ≥ 90% CT-valid (LR sv
  level 0 non-zero in any layer, read at level 2 for selection).
- Pseudo-labels through the PLACE_OK correction; a tile is a candidate if ≥ 25% of its px are supervised.
- Per segment, up to 30 tiles: first those with teacher-ink ≥ 2% of px, densest first; the rest of the 30 drawn at
  random (`default_rng(4501 + segment index)`) from the other candidates. So the model sees no-ink papyrus too.
- Output: `results/b001/tiles.json` (positions, segment, dz, label stats) and `results/b001/tile_labels/<tile>.npz`
  (packed ink and supervision, 260 × 260). CT is fetched on the GPU machine (`b001_data.py fetch`, ~2 MB per tile).

### Test windows (`b001_data.py windows`, human labels and CT coverage only)

- Window format (from our earlier 9 µm reads): 26 × 26-cell cores (520 × 520 px, 4.9 mm) on a 13-cell grid of each labelled test segment's LR
  mesh, ≥ 90% CT-valid, labels through the PLACE_OK correction.
- Score per candidate = labelled ink px inside supervision; greedy non-overlapping, densest first, up to 12 per
  segment, keeping only windows with ≥ 200 ink and ≥ 200 no-ink supervised px.
- Output: `results/b001/windows.json` and `results/b001/window_labels/<window>.npz`. Fixed and committed before any
  read; `b001_read.py` refuses to score if `windows.json` is not committed unchanged.

### Training (`b001_train.py`, one fixed recipe, no search, no checkpoint selection)

- Start: ink_9um `hybrid_3d2d-seed42/step-075000.pth` (sha256 `e635558a…62a9cab`), villa `e0bbb8b`, through
  `code/v028/v028_armf_train.py` by import (loss, crop sampler, normalisation, optimiser, resume, export unchanged).
- **All parameters trained** (an earlier small-data fine-tune froze the stem and encoder; B-001
  has 1,530 tiles of three scrolls, and the question is whether the low-level features adapt to real scans).
- Random 128 × 128 crops, the model's 17-layer window with ±2 layer jitter, dihedral augmentation, per-patch robust
  normalisation, BCE (label smoothing 0.5) + Dice from the checkpoint config, ignore outside supervision or CT.
- SGD Nesterov momentum 0.99, weight decay 3e-5, peak LR 1e-3, 200 warm-up steps then cosine to 0, **6,000 steps of
  batch 16** (96,000 crops), grad-norm clip 1.0, fp16 autocast. Tiles sampled in proportion to supervised px, each
  scroll's tiles weighted so the three scrolls contribute equally. Seed 4500. Resumable; stops cleanly after
  `B001_BUDGET` s (default 6,000) and the same command resumes.
- Output `b001_real.pth` in the checkpoint's own format plus `train_log.json`.

### Read and score (`b001_read.py`)

- Per window: the LR sv crop (28 layers, dz applied), villa `infer_single_zarr` with `--overlap 0.5 --blend-mode hann
  --batch-size 4`, forward and reverse, for **S0** (stock ink_9um step-075000) and **REAL** (`b001_real.pth`); plus
  REAL on the layer-shuffled stack (`default_rng(0).permutation(17)` of the model's 17 layers), forward only.
- **Forward** is the published sv layer order (the organisers' render convention that ink_9um was trained on). It is
  the scored direction; reverse is reported.
- Pixels: CT-valid (centre layer > 0) and supervised; positives = labelled ink. AUC per window (rank-based); a window
  is **scored** if it has ≥ 200 positive and ≥ 200 negative px.

### Pre-registered rule

On the scored windows (n):

- **PASS** if all of
  1. mean paired gain AUC(REAL) − AUC(S0) ≥ **+0.05**;
  2. REAL > S0 on ≥ **70%** of scored windows;
  3. for every test scroll with ≥ 3 scored windows, its mean paired gain ≥ 0;
  4. n ≥ 20.
- **INVALID** if REAL's shuffled-layer AUC, pooled over all scored windows' pixels, lies outside 0.5 ± 0.10 (the model
  reads something other than depth-ordered ink), or if any window lacks a map.
- **UNKNOWN** if n < 20.
- Otherwise **FAIL**.
- Descriptive only: per-scroll means, reverse and max(fwd, rev) AUCs, S0 with grid-index (uncorrected) labels on the
  same maps (how much the alignment alone changes a measured AUC), the 8.64 µm scroll (0009B) separately.

## B-001r: is B-001's gain real, and does it come from real-scan input?

**Status: REGISTERED 2026-10-11**, after B-001 PASSED (`results/b001/RESULT.md`: mean paired gain +0.0914, REAL > S0
on 44 of 49 held-out windows, all three held-out scrolls positive). No B-001r model output exists yet. It uses the same public scrolls, tiles, windows,
labels and base model as B-001, and no First Letters or Grand Prize scroll.

### Questions

1. **R1 (seed replication).** Does B-001's recipe pass B-001's rule again with a different training seed? Our earlier
   fine-tunes showed seed-to-seed swings of ~0.015-0.035 in mean gain, so one seed is not enough to build on.
2. **SIM (input ablation).** B-001's README said a pass should next be checked against "the same tiles pooled from the
   2.4 µm sv instead of the real LR sv". This study runs that check: does the gain come from training on **real**
   9.36 / 8.64 µm scans, or would the same teacher labels on **simulated** 9 µm input (the organisers' approach for
   24 of ink_9um's 29 label sets) do as well? This decides what to scale next. If real input matters, the free paired
   scans are the asset. If it doesn't, cheap simulated data from any 2.4 µm segment is.

### Design (everything not listed is B-001's, unchanged)

- **Tiles, teacher labels, windows, human labels, scoring pixels, base checkpoint, recipe:** B-001's committed
  `results/b001/` (`tiles.json`, `tile_labels/`, `windows.json`, `window_labels/`, `place/`) and `b001_train.py`. That
  means all parameters trained, 6,000 steps of batch 16, peak LR 1e-3, 200 warm-up steps, scroll-balanced sampling.
- **R1:** real LR tiles (`b001_data.py fetch`), training seed **4501** (`B001_SEED=4501`), output `b001_r1.pth`.
- **SIM:** the same 1,530 tiles, but the input is the HR surface volume pooled onto each tile's LR grid
  (`b001r_data.py fetch_sim`):
  - in-plane: HR level 2 (4× averaged);
  - in depth: averaged into the 28 LR layer bins by physical depth (B-001's placement pooling);
  - resampled at the CT-corrected HR position of every LR px, i.e. exactly where the teacher label for that px was
    read.
  So SIM tiles carry identical labels. Seed 4500 (B-001's), output `b001_sim.pth`.
  - Check (CT only, run during registration on 12 tiles): median NCC(SIM, real LR) **0.892**, best within ±3 layers of
    the segment dz.
  - Seen in that check: on PHerc0139 the real LR tile's best depth is up to **2 layers off** its segment's median dz.
    SIM is centred on the HR surface by construction, so SIM also has slightly better depth alignment. This is part of
    "simulated vs real" and is reported, not corrected.
- **Test:** B-001's 49 real-LR windows, unchanged. Maps via `b001_read.read_arms` (B-001's read path and flags):
  - S0 forward, re-read for the REPRO check;
  - R1 and SIM forward and reverse;
  - R1 and SIM on the layer-shuffled stack (forward).

### Pre-registered rules (`b001r_read.py rule`)

- **REPRO:** S0 forward AUC re-read must match B-001's committed S0 AUC within **0.005** on every scored
  window. Otherwise everything is **INVALID** (report and stop).
- **R1:** B-001's rule with R1 in place of REAL, against B-001's committed S0 AUCs:
  - mean paired gain ≥ +0.05;
  - R1 > S0 on ≥ 70% of windows;
  - per-scroll mean gain ≥ 0 for scrolls with ≥ 3 windows;
  - n ≥ 20;
  - R1's shuffled-layer pooled AUC within 0.5 ± 0.10 (else INVALID).

  PASS means **B-001 REPLICATED**. FAIL means **NOT_REPLICATED**: B-001's pass is then treated as a seed effect,
  and nothing scales from it until a third seed decides.
- **SIM:** paired d = AUC(B-001 REAL, committed) − AUC(SIM), forward, on the 49 windows.
  - **REAL_INPUT_MATTERS** if mean d ≥ +0.02 and d > 0 on ≥ 60% of windows.
  - **SIM_BETTER** if mean d ≤ −0.02 and d < 0 on ≥ 60%.
  - Otherwise **NO_CLEAR_DIFFERENCE**.
  - INVALID if SIM's shuffled control falls outside 0.5 ± 0.10.
  - Caveat: REAL is one seed, so a margin near 0.02 is within seed noise. R1 − SIM is reported alongside as a second
    REAL seed.
- **Descriptive:** mean AUCs, the R1/REAL per-window gain correlation, SIM against S0 under B-001's rule, and
  reverse AUCs.

## B-001s: B-001 scaled to all six paired scrolls, leave-one-scroll-out

**Status: REGISTERED 2026-10-11**, after B-001 PASSED and B-001r REPLICATED it and found REAL_INPUT_MATTERS
(`results/b001r/RESULT.md`):
- seed 4501 scored +0.1035 over stock, beating it on 46 of 49 windows; its per-window gain correlation with B-001 is
  0.94;
- B-001 beat the same recipe trained on simulated input by +0.0348, on 39 of 49 windows.

No B-001s model output exists yet.
The six public paired scrolls and the public ink_9um model; no First Letters or Grand Prize scroll.

### Question

Since real paired scans are the asset, does using all of them help? B-001 trained on 3 scrolls (PHerc0139, 0814, 0343P),
30 tiles per segment, 6,000 steps. B-001s trains one model per held-out test scroll, each on **every other paired scroll**,
including the unlabelled segments of the other two test scrolls. It uses up to 60 tiles per segment and 12,000 steps.

- It roughly doubles the 8.64 µm training data for two folds: PHerc0009B's 17 unlabelled segments join PHerc0343P.
- It adds a fourth and fifth scroll to every fold.

Each test scroll is still scored only by the model that never saw any of its segments.

### Design (everything not listed is B-001's, unchanged)

- **Segments:** B-001's `segments.json`, with the test scrolls' unlabelled segments marked usable
  (`results/b001s/segments.json`). Placement uses B-001's committed placements, plus the same CT-only rule
  (`b001_data.place_one`) on the 62 unlabelled test-scroll segments. Only PLACE_OK segments are used.
- **Tiles** (`b001s_data.py tiles`): B-001's tile rule and teacher thresholds, **up to 60 tiles per segment** (B-001:
  30), over every PLACE_OK segment except the five labelled test segments. Those segments are never trained on in
  any fold, so no tile can overlap a test window.
- **Folds:** for test scroll X ∈ {PHerc0841, PHerc0009B, PHerc0500P2}, model F_X trains on every tile whose scroll is
  not X (`B001_EXCLUDE_SCROLL=X`). Scrolls are weighted equally, as in B-001.
- **Recipe:** B-001's (`b001_train.py`, all parameters, batch 16, peak LR 1e-3, 200 warm-up steps, cosine), seed
  4500, **12,000 steps** (`B001_STEPS=12000`), output `<ckpt root>/<X>/b001s.pth`.
- **Test:** B-001's 49 windows and labels, copied unchanged (`b001s_data.py windows`). Each window is read by F_X for
  its own scroll X, forward and reverse plus the layer-shuffled control (`b001_read.read_arms`, B-001's flags). S0
  forward is re-read on every window for REPRO.

### Pre-registered rule (`b001s_read.py rule`)

- **REPRO:** S0 forward AUC within 0.005 of B-001's committed S0 AUC on every window. Otherwise INVALID.
- **Controls:** each fold model's shuffled-layer pooled AUC within 0.5 ± 0.10. Otherwise INVALID.
- **Reference:** per window, the mean forward AUC of B-001 REAL and B-001r R1 (two seeds of the 3-scroll recipe).
- **SCALE_HELPS** if mean(B-001s − reference) ≥ **+0.02** and B-001s is higher on ≥ **60%** of the 49 windows.
- **SCALE_HURTS** if mean ≤ −0.02 and lower on ≥ 60%.
- Otherwise **NO_CLEAR_DIFFERENCE**.
- Also reported, as a sanity line: B-001's rule against stock S0 (VS_S0).
- Descriptive: per-scroll means, reverse AUC, and the 2.4 µm teacher on the same windows (0.9088,
  `results/b001/POSTHOC-teacher.md`).

### Data as registered (2026-10-11; CT, teacher and copied labels only, no model output)

- **Placement:** 59 of the 62 unlabelled test-scroll segments are PLACE_OK: PHerc0009B 16 of 17, PHerc0500P2 43 of 45.
- **Tiles:** 5,717 in total: PHerc0139 2,280, PHerc0500P2 1,725, PHerc0009B 960, PHerc0343P 463, PHerc0814 289.
  That is about 10.9 GB of CT crops. The trainer holds a fold's tiles in RAM, so training needs **at least 32 GB of
  system RAM**.
- **`b001s_data.py status` must print:**
  - `FOLD PHerc0841: 5717 training tiles from 5 scrolls`
  - `FOLD PHerc0009B: 4757 training tiles from 4 scrolls`
  - `FOLD PHerc0500P2: 3992 training tiles from 4 scrolls`
- **Windows:** B-001's 49, copied unchanged.
