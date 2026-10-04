# V-013 — Label-free depth field for off-sheet PHerc0139 March-scan meshes (villa #1912)

Registered 2026-10-04 before any CT read or AUC computation. Status: complete; see [RESULT.md](RESULT.md).
Reproduction control passed 2026-10-04: run `20261004T005644-9df898799c`,
w044 mesh-as-is AUC 0.69595 vs reporter 0.69584 (CPU fp32 vs their GPU
autocast), same label offset [46, 48]; 40 min wall, 30 min of it inference.
First attempt `20261004T005607-90b4b26016` failed before rendering (OpenCV
missing); preserved. Adapted source: a local working copy
(copy of the pinned tree; `infer_crop.py` device/autocast change only, plus new
`v013_*.py` drivers).

## Prior coverage

- Issue: <https://github.com/ScrollPrize/villa/issues/1912> (TAUIL-Abd-Elilah,
  2026-09-27; open, no assignee or linked PR seen 2026-10-04).
- Reporter's code, MIT, pinned `df9db2f4ea7e63fb6584f1aa3548f2773b927617`, cloned
  into a local working copy. It already
  renders ±190 µm along smoothed normals, flattens with a structure-tensor
  horizon fit (`refine_surface.py`), runs `ink_canonical_2um` at 5 windows
  (−96…+96 µm, 48 µm = 20 layers apart) and picks one window per segment by
  "most pixels > 0.5". Benchmark (8 labelled segments, densest 2.9 × 8.2 mm
  window): 2.399 µm reference mean AUC 0.872; mesh as published 0.756;
  label-free pick 0.803; label-chosen window within 0.007 of that. Local
  1–4 mm picks among the same 5 windows did not help (`C2_march_scan_local_depth.json`).
- Their own control says signal drops beyond ~10 layers of misplacement, so the
  20-layer window spacing is coarser than the model's tolerance. That gap, and
  tilt within a window (w045 prefers −96 to +48 µm), are not addressed.

## First check (cheap, CT only)

Before any new ink pass: compare the CT-only predicted depth per ~2 mm
sub-window with the reporter's label-best depth per sub-window
(`C2_march_scan_local_depth.json`, 8 segments x 4 sub-windows). If the predictor
does not beat the constant 0 prediction on that set, stop or redesign before
spending inference time.

## Falsifiable hypothesis

A label-free, smooth, continuous depth field d(y, x) estimated from CT alone
(papyrus-layer tracking at ≤4.8 µm depth resolution, regularised in-plane) and
applied before one ink-model pass raises mean AUC over the same 8 segments and
windows above 0.803, and above 0.82 on w044/w039/w045 combined, while no
segment whose mesh is already aligned (w030, w035, w040, w041, w043) loses more
than 0.02 against its mesh-as-is AUC. Failure on either clause rejects it.

## Controls

- Reproduction: mesh-as-is AUC on w044 must match the reporter's 0.696 within
  0.01 before any new result is interpreted.
- Null: a random smooth depth field with the same amplitude and smoothness
  (seed 0) must not beat mesh-as-is on average.
- Upper bound: label-oracle per-subwindow depth (reported, not a claim).
- Labels are used only for scoring, never for estimating d.

## Compute and bounds

CPU benchmark 2026-10-04 in this container (4 threads, torch CPU, checkpoint
SHA-256 `36dd0de8…c3e0`, HF revision `075855bc…`): about 8 s per 256² tile,
roughly 28 min per depth pass per 2.9 × 8.2 mm window at stride 128. One pass on
all 8 segments is about 4 h on CPU; a fine depth sweep (≥10 depths) is
≈40 h CPU and was run on an RTX 3080 instead. CT budget ≈1.2 GB per segment (≈10 GB total), within 30 GB disk.

## Predictor P1 (registered 2026-10-04 after inspecting w044 only)

w044's flattened CT profile has its strongest sheet peak at about −40 to −36
layers in all four ~2 mm sub-windows, where the reporter's label-best depth is
−40 layers (−96 µm) in all four; the mesh (0) sits on a weaker neighbouring
peak. P1, fixed now before any other segment's CT is inspected:

- per sub-window (the reporter's 4 equal splits along the window length),
  mean intensity over the flattened stack, smoothed along depth (Gaussian σ = 3
  layers);
- predicted offset = argmax of that profile within [−48, +48] layers;
- scored by snapping to the nearest of the reporter's windows {−40, −20, 0,
  +20, +40} and comparing with `subwindow_best_depth_um` (None skipped).

Baseline: constant 0 matches 13 of 30 scorable sub-windows. P1 passes the cheap
check if it matches more than 13 of 30 overall and more than the constant-0
count on the 7 segments other than w044 (w044 is the design segment and is
reported separately). Failing that, P1 is rejected and no ink pass is run for it.

## Label-free selection on the GPU fine sweep (registered before computing)

Sweep: 25 offsets (−48…+48 layers, step 4), 8 segments, committed under
`results/v013-sweep/`. Label-oracle per-segment mean AUC 0.8408 (reporter's
5-window oracle 0.8100; reporter's label-free pick 0.803). Rules fixed now,
scored with the reporter's `eval_map` on the stored ds8 maps:

- R1 (primary): reporter's rule on the fine grid — per segment, the offset
  whose map has the largest fraction of pixels > 0.5.
- R2: largest mean probability per segment.
- R3: per-pixel maximum over all 25 offsets (one composite map).
- R4: R1 applied per ~2 mm sub-window (4 splits along the window length).

V-013 passes if R1 (or, reported with a multiple-comparison caveat, the best of
R1–R4) beats 0.803 mean AUC and costs no already-aligned segment (w030, w035,
w040, w041, w043) more than 0.02 against the reporter's flattened label-free
result for that segment.
