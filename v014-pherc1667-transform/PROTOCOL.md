# V-014 — PHerc1667 1.129 µm transform: does the refit matrix read more ink?

## Registration

Registered 2026-10-04 (UTC) before any CT chunk of the 1.129 µm volume or any
ink inference for this experiment. Target: villa issue
[#1843](https://github.com/ScrollPrize/villa/issues/1843) (kadenpool, 2026-09-20).
The issue showed the published `1.129 µm → 2.399 µm` matrix misses its own
landmarks and that refit surfaces are brighter on papyrus. It did not measure
ink. A PR search for 1843 and its sibling 1835 returned nothing on 2026-10-04,
so the ink measurement is open.

## Pre-check already done (geometry only, no CT)

Done in the cloud container on 2026-10-04. It was not a registered run, and
its numbers are reported as a pre-check.

- `PHerc1667/volumes/20260323082859-1.129um-0.2m-59keV-masked.zarr/transform.json`
  maps the 1.129 µm volume (moving) to the 2.399 µm volume (fixed). Points are
  in xyz order and come with 6 landmark pairs.
- With the published matrix, landmark residuals are 25.1, 31.8, 88.6, 18.3,
  17.5 and 75.6 px at 2.399 µm. The RMS is 51.4 px, which is 123 µm and
  reproduces the issue's figure.
- The least-squares affine fit (`M_lsq`) has an RMS residual of 0.93 px
  (2.2 µm). Leave-one-out errors are 2.8, 1.0, 4.6, 2.1, 2.9 and 1.4 px. Each
  is lower than the published matrix's residual at the same point.
- The singular values of `M_lsq` are 0.4714, 0.4710 and 0.4708. The physical
  ratio is 1.129 / 2.399 = 0.4706. The published matrix's singular values are
  0.477, 0.471 and 0.465.
- Segment w029:
  - The published 1.129 µm mesh, mapped with the published matrix, lies a
    median 7.9 px (90th percentile 11.0 px) from the published 2.399 µm mesh.
    That distance is within the 20 px grid spacing.
  - The same mesh mapped with `M_lsq` lies a median 26.5 px (p90 85.5 px)
    away.
  - So the published 1.129 µm meshes are the 2.399 µm meshes pushed through
    the published matrix, and they inherit its error.

## Falsifiable hypothesis

Ink labels exist for 6 PHerc1667 segments, drawn on the 2.399 µm volume
`20251217075048`: w013, w018, w023, w028, w029 and w031. On each segment's
test window, the canonical model reads ink better from the 1.129 µm scan along
the corrected mesh (arm C) than along the published-equivalent mesh (arm P).

- **Pass:** mean AUC(C) − mean AUC(P) ≥ 0.02, and AUC(C) > AUC(P) on all but
  at most one computable segment (5 of 6 when all six are computable).
- **Fail:** mean gain < 0.02, or C < P on 2 or more segments.
- **UNKNOWN:** fewer than 4 segments are computable, for example because the
  window lies outside the 1.129 µm field of view or data is missing. UNKNOWN is
  reported as UNKNOWN and never as a fail.

## Arms (identical code, sampler and model; only the mesh and volume differ)

All arms use the segment's published 2.399 µm tifxyz mesh grid, so labels,
windows and pixel grids are identical. Let `p` be a 2.399 µm mesh vertex in
xyz.

- **R, the reference:** `p` rendered from the 2.399 µm volume at level 0. One
  output pixel and one layer step equal one 2.399 µm voxel.
- **P, published:** `M_pub⁻¹ · p` rendered from the 1.129 µm volume
  `20260323082859`. It reads level 1 (2.258 µm voxels) with oversample 2, so
  the output keeps 20 px per grid cell (2.399 µm on the surface) and each
  layer step is 2.258 µm.
- **C, corrected:** `M_lsq⁻¹ · p`, with sampling identical to P.

The renderer and model are `render_crop.py` and `infer_crop.py` from
cross-scan-ink-transfer `df9db2f`, the same adapted copy as V-013.
- Renders use 161 layers and smoothed normals with σ = 1 cell.
- The model is scrollprize/ink_canonical_2um at revision 075855bc, checkpoint
  `r152_3ddec_v2_l5_epoch13.ckpt`.
- Scoring is `eval_map` from `march_bench.py`. It searches label shifts of
  ±48 ds8 px and uses the same AUC definition for every arm.

## Window (fixed before any CT read)

The window is the densest labelled 60 × 170 grid-cell window (2.9 × 8.2 mm), as
in #1912. Candidate windows are restricted to cells where every valid vertex,
mapped by both `M_pub⁻¹` and `M_lsq⁻¹`, lies inside the level-0 bounds of the
1.129 µm volume, so each arm sees CT everywhere it is scored. If a segment has
no window with at least 20% of the densest unrestricted window's label
density, that segment is not computable.

## Primary and secondary measures

- **Primary:** AUC at the centre 62-layer window (offset 0) for P and C.
- **Secondary, descriptive with no pass rule:**
  - R at the centre.
  - Best of offsets −40, −20, 0, +20 and +40 layers for each arm, which shows
    whether a depth search alone rescues P.
  - The per-window displacement |`M_pub⁻¹ p` − `M_lsq⁻¹ p`| in µm (median, p90).

## Interpretation limits

- Six segments, one window each. This is a measured effect on the labelled
  areas, not on the whole scroll.
- `M_lsq` rests on 6 landmarks. The leave-one-out error and the issue's
  independent image registration (6.3 µm RMS) support it, but it is not a
  dense registration.
- A pass supports replacing the matrix and rebuilding the 19 derived meshes,
  as the issue proposes. It does not show that 1.129 µm reads better than
  2.399 µm; R versus C is reported only descriptively.

## Budget and environment

- GPU inference runs on one RTX 3080; the environment adds zarr, scipy, tifffile,
  imagecodecs and opencv to the V-013 GPU environment.
- CT comes from the public bucket only. Expected volume is about 1–2 GB per
  1.129 µm arm per segment; chunks are cached and the cache is deleted after
  the run.

## Addendum, 2026-10-04 22:40Z: model-free CT placement check (registered before computing segments 2–6)

**Why the addendum.** On the first segment the GPU run returned ink AUC R 0.9897, P 0.6078 and C 0.5143,
for run `20261004T202856-b3dce64538`, w013. A CPU diagnostic on a 4 × 8 cell crop inside the w013 window
(rows 1030–1034, cols 720–728, 61 layers) then compared the CT itself with the reference.

- The C render matched the 2.399 µm reference: 3D NCC 0.663 at zero shift, and 0.673 at the best
  shift of +2 layers, (−2, −2) px.
- The P render did not: NCC 0.042 at zero shift, and 0.168 at the best shift, which sits at the edge of
  the search box (+20 layers, (+10, +12) px).

So on w013 the corrected mesh lands on the same papyrus as the reference and the published mesh does
not. The canonical model, trained on 2.4 µm 78 keV data, does not read ink on the 59 keV 1.129 µm
scan along either mesh. The primary ink-AUC test therefore measures cross-scan generalisation as much
as mesh placement. It stays as registered and will be reported as such.

**Placement test, all 6 segments.**
- **Crop:** the central 8 × 16 grid cells of each segment's V-014 window, 61 layers. R, P and C are
  rendered exactly as in the arms above.
- **Statistic:** NCC between R's central 21 layers and the same layers of P or C at zero shift. The
  best NCC within ±20 layers and ±12 px, step 2, is reported alongside.
- **Pass:** NCC0(C) > NCC0(P) on all 6 segments, and NCC0(C) ≥ 0.4 on at least 5.
- **Fail:** any segment where NCC0(P) ≥ NCC0(C).
- **UNKNOWN:** missing data, reported per segment.
- w013's crop above is in this rule's family but is not the registered crop; it is recomputed with
  the registered crop.
