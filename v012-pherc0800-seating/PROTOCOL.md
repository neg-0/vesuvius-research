# V-012 — PHerc.0800 seating check, cloud port, segment 225813

## Registration

Registered 2026-10-04 (UTC) before any request for the test segment. Selected segment
`20251028225813-auto_grown_20251028225813045`, the V-011 metadata-ranked next
candidate. Exact eligible volume `20250521135224`, level 2.

Runs in the cloud container (4 CPUs, 15 GiB RAM, no GPU). The local V-008/V-009
run directories are not available here, so `v012_p800_seating_cloud.py`
combines mesh read, sample/footprint, and scoring in one attempt. Sampling,
normals, gap probe, scorer and synthetic controls are copied unchanged from
`v010_p800_seating.py`; thresholds, level, seed, sample count and probe steps
are copied from `v010_p800_seating_manifest.json`.

## Falsifiable hypothesis

1. Reproduction control: on V-010's segment `20251028222030-auto_grown_20251028222030940`,
   the port reproduces the V-010 sample hash `6f56f0cc…` and score 6.6608332793
   within 1e-8. If it does not, the port is wrong and the test segment result is
   not interpreted.
2. Test: on segment 225813 the 600-point seed-0 sample scores ≥15 with coverage
   ≥0.25 (`seated_sample`); 3 < score < 15 is `partial_sample`; otherwise
   `not_seated_sample`. Missing/corrupt data or a bound breach is UNKNOWN.

## Changes from V-010

- The chunk footprint is computed inside the run (centre plus ±2…58 probe
  points, the full set the scorer can touch) and written to evidence before any
  CT request, instead of being frozen by a prior V-009 run. The scorer still
  refuses any unplanned chunk.
- The level-2 `.zarray` header must match the V-010 frozen header exactly.
- CT budget 200 MiB per run; aggregate response cap 250 MiB; per-request
  timeout 20 s; runner timeout 1,200 s; one serial worker; seed 0; no GPU.
- Isolated environment (Python 3.11.15,
  numpy 2.4.6, tifffile 2026.3.3), not the locked local environment.

## Interpretation

As V-010: a 600-point sampled seating check of one segment, not proof of
seating, legibility, 10 letters in 4 cm², or prize eligibility. A seated result
would justify registering a bounded 4 cm² ink-inference pilot with null
controls. The segment also publishes a pre-rendered surface volume
(`surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr`), noted for that
later step and not read here.
