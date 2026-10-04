# V-013 code

Our drivers for the [V-013 protocol](../PROTOCOL.md). They run inside a copy of
the pinned reporter tree, which `setup_local.sh` fetches into `./work`.

| File | Purpose |
|---|---|
| `v013_repro_asis.py` | Reproduce the reporter's mesh-as-published AUC for one segment (CPU or GPU) |
| `v013_render_flat.py` | CT-only stage: window render (161 layers) + `refine_surface.py` flattening |
| `v013_score_p1.py` | CT-only depth predictor P1 vs the reporter's label-best sub-window depths (rejected) |
| `v013_sweep.py` | GPU sweep: 25 flattened depths (−48…+48 layers), AUC + ds8 map per depth |
| `v013_select.py` | Label-free selection rules R1–R4 scored from the committed ds8 maps |

Requirements: Python 3.11+, torch (CUDA for the sweep; `INK_DEVICE=cpu` works for
`infer_crop.py` but is ~8 s per 256² tile), numpy, scipy, zarr, tifffile,
opencv-python-headless.

```sh
bash setup_local.sh
export VILLA_INFERENCE_DIR=$PWD/work/villa/ink-detection/optimized_inference \
  CANON_CKPT=$PWD/work/models/r152_3ddec_v2_l5_epoch13.ckpt \
  MARCH_BENCH_DATA=$PWD/work/data PUZZLE_RUN_DIR=$PWD/work/run
cd work/xscan/src
python v013_sweep.py 20260115000000-w044_2026011522 20260126000000-w045_2026012619 \
  20260302000000-w039_2026030210 20260112000000-w043_2026011217 20250108000005-w030_2025010818 \
  20250831000000-w040_2025083102 20260108000000-w041_2026010816 20260317000000-w035_2026031718
cd - && python v013_select.py work/xscan/results/C2_march_scan_8_segments.json
```

About 1.2 GB of CT is read per segment from the public Vesuvius Challenge bucket;
on one RTX 3080 the full sweep took about 3.5 h, mostly CT download and render.
