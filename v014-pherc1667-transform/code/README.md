# V-014 / V-015 code

`setup_local.sh` fetches the pinned cross-scan-ink-transfer tree (df9db2f, MIT), villa and the
canonical checkpoint into `./work` and copies these drivers into `work/xscan/src`. Then, from
`work/xscan/src`:

```sh
SEGS="20240304141531-w013_20240304141531_flatboi 20240304144031-w018_20240304144031_flatboi \
 20240304161941-w023_20240304161941_flatboi 20251208130119-w028_20251208130119156_flatboi \
 20251212185248-w029_20251212185248662_flatboi 20251223230000-w031_2025122323_flatboi"
# CT placement check (CPU, about 30 min, a few hundred MB of CT)
V014_DATA=../../data python v014_ct_placement.py $SEGS
# ink AUC for R/P/C (GPU recommended; set INK_DEVICE=cpu to run on CPU)
VILLA_INFERENCE_DIR=../../villa/ink-detection/optimized_inference CANON_CKPT=../../models/r152_3ddec_v2_l5_epoch13.ckpt \
  V014_DATA=../../data python v014_transform_ink.py $SEGS
# transform audit (metadata only, any directory)
python v015_transform_audit.py
```

Needs numpy, scipy, tifffile, imagecodecs, zarr, opencv-python-headless, and torch for the ink step.
