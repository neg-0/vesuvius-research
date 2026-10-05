# V-016 code

`v016_cross_scan_ink.py` runs inside the pinned
[cross-scan-ink-transfer](https://github.com/TAUIL-Abd-Elilah/cross-scan-ink-transfer) `src` tree, next to V-014's
`v014_transform_ink.py`. Set it up with [`../../v014-pherc1667-transform/code/setup_local.sh`](../../v014-pherc1667-transform/code/setup_local.sh).
It reuses V-014's refit meshes (`V014_DATA`).

```sh
export VILLA_INFERENCE_DIR=.../villa/ink-detection/optimized_inference CANON_CKPT=.../r152_3ddec_v2_l5_epoch13.ckpt \
  V014_DATA=.../v014/data V016_DATA=.../v016/data XSCAN_CACHE=.../v016/cache
python v016_cross_scan_ink.py prep C          # test windows and training patches for all 6 segments (~2 GB of renders)
python v016_cross_scan_ink.py baseline C 20251208130119-w028_20251208130119156_flatboi   # 0.6268, as V-014
for f in 0 1 2; do python v016_cross_scan_ink.py train C $f; python v016_cross_scan_ink.py eval C $f; done
python v016_cross_scan_ink.py verdict C
```

Each step resumes when run again. Training saves a checkpoint every 200 iterations. The `P` arm, along the
published-matrix meshes, is the same sequence with `P` in place of `C`.
