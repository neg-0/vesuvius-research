# Lasagna-format normals from the public m7 surface prediction

Six First Letters-eligible 8.64 µm scrolls (PHerc0175A/B, 0306B, 0483A, 0490A/B) have no published lasagna normals,
so villa's lasagna fit cannot run on them as published. `make_normals.py` builds drop-in `nx`, `ny` and `grad_mag`
stores from the public m7 surface prediction alone.

## Method

- **Normals:** structure tensor of the Gaussian-smoothed m7 probability at level 2 (σ 1, ρ 3, with ρ 10 as a fallback
  where the tensor is degenerate). The sheet normal is the largest eigenvector.
- **Encoding:** as villa `lasagna/normal_encoding.py`. Flip to the nz ≥ 0 hemisphere, then u8 = round(n·127 + 128).
- **grad_mag:** one value per scroll, 1000 × K / S. S is the median m7 crest spacing along the normals over up to 12
  blocks. K = 0.464 was calibrated on PHerc1218.
- **Output:** `<scroll>_{nx,ny,grad_mag}.ome.zarr`, dataset "2" (scale 4), 32³ blosc-lz4 chunks with '/' keys. This
  is the layout the fit reads.
- **No-signal voxels:** voxels with no m7 signal get 0, the same as a missing chunk in the organisers' stores.

## Check against the organisers' lasagna on PHerc1218

Two level-2 blocks, 192³ each:

| block | median angle | p75 | p90 | our coverage | theirs |
| --- | --- | --- | --- | --- | --- |
| z 2880, y/x 760 | 7.8° | 13.1° | 26.1° | 100% | 86% |
| z 4800, y 700, x 300 (outer, more crushed) | 12.5° | | | | |

Our median crest spacing is 4.50 level-2 voxels, against the organisers' median grad_mag of 103, which sets K.

## Use

```
python3 -I make_normals.py --m7 <m7 .zarr URL or path> --scroll <name> --z0 <L2 z> --z1 <L2 z> --out <dir> \
  [--yx y0 y1 x0 x1] [--workers N]
```

It needs numpy, scipy and numcodecs, and reads the bucket anonymously over HTTPS. A 576-slice level-2 band of a
9363² scroll takes about 1 h on 16 cores.

## Also tried: automatic umbilicus

The fit also needs a scroll axis (umbilicus). An automatic finder based on orientation-field singularities came within
1 mm of A. Drobkov's published axes on two PHerc1218 slices, but was 4-18 mm off on most slices of three scrolls, so we
do not recommend it. Hand-placing the centre of the innermost closed loop on m7 slices came within 1.3-2.8 mm.
