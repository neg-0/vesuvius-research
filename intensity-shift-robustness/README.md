# Ink readers are sensitive to scan-to-scan intensity shift

Two public ink readers lose a lot of accuracy on a labelled 9.362 µm segment when that segment is given the grey-level
distribution of other scans of the same type (9.362 µm, 113 keV). A per-layer histogram match back to a reference
render is a cheap, monotone correction. `match_intensity.py` does both the matching and the per-layer statistics.

The test was registered (method, baselines and pass rule written down) on 2026-10-24, before any shifted render was scored.

## Setup

- **Labelled segment:** PHerc0500P2 `20250825181859` on volume `20250820143440-9.362um-1.2m-113keV`. It is a
  28-layer `vc_render_tifxyz` render (`--num-slices 28 --slice-step 1`), scored against the public ink labels.
- **Readers:**
  - Hecate at depth 0;
  - Reader V3 (`domenicor046/reader-v3`, sha256 `ea405417…984d8a`, `--direction both --layer-start 6 --layer-end 23`).
- **Shift:** each of the 28 layers is CDF-matched, on papyrus pixels only (pixels > 0 on every layer), to the same
  layer of a render from another scroll's 9.362 µm / 113 keV scan. Six such scans were used.
  - The papyrus mask is unchanged, so the labels and the scoring canvas are identical.
  - Geometry is untouched.
- **Score:** pixel AUC on the reading (reverse) side, in 4 × 4 blocks inside the supervised region.

## Result

| grey levels taken from | Hecate d0 | Reader V3 |
| --- | --- | --- |
| (unshifted) | 0.9307 | 0.8955 |
| itself (null control) | 0.9305 | 0.8959 |
| scan 1 | 0.6937 | 0.8538 |
| scan 2 | 0.7370 | 0.8618 |
| scan 3 | 0.8412 | 0.8649 |
| scan 4 | 0.8879 | 0.9002 |
| scan 5 | 0.7756 | 0.8863 |
| scan 6 | 0.6485 | 0.7932 |

- Hecate loses 0.04-0.28 AUC on all six scans; Reader V3 loses 0.03-0.10 on four.
- Reader V3's training set includes other PHerc0500P2 segments from the same volume, so this segment is close to its
  best case.
- An empty ink map on a new scan can come from intensity shift alone. Before trusting a negative, match the scan's
  render to a render where the reader is known to work, and re-run.

## Use

```
python3 match_intensity.py SRC.zarr REF.zarr OUT.zarr   # SRC remapped layer by layer onto REF's distribution
python3 match_intensity.py stats A.zarr [B.zarr ...]    # per-layer p5/p50/p95 over papyrus pixels
python3 match_intensity.py selftest
```

The script needs numpy and zarr. It reads either a plain array or a group with array "0" (as `vc_render_tifxyz`
writes it). Both renders must have the same number of layers.
