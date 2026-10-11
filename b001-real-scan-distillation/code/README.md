# B-001 code

Python 3.11+, numpy, scipy, opencv-python, tifffile, zarr, torch (CUDA for training and reads), and a
[ScrollPrize/villa](https://github.com/ScrollPrize/villa) checkout at `e0bbb8b` for the ink model's own loss,
normalisation and inference.

| folder | what |
| --- | --- |
| `b001/` | `b001_data.py` (resolve, CT-only `place`, `tiles`, `windows`, `fetch`, `status`, `--check`), `b001_train.py`, `b001_read.py` (read and rule), `place_probe.py` (the CT-only probe quoted in the protocol) |
| `b001r/` | seed replication and simulated-input ablation (`fetch_sim`, `rule`) |
| `b001s/` | six-scroll leave-one-scroll-out tiles and rule |
| `b001g/` | the PHerc0800 first look and its shuffled-layer `control` / `compare` |
| `v026/`, `v028/`, `v031/` | shared helpers the drivers import by these names (villa import shim and inference flags; fine-tuning loop; bucket listing and range reads), trimmed to what is used here |

The drivers are the files that ran, unchanged except one line in `b001/b001_read.py`: `committed()` now finds the git
root with `git rev-parse` instead of a fixed number of `..` (the folder sits at a different depth here).

## Run

```bash
export V026_VILLA_SRC=/path/to/villa/vesuvius/src     # villa at e0bbb8b
export V026_CKPT=/path/to/hybrid_3d2d-seed42/step-075000.pth
export B001_DATA=/scratch/b001                        # CT crops and maps (about 3 GB for B-001, 11 GB for B-001s)
cd code/b001
python b001_data.py --check && python b001_train.py --check && python b001_read.py --check
python b001_data.py fetch                             # the committed tiles/windows in results/b001
python b001_train.py train $B001_DATA/ckpt            # about 20 min on an RTX 4090
python b001_read.py read $B001_DATA/ckpt && python b001_read.py rule
```

`resolve`, `place`, `tiles` and `windows` regenerate `results/b001` from the bucket; the committed outputs are what
the scores used. B-001r and B-001s follow their folders' docstrings with the env vars `B001_SEED`, `B001_TILES`,
`B001_CKPT_NAME`, `B001_EXCLUDE_SCROLL` and `B001_STEPS` (see `PROTOCOL.md`). `read` and `rule` refuse to run unless
the window list is committed unchanged.
