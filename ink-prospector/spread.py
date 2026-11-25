"""Spread out from a hit: grow a large patch from a seed and read one side at full depth.

usage: spread.py <ct zarr> <m7 zarr> <out dir> z y x <half_nodes> fwd|rev [ct_level]   (prospect.py spread calls this)
Writes grid.npz (P, ok, n0: export it with export_tifxyz.py), stack.npy, ink.npy, valid.npy, ink.png and ink_ds2.png."""
import sys, time, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lazyvol import LazyVol
import grow, reader
from PIL import Image
ctp, m7p, out = sys.argv[1], sys.argv[2], sys.argv[3]; seed = np.array([float(v) for v in sys.argv[4:7]]); hn = int(sys.argv[7]); rev = sys.argv[8] == 'rev'
os.makedirs(out, exist_ok=True); t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
ct = LazyVol(ctp, level=int(sys.argv[9]) if len(sys.argv) > 9 else 0); m7 = LazyVol(m7p)
sf = grow.seed_frame(m7, seed)
if sf is None: sys.exit(f"no sheet within 14 voxels of seed {seed.tolist()} in the surface prediction; pick a seed from a scan's scores.tsv")
p0, n, U, V, dist, plan = sf
P, ok, N = grow.grow(m7, p0, n, U, V, half_nodes=hn); log(f"tracked {ok.mean():.3f}")
np.savez_compressed(out + "/grid.npz", P=P.astype(np.float32), ok=ok, n0=n)
D, valid, Nd = grow.densify(P, ok, 4)
st = grow.render_fast(ct, D, Nd, 17, valid=valid); st[:, ~valid] = 0; np.save(out + "/stack.npy", st); log("rendered")
cm = reader.load_reader('data/weights/reader-v3.pth', 'ext/villa-mip')
o = reader.run_fused(cm, st, layer_start=0, reverse=rev, overlap=0.5, batch=4); log("read (4 fused views)")
np.save(out + "/ink.npy", o.astype(np.float16)); np.save(out + "/valid.npy", valid)
img = (np.clip((o - 0.25) / 0.5, 0, 1) * 255).astype(np.uint8); img[~valid] = 0
Image.fromarray(img).save(out + "/ink.png"); Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.LANCZOS).save(out + "/ink_ds2.png")
import textscore
L = textscore.lineness(np.asarray(Image.fromarray(np.where(valid, np.maximum(img, 1), 0).astype(np.uint8)).resize((img.shape[1] // 4, img.shape[0] // 4), Image.LANCZOS)), 36.0)   # nominal ds4 pixel, 8.6 to 9.6 um scans
log(f"line banding {L} (above about 2: bands at one angle, as lines of text give; experimental)")
