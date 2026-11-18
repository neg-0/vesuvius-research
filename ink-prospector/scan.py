"""Ink Prospector scan: seed a lattice inside a box, grow mesh-free patches on uncovered sheets, read both sides.

Outputs per patch: <out>/pNNNN.npz (coarse grid, validity), pNNNN_{a,b}.png (ds4 ink maps, a = layers along +n,
b = reversed) and a row in scores.tsv. Resume-safe; `--shard i/n` splits the seed list between machines.
"""
import argparse, os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lazyvol import LazyVol
import grow, reader
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ct", required=True); ap.add_argument("--m7", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--box", type=int, nargs=6, required=True, help="z0 z1 y0 y1 x0 x1")
ap.add_argument("--spacing", type=int, default=250); ap.add_argument("--half-nodes", type=int, default=128)
ap.add_argument("--shard", default="0/1"); ap.add_argument("--max-patches", type=int, default=10**6)
ap.add_argument("--min-tracked", type=float, default=0.15); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--weights", default="data/weights/reader-v3.pth"); ap.add_argument("--villa", default="ext/villa-mip")
ap.add_argument("--cache", default=None); ap.add_argument("--device", default="cpu"); ap.add_argument("--ct-level", type=int, default=0); ap.add_argument("--um", type=float, default=0.0); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--fuse", action="store_true", help="average 4 turned/mirrored reads per face (4x read time, cleaner maps)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); si, sn = [int(v) for v in a.shard.split("/")]
t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)
ct = LazyVol(a.ct, level=a.ct_level, cache_dir=a.cache and a.cache + "/ct"); m7 = LazyVol(a.m7, cache_dir=a.cache and a.cache + "/m7")
cm = reader.load_reader(a.weights, a.villa)
z0, z1, y0, y1, x0, x1 = a.box; S = a.spacing
lat = np.array([(z, y, x) for z in range(z0 + S // 2, z1, S) for y in range(y0 + S // 2, y1, S) for x in range(x0 + S // 2, x1, S)], float)
rng = np.random.default_rng(a.seed); lat = lat[rng.permutation(len(lat))]          # same order on every machine
mine = [i for i in range(len(lat)) if i % sn == si]
log(f"{len(lat)} lattice seeds, {len(mine)} in shard {a.shard}")
covered = []; tree = None
tsv = os.path.join(a.out, "scores.tsv")
if not os.path.exists(tsv):
    open(tsv, "w").write("id\tseed_z\tseed_y\tseed_x\tstatus\ttracked\tarea_cm2\tfrac_a\tfrac_b\tp995_a\tp995_b\tstroke_a\tstroke_b\tsecs\n")
done = {int(l.split("\t")[0]) for l in open(tsv).read().splitlines()[1:]}
for i in sorted(done):
    f = os.path.join(a.out, f"p{i:04d}.npz")
    if os.path.exists(f):
        g = np.load(f); covered.append(g["P"][g["ok"]][::4])
if covered: tree = cKDTree(np.concatenate(covered))

def stroke(o, valid):
    """Stroke-likeness: share of valid area that is bright, thin (lost under a 24 px opening) ink."""
    b = (o > 0.5) & valid
    if b.sum() == 0: return 0.0
    d = ndi.zoom(b.astype(np.uint8), 0.25, order=0).astype(bool)
    thin = d & ~ndi.binary_opening(d, structure=np.ones((13, 13), bool))   # 13 px at ds4 ~ 0.45 mm stroke width cap
    return float(thin.sum() / max(ndi.zoom(valid.astype(np.uint8), 0.25, order=0).sum(), 1))

n_made = 0
for i in mine:
    if i in done or n_made >= a.max_patches: continue
    t = time.time(); row = lambda status, *r: open(tsv, "a").write("\t".join([str(i)] + [f"{v:.0f}" for v in lat[i]] + [status] + [f"{v:.4f}" for v in r]) + "\n")
    sf = grow.seed_frame(m7, lat[i])
    if sf is None: row("no_sheet"); continue
    p0, n, U, V, dist, plan = sf
    if tree is not None and tree.query(p0)[0] < 6: row("covered"); continue
    P, ok, N = grow.grow(m7, p0, n, U, V, half_nodes=a.half_nodes)
    if ok.mean() < a.min_tracked: row("small", ok.mean()); m7.drop(600); continue
    D, valid, Nd = grow.densify(P, ok, 4)
    st = grow.render_fast(ct, D, Nd, 17, valid=valid); st[:, ~valid] = 0
    if (st[8][valid] > 0).mean() < 0.5: row("masked", ok.mean()); m7.drop(600); continue   # sheet lies in masked (zero) CT
    res = {}
    for tag, rev in (("a", False), ("b", True)):
        o = (reader.run_fused if a.fuse else reader.run_stack)(cm, st, layer_start=0, reverse=rev, overlap=0.5, batch=a.batch, device=a.device)
        v = o[valid]; res[tag] = ((v > 0.5).mean(), np.percentile(v, 99.5), stroke(o, valid))
        img = (np.clip((o - 0.25) / 0.5, 0, 1) * 255).astype(np.uint8); img[~valid] = 0
        Image.fromarray(img).resize((img.shape[1] // 4, img.shape[0] // 4), Image.LANCZOS).save(os.path.join(a.out, f"p{i:04d}_{tag}.png"))
    np.savez_compressed(os.path.join(a.out, f"p{i:04d}.npz"), P=P.astype(np.float32), ok=ok, n0=n)
    um = a.um or (8.64 if "8.640um" in a.ct else 9.362); area = valid.sum() * (um * 1e-4) ** 2
    row("ok", ok.mean(), area, res["a"][0], res["b"][0], res["a"][1], res["b"][1], res["a"][2], res["b"][2], time.time() - t)
    covered.append(P[ok][::4]); tree = cKDTree(np.concatenate(covered)); n_made += 1
    log(f"patch {i}: tracked {ok.mean():.2f} area {area:.2f} cm2 frac a/b {res['a'][0]:.3f}/{res['b'][0]:.3f} stroke a/b {res['a'][2]:.4f}/{res['b'][2]:.4f} ({time.time() - t:.0f}s)")
    ct.drop(1500); m7.drop(600)
log("done")
