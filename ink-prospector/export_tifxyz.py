"""Write a grown patch as a tifxyz surface that VC3D (and villa tools) can open.

usage: export_tifxyz.py <pNNNN.npz> <out dir> [--scroll NAME] [--scan ID]
Nodes are 4 level-0 voxels apart, so the tifxyz scale is 0.25. P holds z y x; x.tif, y.tif and z.tif hold the
coordinate of each node and -1 where the patch was not tracked."""
import sys, os, json, argparse, numpy as np, tifffile
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("out")
ap.add_argument("--scroll", default=""); ap.add_argument("--scan", default=""); ap.add_argument("--step", type=float, default=4.0)
a = ap.parse_args()
g = np.load(a.npz); P = g["P"].astype(np.float32); ok = g["ok"].astype(bool) & np.isfinite(P).all(-1)
assert ok.any(), "no tracked nodes"
rows = np.nonzero(ok.any(1))[0]; cols = np.nonzero(ok.any(0))[0]
P = P[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]; ok = ok[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
os.makedirs(a.out, exist_ok=True)
for i, ax in ((2, "x"), (1, "y"), (0, "z")):
    A = np.where(ok, P[..., i], -1.0).astype(np.float32); tifffile.imwrite(os.path.join(a.out, ax + ".tif"), A)
name = os.path.basename(os.path.abspath(a.out)); v = P[ok]
meta = {"uuid": name, "name": name, "type": "seg", "format": "tifxyz", "scale": [1 / a.step, 1 / a.step],
        "bbox": [[float(v[:, 2].min()), float(v[:, 1].min()), float(v[:, 0].min())], [float(v[:, 2].max()), float(v[:, 1].max()), float(v[:, 0].max())]],
        "area_vx2": float(ok.sum() * a.step * a.step), "invalid_value": -1.0, "scroll": a.scroll, "scan": a.scan,
        "source": "Ink Prospector patch grown on the m7 surface prediction; level-0 voxel coordinates"}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("wrote", a.out, "grid", ok.shape, "nodes", int(ok.sum()))
