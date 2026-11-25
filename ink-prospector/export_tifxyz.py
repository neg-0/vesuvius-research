"""Write a grown patch as a tifxyz surface that VC3D (and villa tools) can open.

usage: export_tifxyz.py <pNNNN.npz or grid.npz> <out dir> [--scroll NAME] [--scan ID] [--every N]
Nodes are 4 level-0 voxels apart, so the tifxyz scale is 0.25. P holds z y x; x.tif, y.tif and z.tif hold the
coordinate of each node and -1 where the patch was not tracked. --every N keeps every Nth node in each direction
(scale 1/(4N)); --every 5 gives VC's default tracer step of 20 voxels, the grid vc_grow_seg_from_seed --resume grows on.
"""
import sys, os, json, argparse, numpy as np, tifffile


def write_tifxyz(P, ok, out, step=4.0, every=1, scroll="", scan="", extra=None):
    """P (G, G, 3) z y x node grid, ok (G, G) tracked mask -> tifxyz folder `out`. Returns the number of nodes written."""
    P = np.asarray(P, np.float32); ok = np.asarray(ok, bool) & np.isfinite(P).all(-1)
    if every > 1:
        r0, c0 = (np.argwhere(ok).min(0) if ok.any() else (0, 0))           # keep the grid anchored on tracked nodes
        P = P[r0::every, c0::every]; ok = ok[r0::every, c0::every]
    if not ok.any(): raise ValueError("no tracked nodes")
    rows = np.nonzero(ok.any(1))[0]; cols = np.nonzero(ok.any(0))[0]
    P = P[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]; ok = ok[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    os.makedirs(out, exist_ok=True); s = step * every
    for i, ax in ((2, "x"), (1, "y"), (0, "z")):
        tifffile.imwrite(os.path.join(out, ax + ".tif"), np.where(ok, P[..., i], -1.0).astype(np.float32))
    name = os.path.basename(os.path.abspath(out)); v = P[ok]
    meta = {"uuid": name, "name": name, "type": "seg", "format": "tifxyz", "scale": [1 / s, 1 / s],
            "bbox": [[float(v[:, 2].min()), float(v[:, 1].min()), float(v[:, 0].min())], [float(v[:, 2].max()), float(v[:, 1].max()), float(v[:, 0].max())]],
            "area_vx2": float(ok.sum() * s * s), "invalid_value": -1.0, "scroll": scroll, "scan": scan,
            "source": "Ink Prospector patch grown on the m7 surface prediction; level-0 voxel coordinates"}
    meta.update(extra or {})
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    return int(ok.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--scroll", default=""); ap.add_argument("--scan", default=""); ap.add_argument("--step", type=float, default=4.0)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args(); g = np.load(a.npz)
    n = write_tifxyz(g["P"], g["ok"], a.out, a.step, a.every, a.scroll, a.scan)
    print("wrote", a.out, "nodes", n)
