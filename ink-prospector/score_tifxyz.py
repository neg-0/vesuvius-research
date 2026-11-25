"""Read and score existing surfaces: any tifxyz segment (VC3D, HerculeX, community or grown with VC's tracer).

usage: score_tifxyz.py --out DIR (--scroll S | --ct ZARR [--um U]) [--crop r0 r1 c0 c1] [--device cuda] [--fuse] <tifxyz dir>...
Each surface is densified to one point per voxel, 17 CT layers are sampled along its normals (as in scan), Reader V3
reads both faces, and the text score is computed. The output is a scan folder (scores.tsv, pNNNN.npz, pNNNN_a/b.png,
run.json), so `prospect.py rank`, `viewer`, `stitch` and `vc_export.py` work on it unchanged. surfaces.tsv maps each
pNNNN to its tifxyz path. --crop takes a window of the tifxyz grid (rows r0:r1, cols c0:c1) for large sheets.
Run from the work folder (PROSPECTOR_HOME), which holds data/weights/reader-v3.pth and ext/villa-mip.
"""
import argparse, json, os, sys, time
import numpy as np, tifffile
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lazyvol import LazyVol
import grow
from textscore import stroke


def load_tifxyz(path, crop=None):
    """tifxyz folder -> P (rows, cols, 3) z y x, ok mask, step in voxels between nodes."""
    x, y, z = (tifffile.imread(os.path.join(path, a + ".tif")).astype(np.float32) for a in "xyz")
    meta = json.load(open(os.path.join(path, "meta.json")))
    step = 1.0 / float(meta.get("scale", [1.0])[0])
    if crop: r0, r1, c0, c1 = crop; x, y, z = x[r0:r1, c0:c1], y[r0:r1, c0:c1], z[r0:r1, c0:c1]
    ok = (z > 0) & (x != -1) & (y != -1) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)   # VC treats z <= 0 as invalid
    return np.stack([z, y, x], -1), ok, step


def to_dense(P, ok, step):
    """Densify a tifxyz grid of any (integer) step to one point per voxel."""
    s = max(1, int(round(step)))
    if abs(s - step) > 0.05: raise ValueError(f"tifxyz step {step:.3f} voxels is not a whole number")
    return grow.densify(P, ok, s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("surfaces", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--scroll"); ap.add_argument("--ct"); ap.add_argument("--um", type=float); ap.add_argument("--ct-level", type=int, default=0)
    ap.add_argument("--crop", type=int, nargs=4); ap.add_argument("--device", default="cpu"); ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--weights", default="data/weights/reader-v3.pth"); ap.add_argument("--villa", default="ext/villa-mip"); ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args(argv)
    if a.scroll:
        reg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scrolls.json")))[a.scroll]
        a.ct = a.ct or reg["ct"]; a.um = a.um or reg.get("um")
    if not a.ct: sys.exit("name a --scroll from scrolls.json or pass --ct")
    um = a.um or 8.64; os.makedirs(a.out, exist_ok=True); t0 = time.time()
    json.dump(dict(scroll=a.scroll, ct=a.ct, um=um, source="score_tifxyz", surfaces=a.surfaces, crop=a.crop), open(os.path.join(a.out, "run.json"), "w"), indent=1)
    import reader
    ct = LazyVol(a.ct, level=a.ct_level); cm = reader.load_reader(a.weights, a.villa)
    tsv = os.path.join(a.out, "scores.tsv"); names = os.path.join(a.out, "surfaces.tsv")
    with open(tsv, "w") as f: f.write("id\tseed_z\tseed_y\tseed_x\tstatus\ttracked\tarea_cm2\tfrac_a\tfrac_b\tp995_a\tp995_b\tstroke_a\tstroke_b\tsecs\n")
    with open(names, "w") as f: f.write("id\ttifxyz\n")
    for i, path in enumerate(a.surfaces):
        t = time.time(); P, ok, step = load_tifxyz(path, a.crop)
        open(names, "a").write(f"{i}\t{os.path.abspath(path)}\n")
        c = P[ok].mean(0) if ok.any() else np.zeros(3)
        if ok.sum() < 4:
            open(tsv, "a").write(f"{i}\t{c[0]:.0f}\t{c[1]:.0f}\t{c[2]:.0f}\tsmall\n"); continue
        D, valid, Nd = to_dense(P, ok, step)
        st = grow.render_fast(ct, D, Nd, 17, valid=valid); st[:, ~valid] = 0
        res = {}
        for tag, rev in (("a", False), ("b", True)):
            o = (reader.run_fused if a.fuse else reader.run_stack)(cm, st, layer_start=0, reverse=rev, overlap=0.5, batch=a.batch, device=a.device)
            v = o[valid]; res[tag] = ((v > 0.5).mean(), np.percentile(v, 99.5), stroke(o, valid))
            img = (np.clip((o - 0.25) / 0.5, 0, 1) * 255).astype(np.uint8); img[~valid] = 0
            Image.fromarray(img).resize((max(1, img.shape[1] // 4), max(1, img.shape[0] // 4)), Image.LANCZOS).save(os.path.join(a.out, f"p{i:04d}_{tag}.png"))
        G = D[::4, ::4]; g_ok = valid[::4, ::4]; n0 = Nd[valid].mean(0); n0 = n0 / max(np.linalg.norm(n0), 1e-9)
        np.savez_compressed(os.path.join(a.out, f"p{i:04d}.npz"), P=G.astype(np.float32), ok=g_ok, n0=n0)
        area = valid.sum() * (um * 1e-4) ** 2
        open(tsv, "a").write("\t".join([str(i), f"{c[0]:.0f}", f"{c[1]:.0f}", f"{c[2]:.0f}", "ok"] + [f"{v:.4f}" for v in
                             (ok.mean(), area, res["a"][0], res["b"][0], res["a"][1], res["b"][1], res["a"][2], res["b"][2], time.time() - t)]) + "\n")
        print(f"[{time.time() - t0:6.0f}s] {path}: {area:.2f} cm2 read, bright a/b {res['a'][0]:.3f}/{res['b'][0]:.3f}", flush=True)
        ct.drop(1500)
    print(f"done -> {a.out}; rank with: prospect.py rank {a.out}")


if __name__ == "__main__":
    main()
