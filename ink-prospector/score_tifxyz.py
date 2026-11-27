"""Read and score existing surfaces: any tifxyz segment (VC3D, HerculeX, community or grown with VC's tracer).

usage: score_tifxyz.py --out DIR (--scroll S | --ct ZARR [--um U]) [--crop r0 r1 c0 c1] [--tile N [--min-nodes K]]
                       [--device cuda] [--fuse] <tifxyz dir>...
Each surface is densified to one point per voxel, 17 CT layers are sampled along its normals (as in scan), Reader V3
reads both faces, and the text score is computed. The output is a scan folder (scores.tsv, pNNNN.npz, pNNNN_a/b.png,
run.json), so `prospect.py rank`, `viewer`, `stitch` and `vc_export.py` work on it unchanged. surfaces.tsv maps each
pNNNN to its tifxyz path and grid window. --crop takes a window of the tifxyz grid (rows r0:r1, cols c0:c1).
--tile N cuts each surface into N x N-node tiles (neighbours share their edge row/column, so the densified tiles
abut without gaps; a short last strip joins its neighbour) and scores every tile as its own patch, so a whole HerculeX sheet is ranked tile by tile with
bounded memory; tiles with fewer than --min-nodes valid nodes (default N*N/4) are skipped.
Run from the work folder (PROSPECTOR_HOME), which holds data/weights/reader-v3.pth and ext/villa-mip.
"""
import argparse, functools, json, os, sys, time
import numpy as np, tifffile
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lazyvol import LazyVol
import grow
from textscore import stroke


@functools.lru_cache(maxsize=1)
def _read(path):
    """x, y, z grids and step of one tifxyz folder (the last one read is kept, so tiles of a sheet read it once)."""
    x, y, z = (tifffile.imread(os.path.join(path, a + ".tif")).astype(np.float32) for a in "xyz")
    meta = json.load(open(os.path.join(path, "meta.json")))
    return x, y, z, 1.0 / float(meta.get("scale", [1.0])[0])


def load_tifxyz(path, crop=None):
    """tifxyz folder -> P (rows, cols, 3) z y x, ok mask, step in voxels between nodes."""
    x, y, z, step = _read(os.path.abspath(path))
    if crop: r0, r1, c0, c1 = crop; x, y, z = x[r0:r1, c0:c1], y[r0:r1, c0:c1], z[r0:r1, c0:c1]
    ok = (z > 0) & (x != -1) & (y != -1) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)   # VC treats z <= 0 as invalid
    return np.stack([z, y, x], -1), ok, step


def to_dense(P, ok, step):
    """Densify a tifxyz grid of any (integer) step to one point per voxel."""
    s = max(1, int(round(step)))
    if abs(s - step) > 0.05: raise ValueError(f"tifxyz step {step:.3f} voxels is not a whole number")
    return grow.densify(P, ok, s)


def tiles(shape, n, ok=None, min_nodes=None):
    """Windows (r0, r1, c0, c1) of an n x n tiling of a grid; neighbours share one edge row/column, and a short
    last strip joins the tile before it. With ok, windows holding fewer than min_nodes valid nodes (default
    n*n/4) are left out."""
    n = max(2, int(n)); m = n * n // 4 if min_nodes is None else min_nodes

    def cuts(L):
        s = list(range(0, max(1, L - 1), n - 1))
        if len(s) > 1 and (L - 1) - s[-1] < (n - 1) / 2: s.pop()
        return [(a, b + 1) for a, b in zip(s, s[1:] + [L - 1])]

    out = [(r0, r1, c0, c1) for r0, r1 in cuts(shape[0]) for c0, c1 in cuts(shape[1])]
    return [w for w in out if ok is None or ok[w[0]:w[1], w[2]:w[3]].sum() >= max(m, 4)]


def finished(out):
    """Patches already scored in a scan folder: {id: (scores.tsv line, surfaces.tsv line)}. A patch counts only when both
    files have its row (a run cut off mid-patch has the surfaces row but no score)."""
    try: S = open(os.path.join(out, "scores.tsv")).readlines()[1:]; N = open(os.path.join(out, "surfaces.tsv")).readlines()[1:]
    except FileNotFoundError: return {}
    sc = {int(l.split("\t")[0]): l for l in S if l.endswith("\n") and l.split("\t")[0].isdigit()}
    nm = {int(l.split("\t")[0]): l for l in N if l.endswith("\n") and l.split("\t")[0].isdigit()}
    return {i: (sc[i], nm[i]) for i in sc if i in nm}


def same_job(line, job):
    """Does a surfaces.tsv line describe this (path, window) job? A job with no window matches any recorded window."""
    f = line.rstrip("\n").split("\t"); path, win = job
    return f[1] == os.path.abspath(path) and (win is None or [int(v) for v in f[2:6]] == list(win))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("surfaces", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--scroll"); ap.add_argument("--ct"); ap.add_argument("--um", type=float); ap.add_argument("--ct-level", type=int, default=0)
    ap.add_argument("--crop", type=int, nargs=4); ap.add_argument("--tile", type=int); ap.add_argument("--max-cache-gb", type=float, help="cap on CT chunks held in memory (least recently used are dropped)"); ap.add_argument("--min-nodes", type=int); ap.add_argument("--resume", action="store_true", help="keep patches already scored in --out and score only the rest"); ap.add_argument("--shard", default="0/1", help="i/n: score only patches with id %% n == i (one --out per shard; ids stay global)"); ap.add_argument("--device", default="cpu"); ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--weights", default="data/weights/reader-v3.pth"); ap.add_argument("--villa", default="ext/villa-mip"); ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args(argv)
    if a.scroll:
        reg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scrolls.json")))[a.scroll]
        a.ct = a.ct or reg["ct"]; a.um = a.um or reg.get("um")
    if not a.ct: sys.exit("name a --scroll from scrolls.json or pass --ct")
    um = a.um or 8.64; os.makedirs(a.out, exist_ok=True); t0 = time.time()
    json.dump(dict(scroll=a.scroll, ct=a.ct, um=um, source="score_tifxyz", surfaces=a.surfaces, crop=a.crop, tile=a.tile, shard=a.shard), open(os.path.join(a.out, "run.json"), "w"), indent=1)
    import reader
    ct = LazyVol(a.ct, level=a.ct_level, max_cache_gb=a.max_cache_gb); cm = reader.load_reader(a.weights, a.villa)
    tsv = os.path.join(a.out, "scores.tsv"); names = os.path.join(a.out, "surfaces.tsv")
    done = finished(a.out) if a.resume else {}
    with open(tsv, "w") as f:
        f.write("id\tseed_z\tseed_y\tseed_x\tstatus\ttracked\tarea_cm2\tfrac_a\tfrac_b\tp995_a\tp995_b\tstroke_a\tstroke_b\tsecs\n")
        f.writelines(done[i][0] for i in sorted(done))
    with open(names, "w") as f:
        f.write("id\ttifxyz\tr0\tr1\tc0\tc1\n"); f.writelines(done[i][1] for i in sorted(done))
    jobs = []                                                            # (path, grid window) per patch
    for path in a.surfaces:
        if not a.tile: jobs.append((path, a.crop)); continue
        P, ok, _ = load_tifxyz(path, a.crop); o = a.crop or (0, 0, 0, 0)
        jobs += [(path, (o[0] + w[0], o[0] + w[1], o[2] + w[2], o[2] + w[3])) for w in tiles(ok.shape, a.tile, ok, a.min_nodes)]
        print(f"{path}: {len(jobs)} tiles so far", flush=True)
    if done:
        bad = [i for i in done if i >= len(jobs) or not same_job(done[i][1], jobs[i])]
        if bad: sys.exit(f"--resume: {a.out} was made from other surfaces or tiles (patch {bad[0]} differs); use a new --out")
        print(f"--resume: {len(done)} of {len(jobs)} patches already scored, skipping them", flush=True)
    si, sn = (int(v) for v in a.shard.split("/"))
    if not 0 <= si < sn: sys.exit(f"--shard {a.shard}: need 0 <= i < n")
    if sn > 1: print(f"shard {a.shard}: {sum(1 for i in range(len(jobs)) if i % sn == si)} of {len(jobs)} patches", flush=True)
    for i, (path, win) in enumerate(jobs):
        if i in done or i % sn != si: continue
        t = time.time(); P, ok, step = load_tifxyz(path, win)
        w = win or (0, P.shape[0], 0, P.shape[1])
        open(names, "a").write(f"{i}\t{os.path.abspath(path)}\t{w[0]}\t{w[1]}\t{w[2]}\t{w[3]}\n")
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
        print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(jobs)} {path} {w}: {area:.2f} cm2 read, bright a/b {res['a'][0]:.3f}/{res['b'][0]:.3f}", flush=True)
        ct.drop(1500)
    print(f"done -> {a.out}; rank with: prospect.py rank {a.out}")


if __name__ == "__main__":
    main()
