"""Snap a surface onto the papyrus fibres: move every node of a tifxyz along its normal to the horizontal-fibre ply that
the AFV V2 fibre model finds in the CT (the idea behind the PHerc0846A winners' fibre-snapped mesh, simplified: no
fibre tracing or hand correction).

usage: fibre_snap.py <in.tifxyz> --out OUT.tifxyz (--scroll S | --ct ZARR) [--afv data/weights/afv-v2] [--crop r0 r1 c0 c1]
                     [--radius 6] [--stride 4] [--sigma 40] [--min-peak 0.3] [--cache DIR]
Per sampled point (every --stride voxels on the surface) the horizontal-fibre probability is read at -R..R voxels along
the normal; a point counts when it has an interior peak of at least --min-peak. The offsets of counted points are
smoothed (Gaussian, --sigma voxels on the surface) and filled where no point counts nearby (offset 0 there), and the
nodes are moved by that smooth offset. Writes OUT.tifxyz (same grid, same step; nodes outside --crop unchanged) and
OUT.tifxyz/snap.json: points, share with a confident peak, mean / p95 |shift|, AFV tiles run, seconds.
AFV runs on 128^3 CT blocks (96^3 kept); blocks are cached in --cache (default OUT.tifxyz/../afv-tiles) so a rerun
or a neighbouring surface reuses them. CPU: about 6 s per block on 4 cores, about 1000 blocks for a 5 cm2 surface.
A shift under 1 voxel everywhere means the surface already sits on the fibres.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grow
from export_tifxyz import write_tifxyz
from score_tifxyz import load_tifxyz

CORE, MARG = 96, 16


def snap_offsets(D, valid, Nd, prob, radius=6, stride=4, sigma=40.0, min_peak=0.3):
    """Smooth offset field (voxels along the normal) on the dense surface grid, from prob(points (N,3) zyx) -> (N,)
    horizontal-fibre probability in 0..1. Returns field (H/stride, W/stride), sample validity, confident mask, peak."""
    ys, xs = np.mgrid[0:D.shape[0]:stride, 0:D.shape[1]:stride]; sp = D[ys, xs]; sn = Nd[ys, xs]; sv = valid[ys, xs]
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    Q = sp[None] + offs[:, None, None, None] * sn[None]
    hz = np.zeros(Q.shape[:3], np.float32); m = np.broadcast_to(sv[None], Q.shape[:3])
    hz[m] = prob(Q[m])
    hz = ndi.gaussian_filter1d(hz, 1.0, axis=0); k = hz.argmax(0); peak = hz.max(0)
    conf = sv & (peak >= min_peak) & (k > 0) & (k < len(offs) - 1)
    s = sigma / stride; num = ndi.gaussian_filter(np.where(conf, offs[k], 0.0), s); den = ndi.gaussian_filter(conf.astype(np.float64), s)
    return np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0), sv, conf, peak


def move_nodes(P, ok, Nd, field, step, stride):
    """Move coarse nodes P (h, w, 3) along the dense normals by the offset field sampled at their dense positions."""
    h, w = P.shape[:2]; H, W = Nd.shape[:2]
    r = np.minimum(np.arange(h) * step, H - 1); c = np.minimum(np.arange(w) * step, W - 1)
    N = Nd[r][:, c]
    f = ndi.map_coordinates(field, np.meshgrid(r / stride, c / stride, indexing="ij"), order=1, mode="nearest")
    return np.where(ok[..., None], P + f[..., None] * N, P)


class AFV:
    """Horizontal-fibre probability at points, from AFV V2 run on cached CT blocks."""

    def __init__(self, ct, afv_dir, cache):
        import torch, afv_net
        self.torch = torch; self.model = afv_net.load_model(afv_dir); self.norm = afv_net.normalize_ct
        torch.set_num_threads(os.cpu_count() or 4); self.ct = ct; self.cache = cache; os.makedirs(cache, exist_ok=True)
        self.tiles = {}; self.run = 0

    def tile(self, t):
        t = tuple(int(v) for v in t)
        if t in self.tiles: return self.tiles[t]
        fn = os.path.join(self.cache, "%d_%d_%d.npy" % t)
        if os.path.exists(fn): a = np.load(fn)
        else:
            lo = np.array(t) * CORE - MARG; blk = self.ct.block(lo, lo + CORE + 2 * MARG)
            if blk[MARG:-MARG, MARG:-MARG, MARG:-MARG].max() == 0: a = np.zeros((CORE,) * 3, np.uint8)
            else:
                with self.torch.inference_mode():
                    o = self.model(self.norm(self.torch.from_numpy(np.ascontiguousarray(blk))[None, None]))
                a = (self.torch.softmax(o["seg"].float(), 1)[0, 2, MARG:-MARG, MARG:-MARG, MARG:-MARG] * 255 + 0.5).clamp(0, 255).byte().numpy()
            np.save(fn + ".tmp.npy", a); os.replace(fn + ".tmp.npy", fn); self.run += 1
            if self.run % 25 == 0: print(f"  AFV blocks run: {self.run}", flush=True); self.ct.drop()
        self.tiles[t] = a; return a

    def __call__(self, pts):
        f = np.floor(pts).astype(np.int64); t = np.floor_divide(f, CORE); l = f - t * CORE; v = np.zeros(len(pts), np.float32)
        keys, inv = np.unique(t, axis=0, return_inverse=True); inv = inv.reshape(-1)
        print(f"  {len(keys)} AFV blocks cover {len(pts)} samples", flush=True)
        for j, key in enumerate(keys):
            sel = inv == j; a = self.tile(key); ll = l[sel]; v[sel] = a[ll[:, 0], ll[:, 1], ll[:, 2]] / 255.0
        return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("surface"); ap.add_argument("--out", required=True); ap.add_argument("--scroll"); ap.add_argument("--ct")
    ap.add_argument("--ct-level", type=int, default=0); ap.add_argument("--afv", default="data/weights/afv-v2"); ap.add_argument("--crop", type=int, nargs=4)
    ap.add_argument("--radius", type=int, default=6); ap.add_argument("--stride", type=int, default=4); ap.add_argument("--sigma", type=float, default=40.0)
    ap.add_argument("--min-peak", type=float, default=0.3); ap.add_argument("--cache")
    a = ap.parse_args(argv)
    if a.scroll and not a.ct: a.ct = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scrolls.json")))[a.scroll]["ct"]
    if not a.ct: sys.exit("name a --scroll from scrolls.json or pass --ct")
    from lazyvol import LazyVol
    t0 = time.time(); Pall, okall, step = load_tifxyz(a.surface); s = int(round(step))
    r0, r1, c0, c1 = a.crop or (0, Pall.shape[0], 0, Pall.shape[1]); P, ok = Pall[r0:r1, c0:c1], okall[r0:r1, c0:c1]
    D, valid, Nd = grow.densify(P.astype(np.float64), ok, s)
    afv = AFV(LazyVol(a.ct, level=a.ct_level), a.afv, a.cache or os.path.join(os.path.dirname(os.path.abspath(a.out)), "afv-tiles"))
    field, sv, conf, peak = snap_offsets(D, valid, Nd, afv, a.radius, a.stride, a.sigma, a.min_peak)
    P2 = Pall.copy(); P2[r0:r1, c0:c1] = move_nodes(P, ok, Nd, field, s, a.stride)
    meta = json.load(open(os.path.join(a.surface, "meta.json")))
    rep = dict(source=os.path.abspath(a.surface), crop=a.crop, points=int(sv.sum()), confident_share=round(float(conf.sum() / max(1, sv.sum())), 4),
               shift_abs_mean=round(float(np.abs(field[sv]).mean()), 3) if sv.any() else 0.0,
               shift_abs_p95=round(float(np.percentile(np.abs(field[sv]), 95)), 3) if sv.any() else 0.0,
               afv_blocks_run=afv.run, seconds=round(time.time() - t0), radius=a.radius, stride=a.stride, sigma=a.sigma, min_peak=a.min_peak)
    write_tifxyz(P2, okall, a.out, step=step, scroll=meta.get("scroll", ""), scan=meta.get("scan", ""), extra={"fibre_snap": rep})
    json.dump(rep, open(os.path.join(a.out, "snap.json"), "w"), indent=1)
    print("fibre snap:", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
