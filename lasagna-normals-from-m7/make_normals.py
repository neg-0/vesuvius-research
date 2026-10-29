"""Lasagna-format surface normals (nx, ny, grad_mag) from the public m7 surface prediction.

Method (ours, not the organisers' UNet): structure tensor of the Gaussian-smoothed m7 probability at level 2
(scaledown 4). The sheet normal is the eigenvector of the largest eigenvalue. Encoding matches
villa lasagna/normal_encoding.py: hemisphere flip so nz >= 0, then u8 = round(n * 127 + 128).
grad_mag is one per-scroll constant, 1000 * K / S, where S is the median m7 crest spacing along our normals over up to 12 blocks of the first slab
(level-2 voxels) and K is calibrated on PHerc1218 against the organisers' lasagna (default 0.464: PHerc1218 slab z 2880-2944, our median spacing 4.50 vs the organisers' median grad_mag 103).
Voxels with no m7 support get 0 in all three channels (the same as a missing chunk in the organisers' store).

Usage:
  python3 -I make_normals.py --m7 <m7 .zarr root URL or path> --scroll PHerc1218 --z0 <L2> --z1 <L2> --out <dir>
     [--yx y0 y1 x0 x1] [--workers N] [--k 0.464] [--gm-u8 V]
Writes <out>/<scroll>_{nx,ny,grad_mag}.ome.zarr with dataset "2" (scale 4), 32^3 blosc-lz4 chunks, '/' keys.
"""
import argparse, json, os, sys, time, urllib.request, concurrent.futures as cf
import numpy as np, numcodecs
from scipy import ndimage as nd
from scipy.signal import find_peaks

SG, RHO, RHO2, HALO, TILE, CH = 1.0, 3.0, 10.0, 24, 128, 32
TRACE_MIN = 1e-6


def _read(base, rel):
    if base.startswith('http'):
        for i in range(5):
            try:
                return urllib.request.urlopen(base + '/' + rel, timeout=120).read()
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    return None
                time.sleep(2 ** i)
            except Exception:
                time.sleep(2 ** i)
        raise RuntimeError('fetch failed ' + rel)
    p = os.path.join(base, rel)
    return open(p, 'rb').read() if os.path.exists(p) else None


class Src:
    def __init__(self, base):
        self.base = base
        self.meta = json.loads(_read(base, '.zarray'))
        self.shape, self.c = self.meta['shape'], self.meta['chunks']
        self.codec = numcodecs.get_codec(self.meta['compressor'])

    def box(self, z0, z1, y0, y1, x0, x1):
        c = self.c
        z0c, y0c, x0c = max(z0, 0), max(y0, 0), max(x0, 0)
        z1c, y1c, x1c = min(z1, self.shape[0]), min(y1, self.shape[1]), min(x1, self.shape[2])
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        keys = [(i, j, k) for i in range(z0c // c[0], (z1c - 1) // c[0] + 1)
                for j in range(y0c // c[1], (y1c - 1) // c[1] + 1) for k in range(x0c // c[2], (x1c - 1) // c[2] + 1)]

        def get(key):
            raw = _read(self.base, '%d/%d/%d' % key)
            return key, None if raw is None else np.frombuffer(self.codec.decode(raw), np.uint8).reshape(c)
        with cf.ThreadPoolExecutor(32) as ex:
            for (i, j, k), b in ex.map(get, keys):
                if b is None:
                    continue
                zs, ys, xs = i * c[0], j * c[1], k * c[2]
                a0, b0, c0 = max(zs, z0c), max(ys, y0c), max(xs, x0c)
                a1, b1, c1 = min(zs + c[0], z1c), min(ys + c[1], y1c), min(xs + c[2], x1c)
                out[a0 - z0:a1 - z0, b0 - y0:b1 - y0, c0 - x0:c1 - x0] = b[a0 - zs:a1 - zs, b0 - ys:b1 - ys, c0 - xs:c1 - xs]
        return out


def tensor_normals(m):
    """m: float32 (z,y,x) in [0,1]. Returns (n (z,y,x,3) as (nz,ny,nx), ok mask)."""
    s = nd.gaussian_filter(m, SG)
    g = np.gradient(s)
    res, oks = [], []
    for rho in (RHO, RHO2):
        J = np.empty(m.shape + (3, 3), np.float32)
        for a in range(3):
            for b in range(a, 3):
                J[..., a, b] = J[..., b, a] = nd.gaussian_filter(g[a] * g[b], rho)
        tr = J[..., 0, 0] + J[..., 1, 1] + J[..., 2, 2]
        w, v = np.linalg.eigh(J)
        res.append(v[..., :, 2])
        oks.append(tr > TRACE_MIN)
    n = np.where(oks[0][..., None], res[0], res[1])
    return n, oks[0] | oks[1]


def encode(n, ok):
    nz, ny, nx = n[..., 0], n[..., 1], n[..., 2]
    f = np.where(nz < 0, -1.0, 1.0)
    ex = np.clip(np.round(nx * f * 127 + 128), 0, 255).astype(np.uint8)
    ey = np.clip(np.round(ny * f * 127 + 128), 0, 255).astype(np.uint8)
    ex[~ok] = 0
    ey[~ok] = 0
    return ex, ey


def spacing(m, nzyx, ok, nsamp=3000, seed=0):
    """Median m7 crest spacing (L2 voxels) along the normals, from random voxels with m7 support."""
    s = nd.gaussian_filter(m, 1.0)
    rng = np.random.default_rng(seed)
    t = np.arange(-30, 30.25, 0.25)
    out = []
    lo, hi = 32, np.array(m.shape) - 32
    if np.any(hi <= lo):
        return None
    for _ in range(nsamp * 4):
        p = rng.integers(lo, hi)
        if not ok[tuple(p)] or s[tuple(p)] < 0.05:
            continue
        pts = p[:, None] + nzyx[tuple(p)][:, None] * t[None]
        pk, _ = find_peaks(nd.map_coordinates(s, pts, order=1), height=0.2, distance=8)
        if len(pk) >= 3:
            out.append(np.median(np.diff(t[pk])))
        if len(out) >= nsamp:
            break
    return float(np.median(out)) if out else None


def zattrs(name):
    return {"multiscales": [{"version": "0.4", "name": name,
                             "axes": [{"name": a, "type": "space", "unit": "pixel"} for a in "zyx"],
                             "datasets": [{"path": "2", "coordinateTransformations": [{"type": "scale", "scale": [4.0, 4.0, 4.0]}]}]}]}


class Out:
    def __init__(self, root, name, shape):
        self.d = os.path.join(root, name, '2')
        os.makedirs(self.d, exist_ok=True)
        json.dump({"zarr_format": 2}, open(os.path.join(root, name, '.zgroup'), 'w'))
        json.dump(zattrs(name.split('_')[-1].replace('.ome.zarr', '')), open(os.path.join(root, name, '.zattrs'), 'w'), indent=1)
        json.dump({"shape": shape, "chunks": [CH] * 3, "dtype": "|u1", "fill_value": 0, "order": "C", "filters": None,
                   "dimension_separator": "/", "zarr_format": 2,
                   "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}},
                  open(os.path.join(self.d, '.zarray'), 'w'), indent=1)
        self.codec = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=1)

    def write(self, z0, y0, x0, a):
        for i in range(0, a.shape[0], CH):
            for j in range(0, a.shape[1], CH):
                for k in range(0, a.shape[2], CH):
                    b = a[i:i + CH, j:j + CH, k:k + CH]
                    if not b.any():
                        continue
                    if b.shape != (CH, CH, CH):
                        b = np.pad(b, [(0, CH - s) for s in b.shape])
                    p = os.path.join(self.d, '%d' % ((z0 + i) // CH), '%d' % ((y0 + j) // CH))
                    os.makedirs(p, exist_ok=True)
                    open(os.path.join(p, '%d' % ((x0 + k) // CH)), 'wb').write(self.codec.encode(np.ascontiguousarray(b)))


SLAB = None


def do_tile(args):
    zlo, zhi, y0, y1, x0, x1, Y0, X0 = args
    slab = SLAB
    sub = slab[:, max(y0 - HALO - Y0, 0):y1 + HALO - Y0, max(x0 - HALO - X0, 0):x1 + HALO - X0]
    oy, ox = y0 - max(y0 - HALO, Y0), x0 - max(x0 - HALO, X0)
    m = sub.astype(np.float32) / 255.0
    if not m.any():
        return None
    n, ok = tensor_normals(m)
    ex, ey = encode(n, ok)
    sl = (slice(zlo, zhi), slice(oy, oy + y1 - y0), slice(ox, ox + x1 - x0))
    return (y0, x0, ex[sl], ey[sl], ok[sl])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m7', required=True, help='m7 surface .zarr root (URL or path); level 2 is read')
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--z0', type=int, required=True)
    ap.add_argument('--z1', type=int, required=True)
    ap.add_argument('--yx', type=int, nargs=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--k', type=float, default=0.464)
    ap.add_argument('--gm-u8', type=int)
    ap.add_argument('--slab', type=int, default=64)
    ap.add_argument('--spacing-only', action='store_true')
    a = ap.parse_args()
    src = Src(a.m7.rstrip('/') + '/2')
    Z, Y, X = src.shape
    z0, z1 = a.z0 // CH * CH, -(-a.z1 // CH) * CH
    y0, y1, x0, x1 = a.yx if a.yx else (0, Y, 0, X)
    y0, x0 = y0 // CH * CH, x0 // CH * CH
    outs = {n: Out(a.out, '%s_%s.ome.zarr' % (a.scroll, n), [Z, Y, X]) for n in ('nx', 'ny', 'grad_mag')}
    log = {'m7': a.m7, 'level': 2, 'shape': [Z, Y, X], 'z': [z0, z1], 'yx': [y0, y1, x0, x1], 'sg': SG, 'rho': [RHO, RHO2],
           'k': a.k, 'slabs': []}
    gm_u8, t0 = a.gm_u8, time.time()
    import multiprocessing as mp
    global SLAB
    for zs in range(z0, z1, a.slab):
        ze = min(zs + a.slab, z1)
        slab = src.box(zs - HALO, ze + HALO, max(y0 - HALO, 0), min(y1 + HALO, Y), max(x0 - HALO, 0), min(x1 + HALO, X))
        Y0, X0 = max(y0 - HALO, 0), max(x0 - HALO, 0)
        if gm_u8 is None and slab.any():
            # spacing: median over up to 12 160^2 blocks with the most m7 support, spread over the slab
            cands = []
            for jy in range(0, slab.shape[1] - 160, 160):
                for jx in range(0, slab.shape[2] - 160, 160):
                    f = (slab[:, jy:jy + 160, jx:jx + 160] > 50).mean()
                    if f > 0.05:
                        cands.append((f, jy, jx))
            rng = np.random.default_rng(0)
            pick = [cands[i] for i in sorted(rng.choice(len(cands), min(12, len(cands)), replace=False))] if cands else []
            Ss = []
            for _, jy, jx in pick:
                m = slab[:, jy:jy + 160, jx:jx + 160].astype(np.float32) / 255
                n, ok = tensor_normals(m)
                v = spacing(m, n, ok, nsamp=500)
                if v:
                    Ss.append(v)
            if Ss:
                S = float(np.median(Ss))
                gm_u8 = int(np.clip(round(1000 * a.k / S), 1, 255))
                log['spacing_l2'] = S
                log['spacing_blocks'] = [[zs - HALO, Y0 + jy, X0 + jx, 160] for _, jy, jx in pick]
                log['spacing_per_block'] = Ss
                print('spacing %.2f L2 vox (blocks %s) -> grad_mag u8 %d' % (S, np.round(Ss, 2).tolist(), gm_u8), flush=True)
            if a.spacing_only:
                break
        if gm_u8 is None or a.spacing_only:
            continue
        SLAB = slab
        pool = mp.get_context('fork').Pool(a.workers) if a.workers > 1 else None
        jobs = [(HALO, HALO + ze - zs, ty, min(ty + TILE, y1), tx, min(tx + TILE, x1), Y0, X0)
                for ty in range(y0, y1, TILE) for tx in range(x0, x1, TILE)]
        nvox = 0
        for r in (pool.imap_unordered(do_tile, jobs) if pool else map(do_tile, jobs)):
            if r is None:
                continue
            ty, tx, ex, ey, ok = r
            outs['nx'].write(zs, ty, tx, ex)
            outs['ny'].write(zs, ty, tx, ey)
            outs['grad_mag'].write(zs, ty, tx, np.where(ok, np.uint8(gm_u8), np.uint8(0)))
            nvox += int(ok.sum())
        if pool:
            pool.close(); pool.join()
        log['slabs'].append([zs, ze, nvox, round(time.time() - t0, 1)])
        print('slab %d-%d valid %d  t=%.0fs' % (zs, ze, nvox, time.time() - t0), flush=True)
    log['grad_mag_u8'] = gm_u8
    json.dump(log, open(os.path.join(a.out, 'make_normals.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
