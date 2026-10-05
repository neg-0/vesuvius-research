"""V-018: cross-scan placement check on published surface volumes.

A segment rendered on two scans of the same scroll shares one (u, v) parameterisation, scaled to
micrometres by each surface volume's OME scale. If both meshes sit on the same papyrus, the two
renders show the same structure at the same (u, v, depth). For sampled windows this reports
  ncc0  normalised cross-correlation of the central depth slab with no shift
  best  the max NCC over depth shifts (+-8 grid layers) and in-plane shifts (+-200 um)
on a common grid of twice the coarser scan's voxel size.
usage: sv_placement.py <segment prefix> [--ref SUBSTR] [--windows N] [--size-um UM] [--out file.json]
"""
import argparse
import json
import re
import urllib.request

import numpy as np
from scipy import ndimage as nd
from scipy.signal import fftconvolve

import zread

B = zread.BUCKET
SEARCH_UM, DZR, HS = 200.0, 8, 12         # in-plane search (um), depth search (layers), half slab (layers)


def ls(prefix):
    x = urllib.request.urlopen(f"{B}?list-type=2&prefix={prefix}&delimiter=/").read().decode()
    return [p[len(prefix):] for p in re.findall(r"<Prefix>([^<]*)</Prefix>", x) if p != prefix]


def scales(path):
    a = json.loads(zread._get(B + path + "/.zattrs"))
    return [d["coordinateTransformations"][0]["scale"] for d in a["multiscales"][0]["datasets"]]


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def shift_search(ref, X, dzr, m):
    """Max NCC of ref over every (dz, dy, dx) placement inside X (ref padded by dzr in depth, m in y/x), via FFT."""
    r = ref - ref.mean(); rn = np.sqrt((r * r).sum())
    k = np.ones(ref.shape, np.float32)
    num = fftconvolve(X, r[::-1, ::-1, ::-1], mode="valid")
    s1 = fftconvolve(X, k, mode="valid"); s2 = fftconvolve(X * X, k, mode="valid")
    c = num / (rn * np.sqrt(np.maximum(s2 - s1 * s1 / k.size, 1e-6)))
    i = np.unravel_index(np.argmax(c), c.shape)
    return float(c[i]), int(i[0] - dzr), int(i[1] - m), int(i[2] - m)


def pick_level(sc, grid_um):
    lv = [i for i, s in enumerate(sc) if s[1] <= grid_um * 1.05]
    return lv[-1] if lv else 0


def read_um(path, sc, grid_um, y0_um, x0_um, size_um, nlayers):
    """Read a size_um square at (y0_um, x0_um), nlayers grid layers centred in depth, on grid_um."""
    lv = pick_level(sc, grid_um); s = sc[lv]
    nz = zread.meta(f"{path}/{lv}")["shape"][0]
    zc, hz = nz // 2, int(np.ceil(nlayers * grid_um / s[0] / 2)) + 1
    y0, x0, n = int(y0_um / s[1]), int(x0_um / s[2]), int(np.ceil(size_um / s[1]))
    a = zread.read(f"{path}/{lv}", [zc - hz, y0, x0], [zc + hz, y0 + n, x0 + n]).astype(np.float32)
    a = nd.zoom(a, (s[0] / grid_um, s[1] / grid_um, s[2] / grid_um), order=1)
    c, h = a.shape[0] // 2, min(nlayers, a.shape[0]) // 2
    return a[c - h:c + h]


def windows(path, sc, n, size_um):
    L = len(sc) - 1; sh = zread.meta(f"{path}/{L}")["shape"]
    mip = zread.read(f"{path}/{L}", [sh[0] // 2, 0, 0], [sh[0] // 2 + 1, sh[1], sh[2]])[0] > 0
    ok = nd.binary_erosion(mip, iterations=max(1, int(np.ceil(size_um / sc[L][1]))))
    ys, xs = np.nonzero(ok)
    if len(ys) == 0:
        return []
    idx = np.random.default_rng(0).choice(len(ys), size=min(n, len(ys)), replace=False)
    return [(float(ys[i] * sc[L][1] - size_um / 2), float(xs[i] * sc[L][2] - size_um / 2)) for i in idx]


def check(segment, ref_sub=None, n_windows=6, size_um=1500.0, verbose=True):
    seg = segment.rstrip("/") + "/"
    svs = [p.rstrip("/") for p in ls(seg + "surface-volumes/") if p.rstrip("/").endswith(".zarr")]
    paths = {v: seg + "surface-volumes/" + v for v in svs}
    sc = {v: scales(paths[v]) for v in svs}
    res = {"segment": seg, "pairs": {}}
    if len(svs) < 2:
        res["status"] = "fewer than 2 surface volumes"; return res
    ref = next((v for v in svs if ref_sub and ref_sub in v), None) or min(svs, key=lambda v: abs(sc[v][0][1] - 2.4))
    res["reference"] = ref
    wins = windows(paths[ref], sc[ref], n_windows, size_um)
    if not wins:
        res["status"] = "no reference data"; return res
    for v in svs:
        if v == ref:
            continue
        grid = 2 * max(sc[v][0][1], sc[ref][0][1])
        M = int(np.ceil(SEARCH_UM / grid))
        out = []
        for y_um, x_um in wins:
            try:
                R = read_um(paths[ref], sc[ref], grid, y_um, x_um, size_um, 2 * HS)
                X = read_um(paths[v], sc[v], grid, y_um, x_um, size_um, 2 * HS + 2 * DZR)
            except Exception as e:  # missing chunks, short volumes
                out.append({"window_um": [round(y_um), round(x_um)], "error": str(e)[:200]}); continue
            n = min(R.shape[1], R.shape[2], X.shape[1], X.shape[2])
            hs = min(HS, R.shape[0] // 2, X.shape[0] // 2 - 2)
            dzr = min(DZR, X.shape[0] // 2 - hs)
            if hs < 3 or dzr < 0 or n <= 2 * M + 8:
                out.append({"window_um": [round(y_um), round(x_um)], "skipped": "too few layers or pixels"}); continue
            zr, zx = R.shape[0] // 2, X.shape[0] // 2
            R, X = R[zr - hs:zr + hs, :n, :n], X[zx - hs - dzr:zx + hs + dzr, :n, :n]
            ref_b = R[:, M:n - M, M:n - M]
            if (ref_b > 0).mean() < 0.9 or (X > 0).mean() < 0.9:
                out.append({"window_um": [round(y_um), round(x_um)], "skipped": "under 90% data"}); continue
            n0 = ncc(ref_b, X[dzr:dzr + 2 * hs, M:n - M, M:n - M])
            b = shift_search(ref_b, X, dzr, M)
            out.append({"window_um": [round(y_um), round(x_um)], "grid_um": round(grid, 3), "ncc0": round(n0, 3),
                        "best": round(b[0], 3), "layers": 2 * hs, "dz_range": dzr, "shift_um": [round(t * grid, 1) for t in b[1:]]})
            if verbose:
                print(v[:34], out[-1], flush=True)
        good = [w for w in out if "ncc0" in w]
        res["pairs"][v] = {"grid_um": round(grid, 3), "n": len(good), "windows": out,
                           "median_ncc0": float(np.median([w["ncc0"] for w in good])) if good else None,
                           "median_best": float(np.median([w["best"] for w in good])) if good else None}
    res["status"] = "ok"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segment"); ap.add_argument("--ref"); ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--size-um", type=float, default=1500.0); ap.add_argument("--out")
    a = ap.parse_args()
    res = check(a.segment, a.ref, a.windows, a.size_um)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in p.items() if kk != "windows"} for k, p in res["pairs"].items()}))


if __name__ == "__main__":
    main()
