"""Mesh-free local surface growth on a predicted-surface volume, then CT layer rendering.

grow(): wavefront growth of a (G,G,3) grid of 3D points with ~`step` voxel spacing, starting at a seed on a
predicted sheet. Each new node is extrapolated from tracked neighbours and snapped along the local normal to the
sheet ridge. Nodes where the sheet is lost stay NaN and are not grown through.
"""
import numpy as np
from scipy import ndimage as ndi

def _unit(v, axis=-1): return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)

def seed_frame(surf, p, r=14):
    """Snap p to the nearest sheet voxel and fit normal/U/V by PCA of sheet voxels in a ball."""
    g = np.arange(-r, r + 1); zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
    off = np.stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float32)
    vals = surf.sample(np.asarray(p, np.float32)[:, None] + off)
    on = vals > 127
    if not on.any(): return None
    d = np.linalg.norm(off[:, on], axis=0); p0 = np.asarray(p, np.float32) + off[:, on][:, d.argmin()]
    vals = surf.sample(p0[:, None] + off).reshape(len(g), len(g), len(g)) > 127
    lab, _ = ndi.label(vals, structure=np.ones((3, 3, 3)))
    idx = np.argwhere(lab == lab[r, r, r]).astype(np.float32) - r
    idx = idx[np.linalg.norm(idx, axis=1) <= r]
    if len(idx) < 3: return None          # a speck of sheet: too few voxels to fit a plane (np.cov of one point is NaN)
    w, v = np.linalg.eigh(np.cov(idx.T)); n = v[:, 0]
    ez = np.array([1.0, 0, 0]); V = ez - n * (ez @ n)
    if np.linalg.norm(V) < 0.3: ez = np.array([0, 1.0, 0]); V = ez - n * (ez @ n)
    V = _unit(V); U = _unit(np.cross(V, n))
    return p0.astype(np.float64), n, U, V, float(d.min()), float(w[0] / max(w[1], 1e-9))

def grow(surf, p0, n0, U0, V0, half_nodes=96, step=4.0, search=2.5, max_jump=1.75, thr=90.0, log=None):
    G = 2 * half_nodes + 1; c = half_nodes
    P = np.full((G, G, 3), np.nan); N = np.zeros((G, G, 3)); U = np.zeros((G, G, 3)); V = np.zeros((G, G, 3))
    done = np.zeros((G, G), bool)          # visited (tracked or rejected)
    P[c, c] = p0; N[c, c] = n0; U[c, c] = U0; V[c, c] = V0; done[c, c] = True
    ok = np.zeros((G, G), bool); ok[c, c] = True
    ws = np.arange(-search, search + 1e-6, 0.5)
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    wave = 0
    while True:
        # frontier: unvisited nodes with a tracked 4-neighbour
        nb_ok = np.zeros((G, G), bool)
        nb_ok[1:] |= ok[:-1]; nb_ok[:-1] |= ok[1:]; nb_ok[:, 1:] |= ok[:, :-1]; nb_ok[:, :-1] |= ok[:, 1:]
        fr = np.argwhere(nb_ok & np.logical_not(done))
        if len(fr) == 0: break
        pred = np.zeros((len(fr), 3)); nrm = np.zeros((len(fr), 3)); uu = np.zeros((len(fr), 3)); vv = np.zeros((len(fr), 3))
        cnt = np.zeros(len(fr))
        for di, dj in nbrs:
            a = fr[:, 0] + di; b = fr[:, 1] + dj
            m = (a >= 0) & (a < G) & (b >= 0) & (b < G)
            m[m] &= ok[a[m], b[m]]
            if not m.any(): continue
            am, bm = a[m], b[m]
            # neighbour at (a,b); we are at (a-di, b-dj): step is -(di*V + dj*U) from the neighbour
            lin = P[am, bm] - step * (di * V[am, bm] + dj * U[am, bm])
            # second-order: use the node beyond the neighbour if tracked
            a2, b2 = am + di, bm + dj
            m2 = (a2 >= 0) & (a2 < G) & (b2 >= 0) & (b2 < G); m2[m2] &= ok[a2[m2], b2[m2]]
            ext = lin.copy(); ext[m2] = 0.5 * lin[m2] + 0.5 * (2 * P[am[m2], bm[m2]] - P[a2[m2], b2[m2]])
            pred[m] += ext; nrm[m] += N[am, bm]; uu[m] += U[am, bm]; vv[m] += V[am, bm]; cnt[m] += 1
        pred /= cnt[:, None]; nrm = _unit(nrm)
        pts = pred[:, None, :] + ws[None, :, None] * nrm[:, None, :]            # (F, W, 3)
        vals = surf.sample(pts.reshape(-1, 3).T).reshape(len(fr), len(ws))
        wgt = vals * np.exp(-0.5 * (ws[None] / 1.5) ** 2)
        peak = vals.max(1); off = (wgt * ws[None]).sum(1) / np.maximum(wgt.sum(1), 1e-6)
        good = (peak >= thr) & (np.abs(off) <= max_jump)
        newP = pred + off[:, None] * nrm
        i, j = fr[:, 0], fr[:, 1]
        done[i, j] = True
        gi, gj = i[good], j[good]
        P[gi, gj] = newP[good]; ok[gi, gj] = True
        N[gi, gj] = nrm[good]
        Ut = uu[good] - (uu[good] * nrm[good]).sum(1, keepdims=True) * nrm[good]; Ut = _unit(Ut)
        Vt = _unit(np.cross(nrm[good], Ut)); sgn = np.sign((Vt * vv[good]).sum(1, keepdims=True)); sgn[sgn == 0] = 1
        U[gi, gj] = Ut; V[gi, gj] = Vt * sgn
        # refresh normals of newly tracked nodes from grid geometry where possible
        if wave % 2 == 1: _refresh_normals(P, N, U, V, ok)
        wave += 1
        if log and wave % 20 == 0: log(f"wave {wave} tracked {ok.sum()} frontier {len(fr)}")
    return P, ok, N

def _refresh_normals(P, N, U, V, ok):
    Pf = np.where(ok[..., None], P, np.nan)
    dI = np.full_like(P, np.nan); dJ = np.full_like(P, np.nan)
    dI[1:-1] = Pf[2:] - Pf[:-2]; dJ[:, 1:-1] = Pf[:, 2:] - Pf[:, :-2]
    m = ok & np.isfinite(dI).all(-1) & np.isfinite(dJ).all(-1)
    if not m.any(): return
    n = _unit(np.cross(dJ[m], dI[m])); s = np.sign((n * N[m]).sum(1, keepdims=True)); s[s == 0] = 1; n = n * s
    n = _unit(0.5 * n + 0.5 * N[m]); N[m] = n
    Ut = _unit(U[m] - (U[m] * n).sum(1, keepdims=True) * n); Vt = np.cross(n, Ut)
    s = np.sign((Vt * V[m]).sum(1, keepdims=True)); s[s == 0] = 1
    U[m] = Ut; V[m] = Vt * s

def densify(P, ok, step):
    """Coarse (G0,G1,3) grid with nodes `step` voxels apart -> dense 1-voxel (H,W,3) points, validity, unit normals. Holes are filled smoothly for sampling only."""
    G0, G1 = P.shape[:2]; w = ok.astype(np.float64)
    Pz = np.where(ok[..., None], P, 0.0)
    num = np.stack([ndi.gaussian_filter(Pz[..., k], 1.0) for k in range(3)], -1); den = ndi.gaussian_filter(w, 1.0)
    Ps = np.where(den[..., None] > 0.05, num / np.maximum(den[..., None], 1e-6), np.nan)
    bad = np.isnan(Ps[..., 0])
    if bad.any():
        ind = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True); Ps = Ps[tuple(ind)]
    s = int(step); H, W = (G0 - 1) * s, (G1 - 1) * s
    yy, xx = np.mgrid[0:H, 0:W] / s
    D = np.stack([ndi.map_coordinates(Ps[..., k], [yy, xx], order=3, mode="nearest") for k in range(3)], -1)
    valid = ndi.map_coordinates(w, [yy, xx], order=1, mode="nearest") > 0.6
    dI, dJ = np.gradient(D, axis=0), np.gradient(D, axis=1)
    Nd = _unit(np.cross(dJ, dI))
    return D.astype(np.float32), valid, Nd.astype(np.float32)

def render(ct, D, Nd, layers=33, log=None):
    H, W, _ = D.shape; out = np.zeros((layers, H, W), np.uint8)
    for li, d in enumerate(range(-(layers // 2), layers // 2 + 1)):
        pts = (D + d * Nd).reshape(-1, 3).T
        out[li] = np.clip(ct.sample(pts), 0, 255).reshape(H, W)
        if log: log(f"layer {li + 1}/{layers}")
    return out

def render_fast(ct, D, Nd, layers=17, tile=256, valid=None):
    """Tile-wise render using dense bounding blocks. Tiles with no valid pixel are skipped."""
    H, W, _ = D.shape; out = np.zeros((layers, H, W), np.uint8); ds = np.arange(-(layers // 2), layers // 2 + 1, dtype=np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            if valid is not None and not valid[y:y + tile, x:x + tile].any(): continue
            d = D[y:y + tile, x:x + tile]; n = Nd[y:y + tile, x:x + tile]
            pts = (d[None] + ds[:, None, None, None] * n[None]).reshape(-1, 3).T
            if np.prod(np.ceil(pts.max(1)) - np.floor(pts.min(1)) + 3) > 1.2e9:      # degenerate tile, fall back
                v = ct.sample(pts)
            else:
                v = ct.sample_dense(pts)
            out[:, y:y + tile, x:x + tile] = np.clip(v, 0, 255).reshape(len(ds), d.shape[0], d.shape[1])
    return out
