"""Stitch patches that lie on the same sheet into one flat map.

usage: stitch.py <out dir> <scan dir>... [--top N]
Two patches are joined when their node grids overlap in 3D (nodes within 2.5 voxels). The overlap gives matching grid
positions, an affine map between the two grids is fitted to them, and the maps are chained to the best-scoring patch
of each group. The reads of the same physical face are then laid on one canvas (1 px per grid node, the resolution
of the saved pNNNN_a/b.png). Writes <out>/sheetK_ink.png, sheetK_back.png and sheets.json. Only groups that contain
one of the top N patches (default 30) are written.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runs
from PIL import Image
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
args = sys.argv[1:]; top = 30
if "--top" in args: i = args.index("--top"); top = int(args[i + 1]); del args[i:i + 2]
out = args[0]; os.makedirs(out, exist_ok=True); P = runs.patches(args[1:])
for p in P:
    g = np.load(p["base"] + ".npz"); p["P"], p["ok"], p["n0"] = g["P"].astype(np.float32), g["ok"], g["n0"]; p["c"] = p["P"][p["ok"]].mean(0)
    p["ij"] = np.argwhere(p["ok"]); p["tree"] = cKDTree(p["P"][p["ok"]])
def link(a, b):
    """Affine map from b's grid (i, j) to a's grid, from nodes of b that sit on a's surface. None if they do not overlap."""
    sub = b["ij"][::3]; d, k = a["tree"].query(b["P"][sub[:, 0], sub[:, 1]], distance_upper_bound=2.5); hit = np.isfinite(d)
    if hit.sum() < 150: return None
    src = np.c_[sub[hit], np.ones(hit.sum())]; dst = a["ij"][k[hit]]; M, *_ = np.linalg.lstsq(src, dst, rcond=None)
    res = np.linalg.norm(src @ M - dst, axis=1); good = res < 3 * np.median(res) + 1
    M, *_ = np.linalg.lstsq(src[good], dst[good], rcond=None); res = np.median(np.linalg.norm(src[good] @ M - dst[good], axis=1))
    return (M.T, int(hit.sum())) if res < 2.0 and abs(abs(np.linalg.det(M[:2])) - 1) < 0.25 else None   # M.T: 2x3
edges = {}
for x in range(len(P)):
    for y in range(x + 1, len(P)):
        if np.linalg.norm(P[x]["c"] - P[y]["c"]) > 1.5 * P[x]["P"].shape[0] * 4: continue
        l = link(P[x], P[y])
        if l: edges.setdefault(x, {})[y] = l[0]; A = np.vstack([l[0], [0, 0, 1]]); edges.setdefault(y, {})[x] = np.linalg.inv(A)[:2]
seen = set(); sheets = []
for root in range(min(top, len(P))):
    if root in seen or root not in edges: continue
    T = {root: np.c_[np.eye(2), np.zeros(2)]}; q = [root]                       # patch grid -> root grid
    while q:
        x = q.pop(0)
        for y, M in edges[x].items():
            if y not in T: T[y] = (np.vstack([T[x], [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2]; q.append(y)
    seen |= set(T); G = P[root]["P"].shape[0]; cor = np.array([[0, 0, 1], [0, G, 1], [G, 0, 1], [G, G, 1]], float)
    allc = np.vstack([cor @ T[k].T for k in T]); lo = np.floor(allc.min(0)).astype(int); size = np.ceil(allc.max(0)).astype(int) - lo
    if size.max() > 6000: continue
    ink = P[root]["side"]; rn = P[root]["n0"] * (1 if ink == "a" else -1); can = {"ink": np.zeros(size, np.float32), "back": np.zeros(size, np.float32)}; wsum = np.zeros(size, np.float32)
    for k in sorted(T, key=lambda k: -P[k]["ts"]):
        same = float(np.dot(P[k]["n0"], rn)) > 0; A = np.vstack([T[k], [0, 0, 1]]); A[:2, 2] -= lo; Ai = np.linalg.inv(A)
        w = ndi.affine_transform((P[k]["ok"][:-1, :-1]).astype(np.float32), Ai[:2, :2], Ai[:2, 2], output_shape=tuple(size), order=1)
        for name, face in (("ink", "a" if same else "b"), ("back", "b" if same else "a")):
            img = np.asarray(Image.open(f"{P[k]['base']}_{face}.png"), np.float32)
            can[name] += w * ndi.affine_transform(img, Ai[:2, :2], Ai[:2, 2], output_shape=tuple(size), order=1)
        wsum += w
    n = len(sheets) + 1
    um = P[root]["um"] * 4
    for name in can:
        im = np.clip(can[name] / np.maximum(wsum, 1e-3), 0, 255).astype(np.uint8); Image.fromarray(im).save(f"{out}/sheet{n}_{name}.png")
    sheets.append(dict(sheet=n, patches=[P[k]["id"] for k in T], root=P[root]["id"], height_mm=round(size[0] * um / 1000, 1), width_mm=round(size[1] * um / 1000, 1),
                                               area_cm2=round(float((wsum > 0.5).sum()) * (um / 1e4) ** 2, 2)))
    print(f"sheet {n}: {len(T)} patches from {P[root]['id']}, {sheets[-1]['height_mm']} x {sheets[-1]['width_mm']} mm, {sheets[-1]['area_cm2']} cm2", flush=True)
json.dump(sheets, open(out + "/sheets.json", "w"), indent=1); print(f"{len(sheets)} stitched sheets -> {out}")
