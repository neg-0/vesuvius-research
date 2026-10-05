"""V-018 cross-check without rendering: compare a segment's coarse-scan mesh with its fine-scan mesh.
Maps the fine mesh into the coarse volume with the fine volume's published transform.json, finds the
nearest fine vertex for each coarse vertex, and reports the grid offset from the naive index map
i_fine = i_coarse * r (r = fine/coarse grid ratio). A block-averaged coarse mesh read with a
corner convention shows an offset near (r - 1) / 2 fine cells.
usage: mesh_offset.py <segment prefix> <coarse mesh> <fine mesh> <volume with transform.json> [fine2coarse|coarse2fine]"""
import io, json, sys, urllib.request
import numpy as np, tifffile
from scipy.spatial import cKDTree

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
get = lambda p: urllib.request.urlopen(B + p).read()


def mesh(seg, name):
    return np.stack([tifffile.imread(io.BytesIO(get(f"{seg}mesh/{name}/{c}.tif"))).astype(np.float64) for c in "xyz"], -1)


def main(seg, coarse, fine, tvol, direction="fine2coarse"):
    seg = seg.rstrip("/") + "/"
    scroll = seg.split("/")[0]
    M = np.array(json.loads(get(f"{scroll}/volumes/{tvol}/transform.json"))["transformation_matrix"])
    A, F = mesh(seg, coarse), mesh(seg, fine)
    ok = lambda P: (P[..., 0] > 0) & np.isfinite(P).all(-1)
    fi, fj = np.nonzero(ok(F)); ai, aj = np.nonzero(ok(A))
    if direction == "fine2coarse":
        d, k = cKDTree((F @ M[:, :3].T + M[:, 3])[ok(F)]).query(A[ok(A)])
    else:
        d, k = cKDTree(F[ok(F)]).query((A @ M[:, :3].T + M[:, 3])[ok(A)])
    r = (F.shape[0] / A.shape[0], F.shape[1] / A.shape[1])
    di, dj = fi[k] - ai * r[0], fj[k] - aj * r[1]
    out = {"segment": seg, "ratio": r, "nn_dist_vox_median": float(np.median(d)),
           "row_offset_fine_cells": float(np.median(di)), "col_offset_fine_cells": float(np.median(dj)),
           "half_block_expected": [(r[0] - 1) / 2, (r[1] - 1) / 2]}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:6])
