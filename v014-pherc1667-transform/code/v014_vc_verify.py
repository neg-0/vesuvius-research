"""V-014 verification with villa's own tools (vc_transform_geom, vc_render_tifxyz, check_transform.py).

Per labelled PHerc1667 segment:
  1. vc_transform_geom -i <2.399 mesh> -a transform.json --invert  ->  compare with the PUBLISHED 1.129 mesh
     (does villa's tool + the published matrix reproduce the published mesh?)
  2. vc_render_tifxyz on the published 1.129 mesh (L1, scale 1) for a 512 x 1024 crop at the V-014 window
     centre  ->  compare with the PUBLISHED 1.129 L1 surface volume (does the render reproduce production?)
  3. same render with --affine transform.json transform_refit.json:inv (refit from check_transform.py --refit)
  4. NCC of before (2) and after (3) surface slices against the published 2.399 surface volume over the same
     surface patch (best over its 21 central layers and +-40 px shifts, plus zero shift at its centre layer).
Env: VC_BIN (dir with the two binaries), V014_DATA (V-014 data dir with mesh_2399), V014_PLACEMENT (ct-placement JSON,
for the window centres), CHECK_TRANSFORM (villa check_transform.py), WORK (scratch).
usage: v014_vc_verify.py <segment> [...]
"""
import json, os, subprocess, sys, urllib.request
import numpy as np, tifffile, cv2
from scipy.spatial import cKDTree

B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667'
VOL = f'{B}/volumes/20260323082859-1.129um-0.2m-59keV-masked.zarr'
VC = os.environ.get('VC_BIN', '/home/user/vc-build/bin')
DATA = os.environ['V014_DATA']
WORK = os.environ.get('WORK', 'work_vc')
RUN = os.environ.get('PUZZLE_RUN_DIR', '.')
H, W = 512, 1024


def get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + '.part'); os.replace(path + '.part', path)
    return path


def xyz(d):
    return np.stack([tifffile.imread(os.path.join(d, c + '.tif')) for c in 'xyz'], -1).astype(np.float64)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def fetch_box(url, z0, z1, y0, y1, x0, x1):
    import zarr, zarr.storage
    a = zarr.open_array(zarr.storage.FsspecStore.from_url(url, read_only=True), mode='r', zarr_format=2)
    return np.asarray(a[z0:z1, y0:y1, x0:x1])


def render(mesh, out, cx, cy, extra=()):
    if not os.path.exists(os.path.join(out, '02.tif')):
        subprocess.check_call([os.path.join(VC, 'vc_render_tifxyz'), '-v', os.path.join(WORK, 'cache_v1129.zarr'), '--remote-url', VOL,
                               '-g', '1', '--scale', '1', '-s', mesh, '-n', '5', '--crop-x', str(cx), '--crop-y', str(cy),
                               '--crop-width', str(W), '--crop-height', str(H), '--tif-output', out, '--cache-gb', '2', *extra],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.stack([tifffile.imread(os.path.join(out, f'{k:02d}.tif')) for k in range(5)]).astype(np.float32)


def main():
    os.makedirs(WORK, exist_ok=True)
    tj = get(f'{VOL}/transform.json', os.path.join(WORK, 'transform.json'))
    rj = os.path.join(WORK, 'transform_refit.json')
    if not os.path.exists(rj):
        subprocess.run([sys.executable, os.environ['CHECK_TRANSFORM'], tj, '--refit', rj], check=False)
    M = np.array(json.load(open(tj))['transformation_matrix'])
    out = {'transform': tj, 'refit': json.load(open(rj))['transformation_matrix'], 'segments': {}}
    for seg in sys.argv[1:]:
        ts = seg.split('-')[0]; wd = os.path.join(WORK, seg); os.makedirs(wd, exist_ok=True)
        m2399 = os.path.join(DATA, seg, 'mesh_2399')
        pub = os.path.join(wd, 'published_1129.tifxyz')
        for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
            get(f'{B}/segments/{seg}/mesh/{ts}-on-20260323082859-1.129um.tifxyz/{f}', os.path.join(pub, f))
        # 1. villa's vc_transform_geom with the published matrix vs the published mesh
        vcp = os.path.join(wd, 'vc_transform_geom_pub.tifxyz')
        if not os.path.exists(os.path.join(vcp, 'x.tif')):
            subprocess.check_call([os.path.join(VC, 'vc_transform_geom'), '-i', m2399, '-o', vcp, '-a', tj, '--invert'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        q, g = xyz(pub), xyz(vcp)
        res = {}
        if q.shape == g.shape:
            v = (q > -0.5).all(-1) & (g > -0.5).all(-1)
            dd = np.linalg.norm(q[v] - g[v], axis=1)
            res['vc_transform_geom_vs_published_mesh'] = {'median_vox': float(np.median(dd)), 'p99_vox': float(np.percentile(dd, 99)),
                                                          'n': int(v.sum())}
        else:
            res['vc_transform_geom_vs_published_mesh'] = {'shape_mismatch': [list(q.shape), list(g.shape)]}
        # crop centre: V-014 window centre on the 2.399 grid (from the CT placement payload) -> nearest published 1.129 vertex
        r0, r1, c0, c1 = json.load(open(os.environ['V014_PLACEMENT']))['segments'][seg]['crop_cells']
        p = xyz(m2399); target = p[(r0 + r1) // 2, (c0 + c1) // 2]
        vq = (q > -0.5).all(-1); qi, qj = np.nonzero(vq)
        Pq = q[vq] @ M[:, :3].T + M[:, 3]
        k = int(np.argmin(np.linalg.norm(Pq - target, axis=1)))
        cy, cx = qi[k] * 10 - H // 2, qj[k] * 10 - W // 2               # L1 canvas: 10 px per grid cell
        res['crop_1129_L1'] = [int(cy), int(cy + H), int(cx), int(cx + W)]
        # 2. published-matrix render vs the published surface volume
        before = render(pub, os.path.join(wd, 'render_before'), cx, cy)
        sv = fetch_box(f'{B}/segments/{seg}/surface-volumes/1.129um-0.22m-59keV-volume-20260323082859-L1.zarr/0', 0, 116, cy, cy + H, cx, cx + W)
        r = [max(ncc(before[s], sv[L].astype(np.float32)) for L in range(116)) for s in range(5)]
        res['render_vs_published_surface_volume_ncc'] = r
        # 3. refit render
        after = render(pub, os.path.join(wd, 'render_after'), cx, cy, ['--affine', tj, rj + ':inv'])
        # 4. reference: published 2.399 surface volume over the same patch (corner vertices via nearest 2.399 vertex)
        vp = (p > -0.5).all(-1); pi_, pj_ = np.nonzero(vp); tree = cKDTree(p[vp])
        cs = []
        for (yy, xx) in [(cy, cx), (cy + H - 1, cx + W - 1)]:
            qq = q[min(yy // 10, q.shape[0] - 1), min(xx // 10, q.shape[1] - 1)]
            _, kk = tree.query(M[:, :3] @ qq + M[:, 3]); cs.append((pi_[kk] * 20, pj_[kk] * 20))
        (ry0, rx0), (ry1, rx1) = cs
        ref = fetch_box(f'{B}/segments/{seg}/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/0', 44, 65,
                        min(ry0, ry1), max(ry0, ry1), min(rx0, rx1), max(rx0, rx1)).astype(np.float32)
        ref = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR) for x in ref])
        m = 40
        for name, X in (('before', before), ('after', after)):
            x = X[2]; best = (-2.0, 0, 0, 0)
            for L in range(ref.shape[0]):
                for dy in range(-m, m + 1, 4):
                    for dx in range(-m, m + 1, 4):
                        s = ncc(ref[L][m:H - m, m:W - m], x[m + dy:H - m + dy, m + dx:W - m + dx])
                        if s > best[0]:
                            best = (s, L, dy, dx)
            res[f'{name}_vs_2399_reference'] = {'best_ncc': best[0], 'ref_layer_dy_dx': list(best[1:]),
                                                'ncc_zero_shift_centre': ncc(ref[10][m:H - m, m:W - m], x[m:H - m, m:W - m])}
        def norm(im):
            nz = im[im > 0]
            lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0, 1))
            return np.clip((im - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
        lab = lambda im, t: cv2.putText(cv2.copyMakeBorder(norm(im), 26, 4, 4, 4, cv2.BORDER_CONSTANT, value=255), t, (6, 19),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
        os.makedirs(os.path.join(RUN, 'payload'), exist_ok=True)
        cv2.imwrite(os.path.join(RUN, 'payload', f'{seg.split("-")[1][:4]}_vc_before_after.png'),
                    np.vstack([lab(ref[10], f'{seg.split("-")[1][:4]} reference: published 2.399 um surface volume'),
                               lab(before[2], 'before: vc_render_tifxyz, 1.129 um scan, published matrix'),
                               lab(after[2], 'after: + --affine transform.json transform_refit.json:inv')]))
        out['segments'][seg] = res
        print(seg, json.dumps({k: res[k] for k in res if k != 'crop_1129_L1'}), flush=True)
        json.dump(out, open(os.path.join(RUN, 'payload', 'v014-vc-verify.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
