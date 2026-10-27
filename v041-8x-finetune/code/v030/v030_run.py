"""V-030 driver: V-016 (cross-scan ink adaptation) replicated on PHerc0814 1.129 um, with the organisers' mrg20736-1um
prediction scored on the same held-out windows (descriptive).

Spec: experiments/v030/README.md (registered 2026-10-07T23:05Z, commit 96d90015, before any V-030 read). This script
implements it and does not change it.

Stages (each resumable; a stage whose output exists is skipped; later stages refuse to run after a failed gate):
  step0    CPU. Placement of mesh B (1.129 um) against mesh A (2.399 um) through the published transform: 6 label-chosen
           crops, model-free CT NCC (V-014 addendum statistic); refit with the landmark least-squares matrix (V-014) if
           the published one fails. Prints PLACEMENT_OK / PLACEMENT_REFIT / PLACEMENT_FAIL. Writes step0.json.
  carry    CPU. 2.399 um labels -> 1.129 um L1 canvas by 3D correspondence (nearest mesh-A point + local Jacobian).
           Prints CARRY_OK / CARRY_FAIL (and CARRY SAME / DIFFERENT against the committed carry.json). Writes carry.json.
  plan     CPU, labels and mesh only. 3 z bands, V-020 patches (80 x 160 B cells), folds with a 1 mm guard. Prints
           PLAN SAME / DIFFERENT against the committed windows.json and the ESTIMATE line.
  prep     GPU machine (network). Renders every patch on the 1.129 um volume, level 1 (2.258 um), 63 layers.
  train K  GPU. villa 7a5ba5a finetune_on_scan.py, --amp bf16, defaults, --voxel-um 2.258, one --pair per patch.
  eval K   GPU. Zero-shot canonical vs fine-tuned per held-out window (ds8, offset 0), plus mrg20736-1um (descriptive).
           Prints FOLD lines.
  verdict  CPU. RULE and VERDICT lines, summary.json.
  status   what is done, elapsed GPU-stage time against the 4 h budget, and the estimate for the rest.
  check    (--check) dry run: selftest of the carry and patch maths, env, villa commit, every bucket URL; no download
           larger than a 1 kB range read. Last line CHECK OK or CHECK FAIL [...].

Choices the README left to the driver (repeat them in RESULT.md):
- Step 0 crops: greedy by labelled-ink density over 8 x 16-cell boxes of mesh A whose every vertex (incl. the far row
  and column) is valid, maps (M_pub^-1) inside the 1.129 um level-0 volume (V-014's window restriction) and onto a
  B vertex where the 1.129 um scan has data (b_coverage; README Amendment 1), no overlap; B crop = 16 x 32 B cells centred on the B vertex nearest (3D, through M_pub) to
  the A crop's centre vertex, needing >= 90% valid B vertices. A crop whose renders have < 50% CT-valid pixels counts
  as not passing (reported UNKNOWN).
- The reference render A is mesh B mapped by M_pub into 2.399 um voxels and written as a tifxyz with scale 0.1, so
  render_crop puts it on B's 10 px per cell grid (same 2.258 um in-plane pixels as B); its layer step is 2.399 um.
- Carry: a B vertex is carried if its nearest-point solve stays within one A cell (|delta| <= 1 on both axes) and its
  normal residual is <= 8 voxels (2.399 um). Its (u, v) is upsampled with render_tifxyz.upsample (corner-aligned,
  10 px per B cell, the renderer's own mapping); label pixel = (round(20 u), round(20 v)) on the A canvas.
- Held-out windows: a cell is valid for a band only if all 4 of its corners are in that band, so cells straddling an
  edge are left out (a window's far vertices may still reach just past the edge; `straddles_band_edge`).
- Training patches are chosen per fold (README Amendment 2, V-016's rule): densest 40 x 80-cell patches anywhere whose
  every in-volume vertex is >= GUARD_VOX + 2 * HALF_LAYERS (1014 voxels, 1.14 mm) in 3D from every in-volume vertex
  of that fold's held-out windows. Patches are named r<row>_c<col>, so a patch used twice is rendered once.
- Band edges are the ink-weighted terciles of the B vertex z (1.129 um level 0) over carried ink, rounded to 10 voxels.
- Patches with any uncarried CT pixel keep it at 0 after rendering (unsupervised, unscored); the count is recorded.
- Searched scores (march_bench.eval_map, +-48 ds8 px) are descriptive; its 2 mm band uses 2.399 um (4% off here).

usage: v030_run.py check | step0 [--force] | carry | plan | prep [names] | train K | eval K | verdict | status | demo
Env: V030_DATA (work folder), V030_RESULTS (default projects/vesuvius/results/v030-pherc0814-cross-scan),
V030_SRC (folder with render_crop.py, render_tifxyz.py, infer_crop.py, march_bench.py: the V-016 src tree),
V030_VILLA (neg-0/villa checkout at 7a5ba5a), CANON_CKPT, XSCAN_CACHE, PUZZLE_RUN_DIR, INK_DEVICE (default cuda),
V030_LOWDISK=1 (delete the chunk cache after each step-0 crop; for the cloud).

Desktop run (GPU-QUEUE item 43), from the repo root:

  V=workspaces/vesuvius/villa-diag-7a5ba5a        # V-029's checkout; create it as in item 42 if missing
  git -C $V rev-parse HEAD                        # must print 7a5ba5a01f1ff7070918f2143143c11c6f7268c5
  export V030_VILLA=$PWD/$V V030_DATA=$PWD/workspaces/vesuvius/v030/data \
    V030_SRC=$PWD/workspaces/vesuvius/v013gpu/xscan/src XSCAN_CACHE=$PWD/workspaces/vesuvius/v030/cache \
    CANON_CKPT=$PWD/workspaces/vesuvius/models/r152_3ddec_v2_l5_epoch13.ckpt \
    V030_RESULTS=$PWD/projects/vesuvius/results/v030-pherc0814-cross-scan
  P=$PWD/workspaces/vesuvius/venv-gpu/bin/python
  R() { python3 tools/research.py run vesuvius --hypothesis V-030 --seed 0 --timeout 6600 \
    --cwd projects/vesuvius/experiments/v030 --input v030_run.py --input ../v020/v020_scan_adapt.py \
    --input ../v016/v016_cross_scan_ink.py --input ../v014/v014_transform_ink.py \
    --input $V030_SRC/render_crop.py --input $V030_SRC/render_tifxyz.py --input $V030_SRC/infer_crop.py \
    --input $V030_SRC/march_bench.py \
    --input $V030_VILLA/ink-detection/scan_adapt/finetune_on_scan.py -- $P v030_run.py "$@"; }
  $P projects/vesuvius/experiments/v030/v030_run.py --check   # last line must be CHECK OK
  R step0            # prints the committed PLACEMENT line (it ran in the cloud); stop if PLACEMENT_FAIL
  R carry            # ~2-5 min CPU; must print CARRY SAME and CARRY_OK; stop on CARRY_FAIL
  R plan             # must print PLAN SAME and an ESTIMATE line (23 patches, ~2 h)
  R prep; rm -rf $XSCAN_CACHE                    # all patches; rerun the same line if cut off (resumes)
  R train 0; R eval 0                            # then folds 1 and 2; a train cut off at 2 h resumes
  R train 1; R eval 1
  R train 2; R eval 2
  R verdict          # RULE and VERDICT lines; summary.json
  R status
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v020'))
import v020_scan_adapt as v20                                      # noqa: E402  (choose(): V-020's patch rule)

B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
SEG = 'PHerc0814/segments/20260226000000-46527_2um_try2/'
VOL_A = B + 'PHerc0814/volumes/20260309142202-2.399um-0.2m-78keV-masked.zarr'    # labelled scan, mesh A
VOL_B = B + 'PHerc0814/volumes/20260521123630-1.129um-0.2m-59keV-masked.zarr'    # 1.129 um scan, mesh B
URLS = {
    'transform': VOL_B + '/transform.json',
    'mesh_A': B + SEG + 'mesh/20260226000000-on-20260309142202-2.399um.tifxyz/',
    'mesh_B': B + SEG + 'mesh/20260226000000-on-20260521123630-1.129um.tifxyz/',
    'labels': B + SEG + 'ink-labels/2.399um-volume-20260309142202/20260918/inklabels.zarr/0/',
    'sv_B': B + SEG + 'surface-volumes/1.129um-0.22m-59keV-volume-20260521123630-L1.zarr/',
    'mrg': B + SEG + 'ink-detection/PHerc0814-20260226000000-1.129um-0.22m-59keV-volume-20260521123630-L1-'
                     '20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif',
}
VILLA_SHA = '7a5ba5a01f1ff7070918f2143143c11c6f7268c5'
UM_A, UM_B0 = 2.399, 1.129
UM_PX = 2 * UM_B0                       # render px and layer step on mesh B at level 1 (2.258 um)
CELL_A, CELL_B = 20, 10                 # px per mesh cell: A canvas (labels), B at level 1 (the L1 canvas)
SCALE = UM_B0 / UM_A                    # A cells per B cell (both meshes have 20 L0 voxels per cell)
LABEL_SHAPE, CANVAS_B = (8520, 13820), (9060, 14700)
# step 0 (README "Step 0")
CROPS, CROP_A, CROP_B, P_LAYERS, SLAB, MARGIN, NCC_MIN, NEED = 6, (8, 16), (16, 32), 61, 21, 12, 0.4, 5
SMOOTH = 2.0                            # smoothed normals, sigma in B cells (~ V-016's 1 A cell)
# step 1 (README "Step 1")
DELTA_MAX, RES_MAX, RES_MED_MAX, INK_FRAC_MIN, SCALE_TOL = 1.0, 8.0, 4.0, 0.5, 0.10
# step 2 (README "Step 2")
PATCH = (40, 80)                        # B cells: 400 x 800 px, 0.90 x 1.81 mm (README Amendment 2)
MAX_TEST, MAX_TRAIN = 6, 12
LAYERS = 63
GUARD_VOX = int(round(1000 / UM_B0))    # 1 mm at 1.129 um level 0 = 886
HALF_LAYERS = 64                        # +-31 L1 layers = +-62 L0 voxels, +2 for rounding
NEAR_PX = 2000 / UM_PX                  # 2 mm negative band at 2.258 um
DS = 8
MEAN_MIN, FRAC = 0.75, 5 / 6            # V-016 pass thresholds
BUDGET_S = 4 * 3600
EST = {'render_min': 1.5, 'train_min': 25, 'eval_min_per_window': 1.0}
DATA = os.environ.get('V030_DATA', 'out/v030')
RESULTS = os.environ.get('V030_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v030-pherc0814-cross-scan'))
RUN = os.environ.get('PUZZLE_RUN_DIR', '.')

v20.PATCH_H, v20.PATCH_W = PATCH
v20.MAX_PER, v20.REL, v20.MIN_DENS, v20.MIN_VALID = MAX_TEST, 0.2, 0.02, 0.5


# --------------------------------------------------------------------------------------------------------- basics
def log(*a):
    print(*a, flush=True)


def get(url, path, tries=5):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    for t in range(tries):
        try:
            urllib.request.urlretrieve(url, path + '.part')
            os.replace(path + '.part', path)
            return path
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if t == tries - 1:
                raise
            time.sleep(2 ** (t + 1))


def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def jsave(obj, *paths):
    for p in paths:
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        json.dump(obj, open(p + '.part', 'w'), indent=1)
        os.replace(p + '.part', p)


def src():
    s = os.environ.get('V030_SRC')
    if not s:
        raise SystemExit('set V030_SRC to the folder with render_crop.py (the V-016 src tree)')
    if s not in sys.path:
        sys.path.insert(0, s)
    return s


def timing(stage, seconds):
    tp = os.path.join(DATA, 'timing.json')
    t = jload(tp, {})
    t[stage] = round(t.get(stage, 0) + seconds, 1)
    jsave(t, tp)


def mesh(which):
    """(H, W, 3) float64 xyz in that scan's level-0 voxels, valid mask, local folder."""
    import tifffile
    md = os.path.join(DATA, 'mesh_' + which)
    for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
        get(URLS['mesh_' + which] + f, os.path.join(md, f))
    xyz = np.stack([tifffile.imread(os.path.join(md, c + '.tif')).astype(np.float64) for c in 'xyz'], -1)
    return xyz, (xyz > -0.5).all(-1), md


def write_mesh(xyz, valid, out, scale):
    import tifffile
    os.makedirs(out, exist_ok=True)
    for i, c in enumerate('xyz'):
        tifffile.imwrite(os.path.join(out, c + '.tif'), np.where(valid, xyz[..., i], -1).astype(np.float32))
    json.dump({'format': 'tifxyz', 'type': 'seg', 'scale': [scale, scale], 'uuid': os.path.basename(out)},
              open(os.path.join(out, 'meta.json'), 'w'))
    return out


def apply(M, p):
    return p @ M[:, :3].T + M[:, 3]


def inverse(M):
    A = np.linalg.inv(np.r_[M, [[0, 0, 0, 1]]])
    return A[:3]


def transforms():
    """M_pub (1.129 -> 2.399 voxels, xyz) and the least-squares refit to its landmarks (V-014), with residuals."""
    d = json.loads(urllib.request.urlopen(URLS['transform'], timeout=60).read())
    Mpub = np.array(d['transformation_matrix'], float)
    mv, fx = np.array(d['moving_landmarks'], float), np.array(d['fixed_landmarks'], float)
    H = np.c_[mv, np.ones(len(mv))]
    Mlsq = np.linalg.lstsq(H, fx, rcond=None)[0].T
    rms = lambda M: float(np.sqrt((np.linalg.norm(H @ M.T - fx, axis=1) ** 2).mean()))           # noqa: E731
    loo = []
    for i in range(len(mv)):
        k = np.arange(len(mv)) != i
        Mi = np.linalg.lstsq(H[k], fx[k], rcond=None)[0].T
        loo.append(float(np.linalg.norm(H[i] @ Mi.T - fx[i])))
    info = {'n_landmarks': len(mv), 'fixed_volume': d.get('fixed_volume'),
            'rms_px_2399': {'pub': rms(Mpub), 'lsq': rms(Mlsq)},
            'residual_px_2399_pub': [float(x) for x in np.linalg.norm(H @ Mpub.T - fx, axis=1)],
            'loo_px_2399_lsq': loo,
            'singular_values': {'pub': np.linalg.svd(Mpub[:, :3])[1].tolist(), 'lsq': np.linalg.svd(Mlsq[:, :3])[1].tolist(),
                                'physical': UM_B0 / UM_A},
            'M_pub': Mpub.tolist(), 'M_lsq': Mlsq.tolist()}
    return Mpub, Mlsq, info


def labels():
    """Level-0 human labels on the mesh-A canvas (bool), from a local mirror (zarr v3, one shard)."""
    import zarr
    lz = os.path.join(DATA, 'inklabels0')
    get(URLS['labels'] + 'zarr.json', os.path.join(lz, 'zarr.json'))
    meta = json.load(open(os.path.join(lz, 'zarr.json')))
    grid = [-(-n // c) for n, c in zip(meta['shape'], meta['chunk_grid']['configuration']['chunk_shape'])]
    for i in range(grid[0]):
        for j in range(grid[1]):
            try:
                get(f"{URLS['labels']}c/{i}/{j}", os.path.join(lz, 'c', str(i), str(j)))
            except urllib.error.HTTPError as e:
                if e.code not in (403, 404):
                    raise
    lab = zarr.open_array(lz, mode='r')[:] > 0
    assert lab.shape == LABEL_SHAPE, lab.shape
    return lab


def b_coverage(shape):
    """B vertices where the 1.129 um scan has data: the organisers' L1 surface volume of this segment at level 5
    (32 L1 px per pixel, ~72 um), any layer non-zero. The masked 1.129 um volume has no chunks outside the scroll
    mask, so mesh B is partly off the data even inside the volume bounds (README Amendment 1)."""
    cp = os.path.join(DATA, 'b_coverage.npy')
    if os.path.exists(cp):
        return np.load(cp)
    url = URLS['sv_B'] + '5/'
    meta = json.loads(urllib.request.urlopen(url + '.zarray', timeout=60).read())
    assert meta['compressor'] is None and meta['dtype'] == '|u1' and meta['chunks'][0] == meta['shape'][0], meta
    nz, cy, cx = meta['chunks']
    mip = np.zeros(meta['shape'][1:], bool)
    for iy in range(-(-meta['shape'][1] // cy)):
        for ix in range(-(-meta['shape'][2] // cx)):
            try:
                raw = urllib.request.urlopen(f'{url}0/{iy}/{ix}', timeout=120).read()
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    continue
                raise
            c = (np.frombuffer(raw, np.uint8).reshape(nz, cy, cx) > 0).any(0)
            h, w = mip[iy * cy:(iy + 1) * cy, ix * cx:(ix + 1) * cx].shape
            mip[iy * cy:iy * cy + h, ix * cx:ix * cx + w] = c[:h, :w]
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    cov = mip[np.clip(ii * CELL_B // 32, 0, mip.shape[0] - 1), np.clip(jj * CELL_B // 32, 0, mip.shape[1] - 1)]
    np.save(cp, cov)
    return cov


def vol_shape(url, level=0):
    return json.loads(urllib.request.urlopen(f'{url}/{level}/.zarray', timeout=60).read())['shape']   # z, y, x


def render(mdir, vol, out, rows, cols, layers, level, cache=None):
    s = src()
    env = dict(os.environ)
    if cache:
        env['XSCAN_CACHE'] = cache
    subprocess.check_call([sys.executable, os.path.join(s, 'render_crop.py'), mdir, vol, out, '--rows', str(rows[0]),
                           str(rows[1]), '--cols', str(cols[0]), str(cols[1]), '--layers', str(layers),
                           '--smooth-normals', str(SMOOTH), '--level', str(level), '--oversample', '1',
                           '--cache-chunks', '1500'], stdout=subprocess.DEVNULL, env=env)
    return np.load(out + '.npy')


# --------------------------------------------------------------------------------------------------------- step 0
def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def placement_stat(R, X):
    """V-014 addendum statistic: NCC of R's central SLAB layers vs X at zero shift, best over +-20 layers, +-12 px."""
    R, X = R.astype(np.float32), X.astype(np.float32)
    L0, m = (R.shape[0] - SLAB) // 2, MARGIN
    H, W = R.shape[1:]
    ref = R[L0:L0 + SLAB, m:H - m, m:W - m]
    z = ncc(ref, X[L0:L0 + SLAB, m:H - m, m:W - m])
    best = (-2.0, 0, 0, 0)
    for dz in range(-20, 21):
        for dy in range(-m, m + 1, 2):
            for dx in range(-m, m + 1, 2):
                s = ncc(ref, X[L0 + dz:L0 + dz + SLAB, m + dy:H - m + dy, m + dx:W - m + dx])
                if s > best[0]:
                    best = (s, dz, dy, dx)
    return {'ncc0': round(z, 4), 'best_ncc': round(best[0], 4), 'best_shift_layers_y_x': list(best[1:])}


def greedy_boxes(cells, ok, shape, n):
    """Densest non-overlapping boxes of `shape` cells whose every cell is ok."""
    h, w = shape
    dens = ndimage.uniform_filter(np.where(ok, cells, 0).astype(np.float32), shape, mode='constant')
    taken = np.zeros_like(ok)
    out = []
    while len(out) < n:
        blk = ndimage.maximum_filter((~ok | taken).astype(np.uint8), shape, mode='constant', cval=1) > 0
        dm = np.where(blk, -1, dens)
        cy, cx = np.unravel_index(np.argmax(dm), dm.shape)
        if dm[cy, cx] <= 0:
            break
        r0, c0 = int(cy - h // 2), int(cx - w // 2)
        out.append(([r0, r0 + h, c0, c0 + w], float(dm[cy, cx])))
        taken[r0:r0 + h, c0:c0 + w] = True
    return out


def cell_ok(valid):
    """Cell (i, j) ok when its 4 vertices are valid. Shape (H - 1, W - 1)."""
    return valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]


def step0(force=False):
    out_p = os.path.join(RESULTS, 'step0.json')
    if os.path.exists(out_p) and not force:
        r = json.load(open(out_p))
        log('step0 already recorded in', out_p, '(rerun with --force to recompute)')
        for c in r['crops']:
            log('CROP', json.dumps(c))
        log(r['line'])
        return r
    t0 = time.time()
    Mpub, Mlsq, tinfo = transforms()
    log('TRANSFORM landmark RMS px@2.399: pub', round(tinfo['rms_px_2399']['pub'], 2), 'lsq',
        round(tinfo['rms_px_2399']['lsq'], 2), 'LOO max', round(max(tinfo['loo_px_2399_lsq']), 2))
    A, vA, _ = mesh('A')
    Bm, vB, mdB = mesh('B')
    lab = labels()
    Hc, Wc = min(lab.shape[0] // CELL_A, A.shape[0] - 1), min(lab.shape[1] // CELL_A, A.shape[1] - 1)
    cells = lab[:Hc * CELL_A, :Wc * CELL_A].reshape(Hc, CELL_A, Wc, CELL_A).mean((1, 3), dtype=np.float32)
    del lab
    from scipy.spatial import cKDTree
    idxB = np.argwhere(vB)
    treeB = cKDTree(apply(Mpub, Bm[vB]))
    # as V-014: only A cells whose vertices map inside the 1.129 um volume, and onto B vertices with 1.129 um data
    zs, ys, xs = vol_shape(VOL_B)
    Ab = apply(inverse(Mpub), A)
    inB = vA & (Ab >= 0).all(-1) & (Ab <= np.array([xs - 1, ys - 1, zs - 1])).all(-1)
    cov = b_coverage(vB.shape)
    _, nb = treeB.query(A[inB], workers=-1)
    inB[inB] = cov[idxB[nb, 0], idxB[nb, 1]]
    boxes = greedy_boxes(cells, cell_ok(inB)[:Hc, :Wc], CROP_A, CROPS)
    mesh_ref = write_mesh(np.where(vB[..., None], apply(Mpub, Bm), -1), vB, os.path.join(DATA, 'mesh_AonB'), 0.1)
    Bfit = apply(inverse(Mlsq), apply(Mpub, Bm))
    mesh_fit = write_mesh(np.where(vB[..., None], Bfit, -1), vB, os.path.join(DATA, 'mesh_Bfit'), 0.05)
    crops = []
    for k, (rc, dn) in enumerate(boxes):
        r0, r1, c0, c1 = rc
        ca = A[(r0 + r1) // 2, (c0 + c1) // 2]
        d, j = treeB.query(ca)
        bi, bj = (int(x) for x in idxB[j])
        hb, wb = CROP_B
        br, bc = (bi - hb // 2, bi + hb // 2), (bj - wb // 2, bj + wb // 2)
        c = {'crop': k, 'cells_A': rc, 'density': round(dn, 4), 'centre_B': [bi, bj], 'nn_dist_vox_2399': round(float(d), 2),
             'rows_B': list(br), 'cols_B': list(bc)}
        inb = br[0] >= 0 and bc[0] >= 0 and br[1] < vB.shape[0] and bc[1] < vB.shape[1]
        if not inb or vB[br[0]:br[1] + 1, bc[0]:bc[1] + 1].mean() < 0.9:
            c['status'] = 'UNKNOWN: B crop under 90% valid vertices'
            crops.append(c); log('CROP', json.dumps(c)); continue
        cache = os.path.join(DATA, 'cache_step0') if os.environ.get('V030_LOWDISK') else None
        base = os.path.join(DATA, 'step0', f'crop{k}_')
        os.makedirs(os.path.dirname(base), exist_ok=True)
        R = render(mesh_ref, VOL_A, base + 'A', br, bc, P_LAYERS, 0, cache)
        X = render(mdB, VOL_B, base + 'B', br, bc, P_LAYERS, 1, cache)
        c['valid_frac'] = {'A': round(float((R > 0).any(0).mean()), 3), 'B': round(float((X > 0).any(0).mean()), 3)}
        if min(c['valid_frac'].values()) < 0.5:
            c['status'] = 'UNKNOWN: under 50% CT-valid pixels'
        else:
            c['pub'] = placement_stat(R, X)
            c['status'] = 'ok'
        if cache:
            shutil.rmtree(cache, ignore_errors=True)
        crops.append(c)
        log('CROP', json.dumps(c))
    passed = lambda arm: sum(1 for c in crops if arm in c and c[arm]['ncc0'] >= NCC_MIN)      # noqa: E731
    n_pub = passed('pub')
    verdict = 'PLACEMENT_OK' if n_pub >= NEED else None
    if verdict is None:                      # refit as in V-014: B' = M_lsq^-1 M_pub B, same grid and canvas
        for k, c in enumerate(crops):
            if c['status'] != 'ok':
                continue
            cache = os.path.join(DATA, 'cache_step0') if os.environ.get('V030_LOWDISK') else None
            base = os.path.join(DATA, 'step0', f'crop{k}_')
            R = np.load(base + 'A.npy')
            X = render(mesh_fit, VOL_B, base + 'Bfit', c['rows_B'], c['cols_B'], P_LAYERS, 1, cache)
            c['refit'] = placement_stat(R, X)
            if cache:
                shutil.rmtree(cache, ignore_errors=True)
            log('CROP refit', json.dumps(c))
        n_fit = passed('refit')
        better = sum(1 for c in crops if 'refit' in c and c['refit']['ncc0'] > c['pub']['ncc0'])
        verdict = 'PLACEMENT_REFIT' if n_fit >= NEED and better >= NEED else 'PLACEMENT_FAIL'
    else:
        n_fit = better = None
    line = (f'{verdict}: published transform NCC0 >= {NCC_MIN} on {n_pub} of {len(crops)} crops (need {NEED})'
            + ('' if n_fit is None else f'; refit (M_lsq) NCC0 >= {NCC_MIN} on {n_fit}, refit > published on {better}'))
    res = {'verdict': verdict, 'line': line, 'mesh_for_step2': {'PLACEMENT_OK': 'B', 'PLACEMENT_REFIT': 'Bfit'}.get(verdict),
           'transform': tinfo, 'crops': crops,
           'rule': dict(CROPS=CROPS, CROP_A=CROP_A, CROP_B=CROP_B, LAYERS=P_LAYERS, SLAB=SLAB, MARGIN=MARGIN,
                        NCC_MIN=NCC_MIN, NEED=NEED, SMOOTH=SMOOTH),
           'v018_audit_row': {'pair': '1.129um L1 vs 2.399um surface volumes', 'usable_windows': 1, 'ncc0': 0.543,
                              'best': 0.911, 'shift_um': [9.6, 24.0, 28.8]},
           'seconds': round(time.time() - t0, 1)}
    jsave(res, out_p, os.path.join(RUN, 'payload', 'v030-step0.json'))
    log(line)
    if verdict == 'PLACEMENT_FAIL':
        log('VERDICT STOPPED_AT_PLACEMENT')
    return res


def gate(stage):
    s0 = jload(os.path.join(RESULTS, 'step0.json'))
    if s0 is None:
        raise SystemExit(f'{stage}: run step0 first')
    if s0['verdict'] == 'PLACEMENT_FAIL':
        raise SystemExit(f'{stage}: step 0 is PLACEMENT_FAIL; V-030 stops there (VERDICT STOPPED_AT_PLACEMENT)')
    if stage != 'carry':
        c = jload(os.path.join(RESULTS, 'carry.json'))
        if c is None or not os.path.exists(os.path.join(DATA, 'carry', 'lab_c.npy')):
            raise SystemExit(f'{stage}: run carry first')
        if c['verdict'] != 'CARRY_OK':
            raise SystemExit(f'{stage}: step 1 is CARRY_FAIL; V-030 stops there (VERDICT STOPPED_AT_CARRY)')
    return s0


# --------------------------------------------------------------------------------------------------------- step 1
def jacobians(X, V):
    """Per-vertex d(xyz)/d(row) and d(xyz)/d(col): central differences, one-sided at edges and holes, NaN if none."""
    out = []
    for ax in (0, 1):
        fwd = np.full(X.shape, np.nan); bwd = np.full(X.shape, np.nan)
        sl = [slice(None)] * 2
        a, b = list(sl), list(sl)
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        ok = V[a] & V[b]
        d = np.where(ok[..., None], X[a] - X[b], np.nan)
        fwd[b] = d                     # X[i+1] - X[i] stored at i
        bwd[a] = d                     # X[i] - X[i-1] stored at i
        J = np.where(np.isfinite(fwd) & np.isfinite(bwd), (fwd + bwd) / 2, np.where(np.isfinite(fwd), fwd, bwd))
        out.append(np.where(V[..., None], J, np.nan))
    return out


def carry_vertices(A, vA, q):
    """Sub-cell A grid coords (u, v), the clamp-free delta and the normal residual for 3D points q (2.399 voxels)."""
    from scipy.spatial import cKDTree
    idxA = np.argwhere(vA)
    _, k = cKDTree(A[vA]).query(q, workers=-1)
    i, j = idxA[k].T
    Ju, Jv = jacobians(A, vA)
    J = np.stack([Ju[i, j], Jv[i, j]], -1)                    # (n, 3, 2)
    r = q - A[i, j]
    JtJ = np.einsum('nki,nkj->nij', J, J)
    Jtr = np.einsum('nki,nk->ni', J, r)
    good = np.isfinite(JtJ).all((1, 2)) & (np.abs(np.linalg.det(np.nan_to_num(JtJ))) > 1e-9)
    delta = np.full((len(q), 2), np.nan)
    delta[good] = np.linalg.solve(JtJ[good], Jtr[good][..., None])[..., 0]
    res = np.full(len(q), np.nan)
    res[good] = np.linalg.norm(r[good] - np.einsum('nki,ni->nk', J[good], delta[good]), axis=1)
    return i + delta[:, 0], j + delta[:, 1], delta, res


def carry_planes(A, vA, Bm, vB, M):
    q = apply(M, Bm[vB])
    u, v, delta, res = carry_vertices(A, vA, q)
    ok = np.isfinite(res) & (np.abs(delta) <= DELTA_MAX).all(1) & (res <= RES_MAX)
    ok &= (u >= 0) & (v >= 0) & (u <= A.shape[0] - 1) & (v <= A.shape[1] - 1)
    U = np.full(vB.shape, -1.0, np.float32); Vv = np.full(vB.shape, -1.0, np.float32)
    C = np.zeros(vB.shape, bool)
    U[vB], Vv[vB], C[vB] = np.where(ok, u, -1), np.where(ok, v, -1), ok
    return U, Vv, C, res[ok]


def local_scale(U, V, C):
    """Singular values of d(u, v)/d(B row, col) where a vertex and its +row/+col neighbours are carried."""
    m = C[:-1, :-1] & C[1:, :-1] & C[:-1, 1:]
    J = np.stack([np.stack([U[1:, :-1] - U[:-1, :-1], U[:-1, 1:] - U[:-1, :-1]], -1),
                  np.stack([V[1:, :-1] - V[:-1, :-1], V[:-1, 1:] - V[:-1, :-1]], -1)], -2)[m]   # (n, 2, 2)
    return np.linalg.svd(J, compute_uv=False)


def carry_canvas(U, V, C, lab, strip=64):
    """Per-pixel carried labels and carried mask on the B L1 canvas (10 px per cell, render_tifxyz.upsample)."""
    src()
    from render_tifxyz import upsample
    H, W = C.shape
    lab_c = np.zeros((H * CELL_B, W * CELL_B), bool)
    car = np.zeros_like(lab_c)
    for a in range(0, H, strip):
        b = min(H, a + strip)
        g = slice(a, min(H, b + 1))
        Up, ok = upsample(U[g], C[g], CELL_B, CELL_B)
        Vp, _ = upsample(V[g], C[g], CELL_B, CELL_B)
        n = (b - a) * CELL_B
        Up, Vp, ok = Up[:n], Vp[:n], ok[:n]
        li, lj = np.rint(Up * CELL_A).astype(np.int64), np.rint(Vp * CELL_A).astype(np.int64)
        ok &= (li >= 0) & (lj >= 0) & (li < lab.shape[0]) & (lj < lab.shape[1])
        rows = slice(a * CELL_B, a * CELL_B + n)
        car[rows] = ok
        lab_c[rows] = np.where(ok, lab[np.clip(li, 0, lab.shape[0] - 1), np.clip(lj, 0, lab.shape[1] - 1)], False)
    return lab_c, car


def down_bool(a, ds=DS, strip=4096):
    h, w = a.shape[0] // ds * ds, a.shape[1] // ds * ds
    out = np.zeros((h // ds, w // ds), np.float32)
    for y in range(0, h, strip):
        y1 = min(h, y + strip)
        out[y // ds:y1 // ds] = a[y:y1, :w].reshape((y1 - y) // ds, ds, w // ds, ds).mean((1, 3), dtype=np.float32)
    return out


def save_bits(path, a):
    np.save(path, np.packbits(a, axis=None))
    json.dump(list(a.shape), open(path + '.shape.json', 'w'))


def load_bits(path):
    shp = json.load(open(path + '.shape.json'))
    return np.unpackbits(np.load(path))[:shp[0] * shp[1]].reshape(shp).astype(bool)


def carry():
    s0 = gate('carry')
    t0 = time.time()
    Mpub, _, _ = transforms()
    A, vA, _ = mesh('A')
    Bm, vB, _ = mesh('B')
    lab = labels()
    U, V, C, res = carry_planes(A, vA, Bm, vB, Mpub)
    sv = local_scale(U, V, C)
    lab_c, car = carry_canvas(U, V, C, lab)
    assert lab_c.shape == CANVAS_B, lab_c.shape
    cd = os.path.join(DATA, 'carry'); os.makedirs(cd, exist_ok=True)
    save_bits(os.path.join(cd, 'lab_c.npy'), lab_c)
    save_bits(os.path.join(cd, 'carried.npy'), car)
    lab8 = down_bool(lab_c)
    np.save(os.path.join(cd, 'lab8.npy'), lab8)
    np.save(os.path.join(cd, 'U.npy'), U); np.save(os.path.join(cd, 'V.npy'), V)
    src_mm2 = float(lab.sum() * (UM_A * 1e-3) ** 2)
    car_mm2 = float(lab_c.sum() * (UM_PX * 1e-3) ** 2)
    med_sv = [float(np.median(sv[:, 0])), float(np.median(sv[:, 1]))]
    rules = {
        'median_residual_vox': (float(np.median(res)), float(np.median(res)) <= RES_MED_MAX),
        'carried_ink_frac': (car_mm2 / src_mm2 if src_mm2 else 0.0, src_mm2 > 0 and car_mm2 / src_mm2 >= INK_FRAC_MIN),
        'median_local_scale': (med_sv, all(abs(s / SCALE - 1) <= SCALE_TOL for s in med_sv)),
    }
    verdict = 'CARRY_OK' if all(ok for _, ok in rules.values()) else 'CARRY_FAIL'
    res_d = {
        'verdict': verdict, 'transform': 'M_pub', 'mesh_for_step2': s0['mesh_for_step2'],
        'rules': {k: {'value': v, 'ok': ok} for k, (v, ok) in rules.items()},
        'thresholds': dict(DELTA_MAX=DELTA_MAX, RES_MAX=RES_MAX, RES_MED_MAX=RES_MED_MAX, INK_FRAC_MIN=INK_FRAC_MIN,
                           SCALE=SCALE, SCALE_TOL=SCALE_TOL),
        'residual_vox_p90': float(np.percentile(res, 90)),
        'vertices': {'B_valid': int(vB.sum()), 'carried': int(C.sum()), 'frac': float(C.sum() / vB.sum())},
        'pixels': {'carried': int(car.sum()), 'carried_ink': int(lab_c.sum()), 'canvas': list(lab_c.shape)},
        'ink_mm2': {'source_2399': src_mm2, 'carried_1129': car_mm2},
        'seconds': round(time.time() - t0, 1)}
    res_d['line'] = (f"{verdict}: median normal residual {rules['median_residual_vox'][0]:.2f} vox (<= {RES_MED_MAX}), "
                     f"carried ink {car_mm2:.2f} of {src_mm2:.2f} mm2 ({rules['carried_ink_frac'][0]:.3f}, >= {INK_FRAC_MIN}), "
                     f"median local scale {med_sv[0]:.4f}/{med_sv[1]:.4f} (target {SCALE:.4f} +-{SCALE_TOL:.0%})")
    ref = jload(os.path.join(RESULTS, 'carry.json'))
    if ref is not None:
        same = ref['verdict'] == verdict and all(
            abs(ref[g][k] - res_d[g][k]) <= 1e-3 * max(1, abs(ref[g][k])) for g in ('vertices', 'pixels') for k in ('carried',)
        ) and abs(ref['pixels']['carried_ink'] - res_d['pixels']['carried_ink']) <= 1e-3 * max(1, ref['pixels']['carried_ink'])
        log('CARRY', 'SAME' if same else 'DIFFERENT', 'as', os.path.join(RESULTS, 'carry.json'))
        jsave(res_d, os.path.join(RUN, 'payload', 'v030-carry.json'))
    else:
        jsave(res_d, os.path.join(RESULTS, 'carry.json'), os.path.join(RUN, 'payload', 'v030-carry.json'))
        preview(lab8, down_bool(car), os.path.join(RESULTS, 'carry_ds8.png'))
    timing('carry', time.time() - t0)
    log(res_d['line'])
    if verdict == 'CARRY_FAIL':
        log('VERDICT STOPPED_AT_CARRY')
    return res_d


def preview(lab8, car8, path):
    from PIL import Image
    img = np.where(car8 > 0.5, 40 + 215 * (lab8 >= 0.5), 0).astype(np.uint8)
    Image.fromarray(img).save(path)


# --------------------------------------------------------------------------------------------------------- plan
def patch_dict(rc, dn, cells, inside, inv, Bm, px_mm2, **kw):
    r0, r1, c0, c1 = rc
    q = Bm[r0:r1 + 1, c0:c1 + 1][inv[r0:r1 + 1, c0:c1 + 1]]       # every in-volume vertex the render reads
    return {'name': f'r{r0}_c{c0}', 'cells': rc, 'density': round(dn, 4),
            'ink_mm2': round(float(cells[r0:r1, c0:c1][inside[r0:r1, c0:c1]].sum() * CELL_B ** 2 * px_mm2), 3),
            'valid_cells': round(float(inside[r0:r1, c0:c1].mean()), 3),
            'z_min': round(float(q[:, 2].min()), 1), 'z_max': round(float(q[:, 2].max()), 1),
            'label_origin': [r0 * CELL_B, c0 * CELL_B], 'lab_shape': [(r1 - r0) * CELL_B, (c1 - c0) * CELL_B], **kw}, q


def plan_compute():
    """Held-out windows: per z band (ink terciles), V-020's greedy rule. Training patches: per fold, V-016's rule
    (densest labelled patches anywhere that keep clear of that fold's held-out windows), with the clearance measured
    in 3D: every in-volume vertex >= 1 mm + both renders' reach (GUARD_VOX + 2 * HALF_LAYERS) from every in-volume
    vertex of the fold's test windows (README Amendment 2)."""
    gate('plan')
    from scipy.spatial import cKDTree
    Bm, vB, _ = mesh('B')
    zs, ys, xs = vol_shape(VOL_B)
    lab_c = load_bits(os.path.join(DATA, 'carry', 'lab_c.npy'))
    car = load_bits(os.path.join(DATA, 'carry', 'carried.npy'))
    H, W = vB.shape[0] - 1, vB.shape[1] - 1
    cells = lab_c[:H * CELL_B, :W * CELL_B].reshape(H, CELL_B, W, CELL_B).mean((1, 3), dtype=np.float32)
    full = car[:H * CELL_B, :W * CELL_B].reshape(H, CELL_B, W, CELL_B).all((1, 3))
    del lab_c, car
    inv = vB & (Bm >= 0).all(-1) & (Bm <= np.array([xs - 1, ys - 1, zs - 1])).all(-1) & b_coverage(vB.shape)
    z4 = np.stack([Bm[:-1, :-1, 2], Bm[1:, :-1, 2], Bm[:-1, 1:, 2], Bm[1:, 1:, 2]], -1)[:H, :W]
    zc = z4.min(-1)
    inside = cell_ok(inv) & full
    w = cells * inside
    order = np.argsort(zc[inside])
    cw = np.cumsum(w[inside][order]) / w[inside].sum()
    edges = tuple(int(round(float(zc[inside][order][np.searchsorted(cw, q)]) / 10) * 10) for q in (1 / 3, 2 / 3))
    px_mm2 = (UM_PX * 1e-3) ** 2
    # held-out windows
    v20.BAND_EDGES, v20.MAX_PER = edges, MAX_TEST
    in_band = inside & (v20.band_of(z4.min(-1)) == v20.band_of(z4.max(-1)))   # a test cell lies in one band
    test, tq = {}, {}
    for b in range(3):
        test[b], tq[b] = [], []
        for rc, dn in v20.choose(cells, in_band, zc, b):
            p, q = patch_dict(rc, dn, cells, inside, inv, Bm, px_mm2, band=b)
            p['straddles_band_edge'] = v20.band_of(np.array([p['z_min'], p['z_max']])).tolist() != [b, b]
            test[b].append(p); tq[b].append(q)
    # training patches per fold
    need = GUARD_VOX + 2 * HALF_LAYERS
    train, guard = {}, {}
    v20.BAND_EDGES, v20.MAX_PER = (0.5,), MAX_TRAIN
    for k in range(3):
        tree = cKDTree(np.concatenate(tq[k]))
        far = np.zeros(inv.shape, bool)
        far[inv] = tree.query(Bm[inv], workers=-1)[0] >= need
        ok = cell_ok(far)[:H, :W] & inside
        lab = np.where(ok, 0.0, 1.0)                                   # band 0 = clear of fold k's windows
        train[k], mins = [], []
        for rc, dn in v20.choose(cells, inside, lab, 0):
            p, q = patch_dict(rc, dn, cells, inside, inv, Bm, px_mm2)
            d = float(tree.query(q, workers=-1)[0].min())
            if d < need:                                               # a vertex of a non-valid cell got too close
                continue
            p['min_dist_vox_to_test'] = round(d, 1)
            train[k].append(p); mins.append(d)
        guard[k] = {'need_vox': need, 'min_vox': round(min(mins), 1) if mins else None}
    v20.BAND_EDGES, v20.MAX_PER = edges, MAX_TEST
    band_ink = {b: round(float(w[inside & (v20.band_of(zc) == b)].sum() * CELL_B ** 2 * px_mm2), 3) for b in range(3)}
    return {'rule': dict(PATCH=PATCH, MAX_TEST=MAX_TEST, MAX_TRAIN=MAX_TRAIN, REL=v20.REL, MIN_DENS=v20.MIN_DENS,
                         MIN_VALID=v20.MIN_VALID, BAND_EDGES=edges, GUARD_VOX=GUARD_VOX, HALF_LAYERS=HALF_LAYERS,
                         LAYERS=LAYERS, LEVEL=1, UM_PX=UM_PX, SMOOTH=SMOOTH, NEAR_PX=NEAR_PX),
            'band_ink_mm2': band_ink, 'guard': guard,
            'folds': {str(k): {'test': [p['name'] for p in test[k]], 'train': [p['name'] for p in train[k]]} for k in range(3)},
            'patches': unique([p for k in range(3) for p in test[k] + train[k]])}


def unique(ps):
    out = {}
    for p in ps:
        if p['name'] in out:
            if 'band' in p:
                out[p['name']].update({x: p[x] for x in ('band', 'straddles_band_edge')})
        else:
            out[p['name']] = dict(p)
    return sorted(out.values(), key=lambda p: p['name'])


def folds(pl):
    return {int(k): v for k, v in pl['folds'].items()}


def estimate(pl):
    n = len(pl['patches'])
    nt = sum(len(f['test']) for f in pl['folds'].values())
    r, t, e = n * EST['render_min'], 3 * EST['train_min'], nt * EST['eval_min_per_window']
    return n, r, t, e


def plan(write=True):
    pj = os.path.join(DATA, 'windows.json')
    if os.path.exists(pj):
        pl = json.load(open(pj))
    else:
        t0 = time.time()
        pl = json.loads(json.dumps(plan_compute()))      # JSON types, so the SAME check compares like with like
        reg = os.path.join(RESULTS, 'windows.json')
        if os.path.exists(reg):
            log('PLAN', 'SAME' if json.load(open(reg)) == pl else 'DIFFERENT', 'as', reg)
        elif write:
            jsave(pl, reg)
        if write:
            jsave(pl, pj)
        timing('plan', time.time() - t0)
    P = {p['name']: p for p in pl['patches']}
    for k, f in folds(pl).items():
        log(f'FOLD {k} band {k} test {len(f["test"])} ({sum(P[n]["ink_mm2"] for n in f["test"]):.2f} mm2 ink) '
            f'train {len(f["train"])} ({sum(P[n]["ink_mm2"] for n in f["train"]):.2f} mm2 ink)')
    n, r, t, e = estimate(pl)
    log(f'ESTIMATE {n} patches to render: renders ~{r:.0f} min, training ~{t:.0f} min (3 folds), eval ~{e:.0f} min; '
        f'total ~{(r + t + e) / 60:.1f} h (budget 4 h)')
    return pl


# --------------------------------------------------------------------------------------------------------- prep
def step2_mesh():
    s0 = gate('prep')
    which = s0['mesh_for_step2']
    if which == 'B':
        return mesh('B')[2]
    if which == 'Bfit':
        Mpub, Mlsq, _ = transforms()
        Bm, vB, _ = mesh('B')
        out = os.path.join(DATA, 'mesh_Bfit')
        if not os.path.exists(os.path.join(out, 'z.tif')):
            write_mesh(np.where(vB[..., None], apply(inverse(Mlsq), apply(Mpub, Bm)), -1), vB, out, 0.05)
        return out
    raise SystemExit(f'no mesh for step 2 (step 0 verdict {s0["verdict"]})')


def patch_labels(p, lab_c=None, car=None):
    pd = os.path.join(DATA, 'patches')
    lp, mp = os.path.join(pd, p['name'] + '_lab.png'), os.path.join(pd, p['name'] + '_carried.npy')
    if not os.path.exists(lp):
        from PIL import Image
        os.makedirs(pd, exist_ok=True)
        y, x = p['label_origin']
        h, w = p['lab_shape']
        Image.fromarray(lab_c[y:y + h, x:x + w].astype(np.uint8) * 255).save(lp)
        save_bits(mp, car[y:y + h, x:x + w])
    from PIL import Image
    return np.asarray(Image.open(lp)) >= 128, load_bits(mp)


def prep(names):
    pl = plan(write=False)
    md = step2_mesh()
    lab_c = load_bits(os.path.join(DATA, 'carry', 'lab_c.npy'))
    car = load_bits(os.path.join(DATA, 'carry', 'carried.npy'))
    todo = [p for p in pl['patches'] if not names or p['name'] in names]
    for i, p in enumerate(todo):
        t0 = time.time()
        lab, cm = patch_labels(p, lab_c, car)
        out = os.path.join(DATA, 'patches', p['name'])
        info = jload(out + '.v030.json')
        if info is None:
            r0, r1, c0, c1 = p['cells']
            ct = render(md, VOL_B, out, (r0, r1), (c0, c1), LAYERS, 1)
            assert ct.shape[1:] == tuple(p['lab_shape']), (ct.shape, p['lab_shape'])
            unc = (ct != 0).any(0) & ~cm
            ct[:, unc] = 0                                # uncarried CT is neither trained on nor scored
            np.save(out + '.npy', ct)
            info = {'uncarried_ct_px': int(unc.sum()), 'ct_valid_px': int((ct != 0).any(0).sum()),
                    'seconds': round(time.time() - t0, 1)}
            jsave(info, out + '.v030.json')
            timing('prep', time.time() - t0)
        log(f'[{i + 1}/{len(todo)}]', p['name'], 'cells', p['cells'], 'ink', p['ink_mm2'], 'mm2', json.dumps(info))


# --------------------------------------------------------------------------------------------------------- train
def villa():
    v = os.environ.get('V030_VILLA')
    if not v:
        raise SystemExit('set V030_VILLA to the neg-0/villa checkout at 7a5ba5a')
    return v


def cdir(k):
    return os.path.join(DATA, f'ckpt_fold{k}')


def train(k):
    pl = plan(write=False)
    gate('train')
    names = folds(pl)[k]['train']
    assert names, f'fold {k}: no training patches'
    out = cdir(k)
    if os.path.exists(os.path.join(out, 'final.pt')):
        log('fold', k, 'already trained')
    else:
        t0 = time.time()
        port = os.path.join(villa(), 'ink-detection', 'scan_adapt', 'finetune_on_scan.py')
        pairs = []
        for n in names:
            stack = os.path.join(DATA, 'patches', n + '.npy')
            assert os.path.exists(stack), f'{n} is not rendered (run prep)'
            pairs += ['--pair', stack, os.path.join(DATA, 'patches', n + '_lab.png')]
        cmd = [sys.executable, port, '--checkpoint', os.environ['CANON_CKPT'], '--out-dir', out, '--amp', 'bf16',
               '--voxel-um', str(UM_PX), '--device', os.environ.get('INK_DEVICE', 'cuda')] + pairs
        log('train fold', k, 'patches', len(names), ' '.join(cmd[:12]), '...')
        subprocess.check_call(cmd)
        timing(f'train{k}', time.time() - t0)
    tl = json.load(open(os.path.join(out, 'train_log.json')))
    jsave({**tl, 'fold': k, 'patches': names}, os.path.join(RESULTS, f'train_log_fold{k}.json'))
    log('TRAINED fold', k, 'iters', tl.get('iters'), 'skipped_frac', tl.get('skipped_frac'))


# --------------------------------------------------------------------------------------------------------- eval
def infer_mod():
    src()
    os.environ['V016_AMP'] = 'bf16'                       # V-016b/V-019: bf16 inference
    os.environ.setdefault('VILLA_INFERENCE_DIR', os.path.join(villa(), 'ink-detection', 'optimized_inference'))
    for sub in ('v014', 'v016'):
        sys.path.insert(0, os.path.join(HERE, '..', sub))
    import v016_cross_scan_ink as d
    assert d.AMP == 'bf16'
    return d


def pixel_sets(lab, cm, ct):
    """ds8 positives / negatives: label >= 0.5, or < 0.05 within 2 mm of ink; CT-valid and carried only."""
    src()
    import march_bench as mb
    L = mb.down(lab.astype(np.float32), DS)
    ok = mb.down(((np.asarray(ct) != 0).any(0) & cm).astype(np.float32), DS) == 1.0
    near = ndimage.distance_transform_edt(L < 0.5) <= NEAR_PX / DS
    return (L >= 0.5) & ok, near & (L < 0.05) & ok


def auc_on(m, sets):
    src()
    import march_bench as mb
    pos, neg = sets
    t = mb.down(m, DS)
    return mb.auc(t, pos, neg)


def mrg_crop(p, pl):
    """The organisers' mrg20736-1um map on the window (same L1 canvas), 0-1; the full map is read once."""
    out = os.path.join(DATA, 'mrg', p['name'] + '.npy')
    if os.path.exists(out):
        return np.load(out)
    import tifffile
    tif = get(URLS['mrg'], os.path.join(DATA, 'mrg', 'mrg20736-1um.tif'))
    a = tifffile.imread(tif)
    assert a.shape == CANVAS_B and a.dtype == np.uint8, (a.shape, a.dtype)
    for q in pl['patches']:
        y, x = q['label_origin']
        h, w = q['lab_shape']
        np.save(os.path.join(DATA, 'mrg', q['name'] + '.npy'), a[y:y + h, x:x + w].astype(np.float32) / 255.0)
    return np.load(out)


def evaluate(k):
    pl = plan(write=False)
    gate('eval')
    d = infer_mod()
    import march_bench as mb
    P = {p['name']: p for p in pl['patches']}
    names = folds(pl)[k]['test']
    ft_ckpt = os.path.join(cdir(k), 'final.pt')
    assert os.path.exists(ft_ckpt), f'fold {k} is not trained'
    lab8 = np.load(os.path.join(DATA, 'carry', 'lab8.npy'))
    t0 = time.time()
    per, maps, sets_all = {}, {'zero_shot': [], 'finetuned': [], 'mrg': []}, []
    for n in names:
        p = P[n]
        stack = np.load(os.path.join(DATA, 'patches', n + '.npy'), mmap_mode='r')[:62]
        lab, cm = patch_labels(p)
        s = pixel_sets(lab, cm, stack)
        z = d.infer(os.environ['CANON_CKPT'], stack, os.path.join(DATA, 'patches', f'{n}_ink_zero.npy'))
        f = d.infer(ft_ckpt, stack, os.path.join(cdir(k), f'ink_{n}.npy'))
        g = mrg_crop(p, pl)
        e0 = (p['label_origin'][0] // DS, p['label_origin'][1] // DS)
        r = {'zero_shot': auc_on(z, s), 'finetuned': auc_on(f, s), 'mrg20736_1um': auc_on(g, s),
             'pos_ds8': int(s[0].sum()), 'neg_ds8': int(s[1].sum()),
             'searched': {kk: mb.eval_map(m, lab8, e0) for kk, m in (('zero_shot', z), ('finetuned', f), ('mrg20736_1um', g))}}
        per[n] = r
        maps['zero_shot'].append(z); maps['finetuned'].append(f); maps['mrg'].append(g); sets_all.append(s)
        log('WINDOW', n, f"zero-shot {r['zero_shot']:.4f} finetuned {r['finetuned']:.4f} mrg20736 {r['mrg20736_1um']:.4f}",
            f"pos {r['pos_ds8']} neg {r['neg_ds8']}")

    def pooled(ms):
        sp = np.concatenate([mb.down(m, DS)[a] for m, (a, _) in zip(ms, sets_all)])
        sn = np.concatenate([mb.down(m, DS)[b] for m, (_, b) in zip(ms, sets_all)])
        sc = np.concatenate([sp, sn])
        return mb.auc(sc, np.arange(sc.size) < sp.size, np.arange(sc.size) >= sp.size)

    tl = json.load(open(os.path.join(cdir(k), 'train_log.json')))
    res = {'fold': k, 'test': names, 'train': folds(pl)[k]['train'], 'windows': per,
           'pooled': {kk: pooled(v) for kk, v in maps.items()},
           'skipped_frac': tl.get('skipped_frac', tl['skipped_nonfinite'] / (tl.get('micro_batches', 2) * tl['iters'])),
           'iters': tl['iters']}
    jsave(res, os.path.join(cdir(k), 'eval.json'), os.path.join(RESULTS, f'eval_fold{k}.json'),
          os.path.join(RUN, 'payload', f'v030-eval-fold{k}.json'))
    timing(f'eval{k}', time.time() - t0)
    log(f"FOLD {k} pooled zero-shot {res['pooled']['zero_shot']:.4f} finetuned {res['pooled']['finetuned']:.4f} "
        f"mrg20736 {res['pooled']['mrg']:.4f} skipped {res['skipped_frac']:.4f}")


# --------------------------------------------------------------------------------------------------------- verdict
def rule(rows, n_folds_done):
    """Registered rule on per-window rows {'zero_shot', 'finetuned'} plus fold health."""
    n = len(rows)
    need = int(np.ceil(FRAC * n)) if n else 0
    if n_folds_done < 3:
        return 'UNKNOWN', f'{n_folds_done} of 3 folds evaluated', need
    if n < 6:
        return 'UNKNOWN', f'only {n} held-out windows (need >= 6)', need
    return None, None, need


def verdict():
    pl = plan(write=False)
    rows, health, evals = [], [], []
    for k in range(3):
        e = jload(os.path.join(cdir(k), 'eval.json')) or jload(os.path.join(RESULTS, f'eval_fold{k}.json'))
        if e is None:
            continue
        evals.append(e)
        health.append(e['skipped_frac'])
        for n, r in e['windows'].items():
            rows.append({'name': n, 'fold': k, **{x: r[x] for x in ('zero_shot', 'finetuned', 'mrg20736_1um')}})
    vd, why, need = rule(rows, len(evals))
    ft = [r['finetuned'] for r in rows]
    zs = [r['zero_shot'] for r in rows]
    above = sum(f > z for f, z in zip(ft, zs))
    mean_ft = float(np.mean(ft)) if ft else None
    if vd is None:
        bad = [r['name'] for r in rows if not np.isfinite(r['finetuned']) or not np.isfinite(r['zero_shot'])
               or r['finetuned'] == 0.0 or r['zero_shot'] == 0.0]
        if bad or max(health) > 0.01:
            vd, why = 'INVALID', f'skipped fractions {health}; undefined or zero AUC on {bad}'
        elif sum(timing_total().values()) > BUDGET_S:
            vd, why = 'UNKNOWN', f'GPU stages took {sum(timing_total().values()) / 3600:.2f} h > 4 h budget'
        elif mean_ft >= MEAN_MIN and above >= need:
            vd, why = 'PASS', f'mean held-out AUC {mean_ft:.4f} >= {MEAN_MIN} and {above} of {len(rows)} windows above zero-shot (need {need})'
        else:
            vd, why = 'FAIL', f'mean held-out AUC {mean_ft:.4f} (need {MEAN_MIN}); {above} of {len(rows)} windows above zero-shot (need {need})'
    mrg = [r['mrg20736_1um'] for r in rows]
    out = {'verdict': vd, 'reason': why, 'n_windows': len(rows), 'need_above': need, 'above_zero_shot': above,
           'mean_finetuned': mean_ft, 'mean_zero_shot': float(np.mean(zs)) if zs else None,
           'mean_mrg20736_1um': float(np.mean(mrg)) if mrg else None, 'skipped_frac': health,
           'pooled': {e['fold']: e['pooled'] for e in evals}, 'rows': rows, 'timing_s': timing_total(),
           'step0': jload(os.path.join(RESULTS, 'step0.json'), {}).get('line'),
           'carry': jload(os.path.join(RESULTS, 'carry.json'), {}).get('line'),
           'rule': {'MEAN_MIN': MEAN_MIN, 'FRAC': '5/6', 'n_min': 6, 'skipped_max': 0.01, 'budget_h': 4}}
    jsave(out, os.path.join(RESULTS, 'summary.json'), os.path.join(RUN, 'payload', 'v030-summary.json'))
    for r in rows:
        log(r['name'], 'fold', r['fold'], f"zero-shot {r['zero_shot']:.4f} finetuned {r['finetuned']:.4f} "
            f"mrg20736 {r['mrg20736_1um']:.4f}")
    log(f"RULE mean held-out AUC {mean_ft if mean_ft is None else round(mean_ft, 4)} (need >= {MEAN_MIN}); "
        f"finetuned > zero-shot on {above} of {len(rows)} windows (need {need}); "
        f"mrg20736-1um mean {out['mean_mrg20736_1um'] if not mrg else round(out['mean_mrg20736_1um'], 4)} (descriptive)")
    log('VERDICT', vd, '-', why)
    return vd


def timing_total():
    t = jload(os.path.join(DATA, 'timing.json'), {})
    return {k: v for k, v in t.items() if k.startswith(('prep', 'train', 'eval'))}


def status():
    s0, c = jload(os.path.join(RESULTS, 'step0.json')), jload(os.path.join(RESULTS, 'carry.json'))
    log('step0', s0['line'] if s0 else 'not run')
    log('carry', c['line'] if c else 'not run')
    pl = jload(os.path.join(DATA, 'windows.json')) or jload(os.path.join(RESULTS, 'windows.json'))
    if not pl:
        log('plan not run'); return
    rend = [p['name'] for p in pl['patches'] if os.path.exists(os.path.join(DATA, 'patches', p['name'] + '.v030.json'))]
    tr = [k for k in range(3) if os.path.exists(os.path.join(cdir(k), 'final.pt'))]
    ev = [k for k in range(3) if os.path.exists(os.path.join(cdir(k), 'eval.json'))]
    n, r, t, e = estimate(pl)
    left = (n - len(rend)) * EST['render_min'] + (3 - len(tr)) * EST['train_min'] + \
        sum(len(folds(pl)[k]['test']) for k in range(3) if k not in ev) * EST['eval_min_per_window']
    used = sum(timing_total().values())
    log(f'rendered {len(rend)}/{n}, trained folds {tr}, evaluated folds {ev}; GPU-stage time {used / 3600:.2f} h of 4 h; '
        f'estimated remaining ~{left:.0f} min')


# --------------------------------------------------------------------------------------------------------- check
def demo():
    """Selftest of the carry and patch maths on synthetic meshes (no network)."""
    # mesh A: a tilted plane, 20 voxels per cell at 2.399; mesh B: the same surface at 1.129, 20 L0 voxels per cell
    Ha, Wa = 40, 60
    ii, jj = np.meshgrid(np.arange(Ha, dtype=float), np.arange(Wa, dtype=float), indexing='ij')
    A = np.stack([1000 + 20 * jj, 2000 + 20 * ii * 0.8, 3000 + 20 * ii * 0.6 + 0.01 * jj ** 2], -1)
    vA = np.ones((Ha, Wa), bool)
    M = np.array([[SCALE, 0, 0, 500.0], [0, SCALE, 0, 600.0], [0, 0, SCALE, 700.0]])      # 1.129 -> 2.399
    Hb, Wb = 60, 100
    bi, bj = np.meshgrid(np.arange(Hb, dtype=float), np.arange(Wb, dtype=float), indexing='ij')
    u_true, v_true = 2 + bi * SCALE, 3 + bj * SCALE
    # B vertex = M^-1 of the point on A at (u_true, v_true): A is linear in i and nearly linear in j
    Ap = np.stack([1000 + 20 * v_true, 2000 + 20 * u_true * 0.8, 3000 + 20 * u_true * 0.6 + 0.01 * v_true ** 2], -1)
    Bm = apply(inverse(M), Ap)
    vB = (u_true <= Ha - 1) & (v_true <= Wa - 1)
    U, V, C, res = carry_planes(A, vA, Bm, vB, M)
    assert C.sum() > 0.9 * vB.sum(), (C.sum(), vB.sum())
    assert np.abs(U[C] - u_true[C]).max() < 0.02 and np.abs(V[C] - v_true[C]).max() < 0.02, 'carry (u, v) off'
    assert np.median(res) < 0.05, np.median(res)
    sv = np.median(local_scale(U, V, C), 0)
    assert np.allclose(sv, SCALE, rtol=0.01), sv
    J = jacobians(A, vA)
    assert np.allclose(J[0][5, 5], [0, 16, 12]) and np.allclose(J[1][5, 5, :2], [20, 0])
    # placement statistic: identical stacks give NCC0 1 at shift 0
    rng = np.random.default_rng(0)
    X = ndimage.gaussian_filter(rng.random((P_LAYERS, 60, 80)).astype(np.float32), 2)
    s = placement_stat(X, X)
    assert s['ncc0'] > 0.999 and s['best_shift_layers_y_x'] == [0, 0, 0], s
    # greedy boxes and the V-020 band chooser with this driver's patch size
    cells = np.zeros((300, 500), np.float32); cells[10:90, 20:180] = 0.5; cells[150:230, 300:460] = 0.3
    ok = np.ones(cells.shape, bool)
    assert [b[0] for b in greedy_boxes(cells, ok, (80, 160), 2)] == [[10, 90, 20, 180], [150, 230, 300, 460]]
    v20.BAND_EDGES, v20.PATCH_H, v20.PATCH_W = (0.5,), 80, 160
    got = v20.choose(cells, ok, np.zeros(cells.shape), 0)
    assert [g[0] for g in got[:2]] == [[10, 90, 20, 180], [150, 230, 300, 460]], got
    lab = np.zeros(cells.shape); lab[100:200, :] = 1.0             # cells marked 1 are too close to the test windows
    got = v20.choose(cells, ok, lab, 0)
    assert all(g[0][1] <= 100 or g[0][0] >= 200 for g in got) and got[0][0] == [10, 90, 20, 180], got
    v20.PATCH_H, v20.PATCH_W = PATCH
    log('DEMO ok (carry, scale, jacobian, placement statistic, boxes, bands)')


def check():
    bad = []

    def ok(name, cond, note=''):
        log('CHECK', 'ok  ' if cond else 'FAIL', name, note)
        if not cond:
            bad.append(name)

    try:
        demo(); ok('selftest (carry, placement, patches)', True)
    except Exception as e:                                               # noqa: BLE001
        ok('selftest (carry, placement, patches)', False, repr(e))
    probe = {'vol_A': VOL_A + '/0/.zarray', 'vol_B_L1': VOL_B + '/1/.zarray', 'transform': URLS['transform'],
             'mesh_A': URLS['mesh_A'] + 'x.tif', 'mesh_B': URLS['mesh_B'] + 'x.tif', 'labels': URLS['labels'] + 'zarr.json',
             'mrg20736-1um': URLS['mrg']}
    for k, u in probe.items():
        try:
            rq = urllib.request.Request(u, headers={'Range': 'bytes=0-1023'})
            with urllib.request.urlopen(rq, timeout=60) as r:
                ok(f'bucket {k}', r.status in (200, 206), str(r.status))
        except Exception as e:                                           # noqa: BLE001
            ok(f'bucket {k}', False, f'{e} {u}')
    for k, f in (('step0.json', 'line'), ('carry.json', 'line')):
        r = jload(os.path.join(RESULTS, k))
        log('CHECK info', k, r[f] if r else 'not committed yet')
    w = jload(os.path.join(RESULTS, 'windows.json'))
    log('CHECK info windows.json', f"{len(w['patches'])} patches, band edges {w['rule']['BAND_EDGES']}" if w else 'not committed yet')
    s = os.environ.get('V030_SRC')
    if s:
        for f in ('render_crop.py', 'render_tifxyz.py', 'infer_crop.py', 'march_bench.py'):
            ok(f'V030_SRC/{f}', os.path.exists(os.path.join(s, f)), s)
    else:
        log('CHECK skip V030_SRC (unset)')
    v = os.environ.get('V030_VILLA')
    if v:
        head = subprocess.run(['git', '-C', v, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        ok('villa commit', head == VILLA_SHA, head)
        ok('villa finetune_on_scan.py', os.path.exists(os.path.join(v, 'ink-detection', 'scan_adapt', 'finetune_on_scan.py')))
        ok('villa optimized_inference', os.path.exists(os.path.join(v, 'ink-detection', 'optimized_inference',
                                                                     'model_resnet3d_3d_decoder.py')))
    else:
        log('CHECK skip villa (V030_VILLA unset)')
    ck = os.environ.get('CANON_CKPT')
    if ck:
        ok('CANON_CKPT', os.path.exists(ck), ck)
    else:
        log('CHECK skip CANON_CKPT (unset)')
    try:
        import torch
        if os.environ.get('INK_DEVICE', 'cuda') == 'cuda':
            ok('cuda', torch.cuda.is_available(), torch.__version__)
    except ImportError:
        log('CHECK skip cuda (no torch in this python)')
    for mod, cond in (('zarr', lambda m: int(m.__version__.split('.')[0]) >= 3), ('cv2', lambda m: True),
                      ('tifffile', lambda m: True), ('PIL', lambda m: True)):
        try:
            m = __import__(mod)
            ok(f'import {mod}', cond(m), getattr(m, '__version__', ''))
        except ImportError as e:
            ok(f'import {mod}', False, repr(e))
    log('CHECK OK' if not bad else f'CHECK FAIL {bad}')
    return not bad


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd in ('check', '--check'):
        sys.exit(0 if check() else 1)
    os.makedirs(DATA, exist_ok=True)
    if cmd == 'demo':
        demo()
    elif cmd == 'step0':
        step0(force='--force' in args)
    elif cmd == 'carry':
        carry()
    elif cmd == 'plan':
        plan()
    elif cmd == 'prep':
        prep(args)
    elif cmd == 'train':
        train(int(args[0]))
    elif cmd == 'eval':
        evaluate(int(args[0]))
    elif cmd == 'verdict':
        verdict()
    elif cmd == 'status':
        status()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
