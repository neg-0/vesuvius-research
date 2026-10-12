"""B-001 data: paired low-res (9.36 / 8.64 um) and high-res (2.2-2.4 um) segments, a CT-only alignment correction per
segment, teacher-labelled training tiles and human-labelled test windows on the published low-res surface volumes.

Spec: experiments/b001/README.md (registered before any B-001 model output). This script implements it.

Stages (each resumable; a stage whose output exists is skipped unless --force):
  resolve            CPU, bucket listing only -> results/b001/segments.json (paths, shapes, roles).
  place [SEG..]      CPU, CT only. 12 crops per segment, NCC of the LR surface volume against the HR surface volume
                     pooled to the LR grid, affine fit of the shift -> results/b001/place/<seg>.json (PLACE_OK/FAIL).
  tiles [SEG..]      CPU, teacher map + CT coverage. Train tiles (13 x 13 cells) with teacher pseudo-labels through
                     the correction -> results/b001/tiles.json, results/b001/tile_labels/<tile>.npz.
  windows [SEG..]    CPU, human labels + CT coverage. Test windows (26 x 26 cells) -> results/b001/windows.json,
                     results/b001/window_labels/<window>.npz (corrected and grid-index labels).
  fetch              desktop (CPU + network). LR surface-volume crops of every tile and window, centre 28 layers,
                     dz applied -> $B001_DATA/{tiles,windows}/<name>.npy (uint8, 28 x H x W). Resumable.
  status             what is done.
  --check            selftest (alignment maths on a synthetic pair, label mapping, fit) and one range read per scroll.

usage: b001_data.py resolve | place [SEG..] [--force] | tiles [SEG..] | windows [SEG..] | fetch | status | --check
SEG is a segment short name from segments.json or a scroll name. Env: B001_RESULTS (default
projects/vesuvius/results/b001), B001_DATA (default out/b001), B001_WORKERS (place, default 4).
"""
import io
import json
import os
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v031'))
import v031_run as v31                                              # noqa: E402  (bucket listing and range reads)

B = v31.B
RESULTS = os.environ.get('B001_RESULTS', os.path.join(HERE, '..', '..', 'results', 'b001'))
DATA = os.environ.get('B001_DATA', 'out/b001')
SCROLLS = {
    'PHerc0139': {'role': 'train', 'lr': '20250728140407', 'hr': '20260102150214', 'um_lr': 9.362, 'um_hr': 2.399},
    'PHerc0814': {'role': 'train', 'lr': '20250804134230', 'hr': '20260309142202', 'um_lr': 9.362, 'um_hr': 2.399},
    'PHerc0343P': {'role': 'train', 'lr': '20250521134555', 'hr': '20260304131111', 'um_lr': 8.64, 'um_hr': 2.215},
    'PHerc0841': {'role': 'test', 'lr': '20250821151531', 'hr': '20260319124803', 'um_lr': 9.366, 'um_hr': 2.403},
    'PHerc0009B': {'role': 'test', 'lr': '20250521125136', 'hr': '20250820154339', 'um_lr': 8.64, 'um_hr': 2.401},
    'PHerc0500P2': {'role': 'test', 'lr': '20250820143440', 'hr': '20250526151718', 'um_lr': 9.362, 'um_hr': 2.215},
}
CELL = 20                               # px per mesh cell on both canvases (tifxyz scale 0.05)
NL = 28                                 # LR layers kept (31-layer 8.64 um stacks -> centre 28)
# place (README "Why alignment")
N_CROPS, CROP, MIN_SEP, COVER = 12, 192, 384, 0.98
DZ, R_COARSE, R_FINE = 3, 20, 2         # layers; LR px searched coarse (at 2x downsampling) and refined
NCC_LAYERS = (8, 21)                    # LR layers 8..20 of the 28 (centre 14)
NCC_OK, NEED_CROPS, RMS_MAX, TRIM_PX = 0.5, 8, 2.0, 3.0
# tiles / windows
T_CELLS, W_CELLS, GRID = 13, 26, 13
CT_FRAC = 0.90
TEACH_POS, TEACH_NEG = 160, 50
T_SUP_MIN, T_INK_MIN, T_PER_SEG = 0.25, 0.02, 30
W_PER_SEG, W_MIN_PX = 12, 200
SEED_TILES = 4501


def log(*a):
    print(*a, flush=True)


def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def jsave(obj, p):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    json.dump(obj, open(p + '.part', 'w'), indent=1)
    os.replace(p + '.part', p)


def segs():
    s = jload(os.path.join(RESULTS, 'segments.json'))
    if s is None:
        raise SystemExit('run resolve first')
    return s


def select(args, role=None, labelled=None):
    S = segs()['segments']
    names = [k for k, v in S.items() if (not args or k in args or v['scroll'] in args) and v['status'] == 'ok']
    if role:
        names = [k for k in names if S[k]['role'] == role]
    if labelled is not None:
        names = [k for k in names if bool(S[k].get('labels')) == labelled]
    return names


# --------------------------------------------------------------------------------------------------------- zarr io
_META = {}


def zmeta(root, level):
    k = (root, level)
    if k not in _META:
        _META[k] = json.loads(v31.fetch(f'{B}{root}/{level}/.zarray'))
    return _META[k]


def zcrop(root, level, y0, x0, h, w, workers=16):
    """(layers, h, w) uint8 crop of an uncompressed zarr v2 surface volume; outside the array or missing chunks -> 0."""
    m = zmeta(root, level)
    assert m['compressor'] is None and m['dtype'] == '|u1' and m['chunks'][0] == m['shape'][0], m
    nz, cy, cx = m['chunks']
    _, H, W = m['shape']
    out = np.zeros((nz, h, w), np.uint8)
    iy0, iy1 = max(0, y0) // cy, (min(H, y0 + h) - 1) // cy
    ix0, ix1 = max(0, x0) // cx, (min(W, x0 + w) - 1) // cx
    jobs = [(iy, ix) for iy in range(iy0, iy1 + 1) for ix in range(ix0, ix1 + 1)]

    def get(j):
        iy, ix = j
        try:
            raw = v31.fetch(f'{B}{root}/{level}/0/{iy}/{ix}')
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return j, None
            raise
        return j, np.frombuffer(raw, np.uint8).reshape(nz, cy, cx)

    with ThreadPoolExecutor(workers) as ex:
        for (iy, ix), c in ex.map(get, jobs):
            if c is None:
                continue
            ys, xs = iy * cy, ix * cx
            a0, a1 = max(y0, ys), min(y0 + h, ys + cy)
            b0, b1 = max(x0, xs), min(x0 + w, xs + cx)
            if a1 > a0 and b1 > b0:
                out[:, a0 - y0:a1 - y0, b0 - x0:b1 - x0] = c[:, a0 - ys:a1 - ys, b0 - xs:b1 - xs]
    return out


def coverage(root, level):
    """CT-covered px (any layer non-zero) of a whole surface volume at a pyramid level."""
    m = zmeta(root, level)
    return (zcrop(root, level, 0, 0, m['shape'][1], m['shape'][2]) > 0).any(0)


def centre28(stack):
    off = (stack.shape[0] - NL) // 2
    assert off >= 0, stack.shape
    return stack[off:off + NL]


def apply_dz(stack, dz):
    """new[j] = old[j + dz], zero-filled: LR layer centre + dz holds the HR surface (README)."""
    out = np.zeros_like(stack)
    if dz >= 0:
        out[:stack.shape[0] - dz] = stack[dz:]
    else:
        out[-dz:] = stack[:stack.shape[0] + dz]
    return out


# --------------------------------------------------------------------------------------------------------- resolve
def resolve_one(scroll, seg):
    cfg = SCROLLS[scroll]
    base = f'{scroll}/segments/{seg}/'
    meshes, _ = v31.ls(base + 'mesh/')
    svs, _ = v31.ls(base + 'surface-volumes/')
    _, dets = v31.ls(base + 'ink-detection/')
    labroots, _ = v31.ls(base + 'ink-labels/')
    r = {'scroll': scroll, 'segment': seg, 'role': cfg['role'], 'k': cfg['um_lr'] / cfg['um_hr']}
    pick = lambda xs, f: [x for x in xs if f(x)]                                             # noqa: E731
    mA = pick(meshes, lambda m: f"-on-{cfg['hr']}-" in m)
    mB = pick(meshes, lambda m: f"-on-{cfg['lr']}-" in m)
    sA = pick(svs, lambda m: f"volume-{cfg['hr']}.zarr" in m)
    sB = pick(svs, lambda m: f"volume-{cfg['lr']}.zarr" in m)
    te = pick([k for k, _ in dets], lambda k: 'new_canon_autoresearch_recipe' in k and f"volume-{cfg['hr']}-" in k)
    lr = pick(labroots, lambda m: cfg['hr'] in m)
    missing = [n for n, x in (('mesh_hr', mA), ('mesh_lr', mB), ('sv_hr', sA), ('sv_lr', sB), ('teacher', te))
               if len(x) != 1]
    if missing:
        r['status'] = 'missing ' + ','.join(missing)
        return r
    r.update({'mesh_hr': mA[0], 'mesh_lr': mB[0], 'sv_hr': sA[0].rstrip('/'), 'sv_lr': sB[0].rstrip('/'),
              'teacher': te[0], 'labels': None, 'status': 'ok'})
    if lr:
        dates, _ = v31.ls(lr[0])
        r['labels'] = sorted(dates)[-1]
    r['sv_lr_shape'] = zmeta(r['sv_lr'], 0)['shape']
    r['sv_hr_shape'] = zmeta(r['sv_hr'], 0)['shape']
    r['mesh_hr_shape'] = v31.tif_shape(B + r['mesh_hr'] + 'x.tif')[0]
    r['mesh_lr_shape'] = v31.tif_shape(B + r['mesh_lr'] + 'x.tif')[0]
    r['teacher_shape'] = v31.tif_shape(B + r['teacher'])[0]
    ok = (r['sv_lr_shape'][1:] == [n * CELL for n in r['mesh_lr_shape']]
          and r['sv_hr_shape'][1:] == [n * CELL for n in r['mesh_hr_shape']] == r['teacher_shape']
          and r['sv_lr_shape'][0] >= NL)
    if not ok:
        r['status'] = 'shape mismatch'
    if cfg['role'] == 'test' and not r['labels']:
        r['status'] = 'unlabelled test segment (not used)'
    return r


def resolve():
    out = {'bucket': B, 'listed_utc': time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()), 'scrolls': SCROLLS,
           'segments': {}}
    jobs = []
    for scroll in SCROLLS:
        ps, _ = v31.ls(f'{scroll}/segments/')
        jobs += [(scroll, p.rstrip('/').split('/')[-1]) for p in sorted(ps) if not p.endswith('/raw/')]
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda j: resolve_one(*j), jobs))
    for (scroll, seg), r in zip(jobs, res):
        short = f"{scroll}_{seg.split('-')[0]}"
        while short in out['segments']:
            short += 'b'
        out['segments'][short] = r
        log('RESOLVE', short, r['status'], r.get('sv_lr_shape'), r.get('sv_hr_shape'), 'labels' if r.get('labels') else '')
    names = sorted(out['segments'])
    for i, n in enumerate(names):
        out['segments'][n]['index'] = i
    out['segments'] = {n: out['segments'][n] for n in names}
    jsave(out, os.path.join(RESULTS, 'segments.json'))
    for sc in SCROLLS:
        S = [v for v in out['segments'].values() if v['scroll'] == sc]
        log('RESOLVE', sc, SCROLLS[sc]['role'], f"{sum(v['status'] == 'ok' for v in S)} usable of {len(S)}")


# --------------------------------------------------------------------------------------------------------- place
def pooled_hr(s, y0, x0, h, w):
    """HR surface volume pooled onto the LR grid of LR px box (y0, x0, h, w): layers by physical depth bins of the LR
    layers, in-plane by bilinear resampling of HR level 2 at LR px centres (grid-index convention: LR px centre
    p + 0.5 <-> HR level-0 px (p + 0.5) k - 0.5)."""
    import cv2
    k = s['k']
    nh, nl = s['sv_hr_shape'][0], s['sv_lr_shape'][0]
    f = k / 4.0
    hy0, hx0 = int(np.floor(y0 * f)) - 2, int(np.floor(x0 * f)) - 2
    hh, hw = int(np.ceil(h * f)) + 5, int(np.ceil(w * f)) + 5
    hr = zcrop(s['sv_hr'], 2, hy0, hx0, hh, hw).astype(np.float32)
    um = SCROLLS[s['scroll']]
    zh = (np.arange(nh) - (nh - 1) / 2) * um['um_hr']
    out = np.zeros((nl, hh, hw), np.float32)
    for j in range(nl):
        c = (j - (nl - 1) / 2) * um['um_lr']
        sel = (zh >= c - um['um_lr'] / 2) & (zh < c + um['um_lr'] / 2)
        if sel.any():
            out[j] = hr[sel].mean(0)
    ys = ((np.arange(h) + y0 + 0.5) * k - 0.5 - 1.5) / 4.0 - hy0
    xs = ((np.arange(w) + x0 + 0.5) * k - 0.5 - 1.5) / 4.0 - hx0
    mx, my = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    return np.stack([cv2.remap(out[j], mx, my, cv2.INTER_LINEAR, borderValue=0) for j in range(nl)])


def ncc(a, b, m):
    a, b = a[m], b[m]
    if a.size < 500:
        return -2.0
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else -2.0


def best_shift(lr, hq, R, layers):
    """lr: (L, S + 2R, S + 2R) LR stack with margin R; hq: (L, S, S) pooled HR on the unshifted box. Returns
    (ncc0, best, dz, dy, dx): LR content at (layer c + dz, px p + d) matches HR at (c, p)."""
    L, S = hq.shape[0], hq.shape[1]
    z0, z1 = layers
    tgt = hq[z0:z1]
    hv = (tgt > 0).all(0)

    def score(dz, dy, dx, a=lr, t=tgt, hvm=hv, r=R, s=S):
        if z0 + dz < 0 or z1 + dz > L:
            return -2.0
        x = a[z0 + dz:z1 + dz, r + dy:r + dy + s, r + dx:r + dx + s]
        m = hvm & (x > 0).all(0)
        return ncc(x, t, np.broadcast_to(m, x.shape))

    n0 = score(0, 0, 0)
    # coarse: 2x mean pooled
    pool = lambda x: x[..., :x.shape[-2] // 2 * 2, :x.shape[-1] // 2 * 2].reshape(            # noqa: E731
        *x.shape[:-2], x.shape[-2] // 2, 2, x.shape[-1] // 2, 2).mean((-3, -1))
    lr2, t2 = pool(lr), pool(tgt)
    hv2 = pool(hv.astype(np.float32)) > 0.999
    Rc, Sc = R // 2, S // 2
    best = (-3.0, 0, 0, 0)
    for dz in range(-DZ, DZ + 1):
        if z0 + dz < 0 or z1 + dz > L:
            continue
        for dy in range(-R_COARSE // 2, R_COARSE // 2 + 1):
            for dx in range(-R_COARSE // 2, R_COARSE // 2 + 1):
                x = lr2[z0 + dz:z1 + dz, Rc + dy:Rc + dy + Sc, Rc + dx:Rc + dx + Sc]
                m = hv2 & (x > 0).all(0)
                c = ncc(x, t2, np.broadcast_to(m, x.shape))
                if c > best[0]:
                    best = (c, dz, 2 * dy, 2 * dx)
    _, bz, by, bx = best
    best = (-3.0, 0, 0, 0)
    for dz in range(bz - 1, bz + 2):
        for dy in range(by - R_FINE, by + R_FINE + 1):
            for dx in range(bx - R_FINE, bx + R_FINE + 1):
                if abs(dy) > R or abs(dx) > R:
                    continue
                c = score(dz, dy, dx)
                if c > best[0]:
                    best = (c, dz, dy, dx)
    return n0, best[0], best[1], best[2], best[3]


def crop_positions(cov4, idx, shape):
    """N_CROPS box origins (LR px) whose CROP box is >= COVER covered (level-4 mask), spread >= MIN_SEP apart."""
    _, H, W = shape
    g = CROP // 16
    cands = []
    for y in range(0, H - CROP - 2 * R_COARSE, 32):
        for x in range(0, W - CROP - 2 * R_COARSE, 32):
            a = cov4[y // 16:y // 16 + g + 1, x // 16:x // 16 + g + 1]
            if a.size and a.mean() >= COVER:
                cands.append((y, x))
    rng = np.random.default_rng(idx)
    order = [cands[i] for i in rng.permutation(len(cands))]
    chosen = []
    for c in order:
        if all(max(abs(c[0] - a), abs(c[1] - b)) >= MIN_SEP for a, b in chosen):
            chosen.append(c)
        if len(chosen) == N_CROPS:
            break
    for c in order:
        if len(chosen) == N_CROPS:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def fit_shift(rows):
    """Affine d(p) = [1, y, x] @ coef fitted to informative crops' best shifts, one trimming pass."""
    pts = [r for r in rows if r['best'] >= NCC_OK]
    out = {'informative': len(pts)}
    if len(pts) < 3:
        return dict(out, coef=None, rms=None, used=0)

    def fit(pp):
        A = np.array([[1.0, *r['centre']] for r in pp])
        Y = np.array([[r['dy'], r['dx']] for r in pp], float)
        coef = np.linalg.lstsq(A, Y, rcond=None)[0]
        res = np.linalg.norm(A @ coef - Y, axis=1)
        return coef, res

    coef, res = fit(pts)
    keep = [p for p, e in zip(pts, res) if e <= TRIM_PX]
    if len(keep) >= 3 and len(keep) < len(pts):
        coef, res = fit(keep)
        pts = keep
    return dict(out, coef=coef.tolist(), rms=float(np.sqrt((res ** 2).mean())), used=len(pts))


def place_one(name, force=False):
    p = os.path.join(RESULTS, 'place', name + '.json')
    if os.path.exists(p) and not force:
        return jload(p)
    s = segs()['segments'][name]
    t0 = time.time()
    cov4 = coverage(s['sv_lr'], 4)
    pos = crop_positions(cov4, s['index'], s['sv_lr_shape'])
    R = R_COARSE + R_FINE
    rows = []
    for (y0, x0) in pos:
        lr = centre28(zcrop(s['sv_lr'], 0, y0 - R, x0 - R, CROP + 2 * R, CROP + 2 * R)).astype(np.float32)
        hq = centre28(pooled_hr(s, y0, x0, CROP, CROP))
        n0, b, dz, dy, dx = best_shift(lr, hq, R, NCC_LAYERS)
        rows.append({'origin': [y0, x0], 'centre': [y0 + CROP / 2, x0 + CROP / 2], 'ncc0': round(n0, 4),
                     'best': round(b, 4), 'dz': dz, 'dy': dy, 'dx': dx})
    f = fit_shift(rows)
    dzs = [r['dz'] for r in rows if r['best'] >= NCC_OK]
    ok = f['coef'] is not None and f['used'] >= NEED_CROPS and f['rms'] <= RMS_MAX
    out = {'segment': name, 'verdict': 'PLACE_OK' if ok else 'PLACE_FAIL', 'k': s['k'], 'crops': rows, 'fit': f,
           'dz': int(np.median(dzs)) if dzs else 0,
           'rule': dict(N_CROPS=N_CROPS, CROP=CROP, DZ=DZ, R_COARSE=R_COARSE, R_FINE=R_FINE, NCC_LAYERS=NCC_LAYERS,
                        NCC_OK=NCC_OK, NEED_CROPS=NEED_CROPS, RMS_MAX=RMS_MAX, TRIM_PX=TRIM_PX),
           'seconds': round(time.time() - t0, 1)}
    if f['coef'] is not None:
        _, H, W = s['sv_lr_shape']
        corners = np.array([[1, y, x] for y in (0, H) for x in (0, W)], float) @ np.array(f['coef'])
        out['shift_px_at_corners'] = np.round(corners, 2).tolist()
        out['shift_um_max'] = round(float(np.linalg.norm(corners, axis=1).max()) * SCROLLS[s['scroll']]['um_lr'], 1)
    out['line'] = (f"{out['verdict']}: {f['used']} of {len(rows)} crops used (informative {f['informative']}, need "
                   f"{NEED_CROPS}), fit RMS {f['rms'] if f['rms'] is None else round(f['rms'], 2)} px (<= {RMS_MAX}), "
                   f"median NCC {np.median([r['ncc0'] for r in rows]):.3f} -> {np.median([r['best'] for r in rows]):.3f}, "
                   f"dz {out['dz']}, max shift {out.get('shift_um_max')} um")
    jsave(out, p)
    log('PLACE', name, out['line'], f"{out['seconds']}s")
    return out


# --------------------------------------------------------------------------------------------------------- mapping
def hr_index(pl, yy, xx, corrected=True):
    """HR level-0 px (nearest) holding the content of LR px (yy, xx): grid index after removing the fitted shift."""
    k = pl['k']
    yy, xx = np.asarray(yy, np.float64), np.asarray(xx, np.float64)
    if corrected:
        c = np.array(pl['fit']['coef'])
        dy = c[0, 0] + c[1, 0] * yy + c[2, 0] * xx
        dx = c[0, 1] + c[1, 1] * yy + c[2, 1] * xx
        yy, xx = yy - dy, xx - dx
    return np.rint((yy + 0.5) * k - 0.5).astype(np.int64), np.rint((xx + 0.5) * k - 0.5).astype(np.int64)


def sample(arr, pl, y0, x0, h, w, corrected=True):
    """Values of an HR-canvas array at the LR px box (y0, x0, h, w); out of bounds -> -1."""
    yy, xx = np.meshgrid(np.arange(y0, y0 + h), np.arange(x0, x0 + w), indexing='ij')
    iy, ix = hr_index(pl, yy, xx, corrected)
    ok = (iy >= 0) & (ix >= 0) & (iy < arr.shape[0]) & (ix < arr.shape[1])
    out = np.full((h, w), -1, np.int16)
    out[ok] = arr[iy[ok], ix[ok]]
    return out


def read_teacher(s):
    import tifffile
    return tifffile.imread(io.BytesIO(v31.fetch(B + s['teacher'])))


def read_labels(s):
    """Human ink labels and supervision (level 0, HR canvas), bool."""
    import tempfile
    import zarr
    out = {}
    for z in ('inklabels', 'supervision'):
        root = f"{s['labels']}{z}.zarr/0/"
        meta = json.loads(v31.fetch(B + root + 'zarr.json'))
        d = tempfile.mkdtemp(prefix='b001lab')
        grid = [-(-n // c) for n, c in zip(meta['shape'], meta['chunk_grid']['configuration']['chunk_shape'])]
        for i in range(grid[0]):
            for j in range(grid[1]):
                try:
                    raw = v31.fetch(f'{B}{root}c/{i}/{j}')
                except urllib.error.HTTPError as e:
                    if e.code in (403, 404):
                        continue
                    raise
                os.makedirs(os.path.join(d, 'c', str(i)), exist_ok=True)
                open(os.path.join(d, 'c', str(i), str(j)), 'wb').write(raw)
        json.dump(meta, open(os.path.join(d, 'zarr.json'), 'w'))
        out[z] = np.asarray(zarr.open_array(d, mode='r')[:]) > 0
        assert list(out[z].shape) == s['teacher_shape'], (out[z].shape, s['teacher_shape'])
    return out


def grid_boxes(s, cells, cov2):
    """Cores of `cells` x `cells` mesh cells on a GRID-cell grid from (0, 0), >= CT_FRAC covered (level-2 mask)."""
    _, H, W = s['sv_lr_shape']
    n = cells * CELL
    out = []
    for r in range(0, H // CELL - cells + 1, GRID):
        for c in range(0, W // CELL - cells + 1, GRID):
            y0, x0 = r * CELL, c * CELL
            a = cov2[y0 // 4:(y0 + n) // 4, x0 // 4:(x0 + n) // 4]
            if a.size and a.mean() >= CT_FRAC:
                out.append((y0, x0))
    return out


def cov_px(cov2, y0, x0, n):
    return np.repeat(np.repeat(cov2[y0 // 4:(y0 + n) // 4, x0 // 4:(x0 + n) // 4], 4, 0), 4, 1)[:n, :n]


# --------------------------------------------------------------------------------------------------------- tiles
def tiles(names):
    reg = jload(os.path.join(RESULTS, 'tiles.json')) or {'tiles': {}, 'segments': {}}
    reg['rule'] = dict(T_CELLS=T_CELLS, GRID=GRID, CT_FRAC=CT_FRAC, TEACH_POS=TEACH_POS, TEACH_NEG=TEACH_NEG,
                       T_SUP_MIN=T_SUP_MIN, T_INK_MIN=T_INK_MIN, T_PER_SEG=T_PER_SEG, SEED_TILES=SEED_TILES)
    S = segs()['segments']
    for name in names:
        if name in reg['segments']:
            continue
        pl = jload(os.path.join(RESULTS, 'place', name + '.json'))
        if pl is None:
            raise SystemExit(f'tiles {name}: run place first')
        s = S[name]
        if pl['verdict'] != 'PLACE_OK':
            reg['segments'][name] = {'stopped': pl['line'], 'tiles': 0}
            jsave(reg, os.path.join(RESULTS, 'tiles.json'))
            continue
        t0 = time.time()
        teach = read_teacher(s)
        assert list(teach.shape) == s['teacher_shape']
        cov2 = coverage(s['sv_lr'], 2)
        n = T_CELLS * CELL
        dense, rest = [], []
        for (y0, x0) in grid_boxes(s, T_CELLS, cov2):
            t = sample(teach, pl, y0, x0, n, n)
            cp = cov_px(cov2, y0, x0, n)
            ink = (t >= TEACH_POS) & cp
            sup = ((t >= TEACH_POS) | ((t >= 0) & (t <= TEACH_NEG))) & cp
            if sup.mean() < T_SUP_MIN:
                continue
            item = (float(ink.mean()), y0, x0, ink, sup)
            (dense if item[0] >= T_INK_MIN else rest).append(item)
        dense.sort(key=lambda a: (-a[0], a[1], a[2]))
        chosen = dense[:T_PER_SEG]
        if len(chosen) < T_PER_SEG and rest:
            rng = np.random.default_rng(SEED_TILES + s['index'])
            idx = rng.permutation(len(rest))[:T_PER_SEG - len(chosen)]
            chosen += [rest[i] for i in sorted(idx)]
        os.makedirs(os.path.join(RESULTS, 'tile_labels'), exist_ok=True)
        for fr, y0, x0, ink, sup in chosen:
            tid = f'{name}_t{y0}_{x0}'
            np.savez_compressed(os.path.join(RESULTS, 'tile_labels', tid + '.npz'), ink=np.packbits(ink),
                                supervision=np.packbits(sup), shape=np.array([n, n]))
            reg['tiles'][tid] = {'segment': name, 'scroll': s['scroll'], 'origin': [y0, x0], 'px': [n, n],
                                 'dz': pl['dz'], 'teacher_ink_frac': round(fr, 4),
                                 'supervised_frac': round(float(sup.mean()), 4)}
        reg['segments'][name] = {'tiles': len(chosen), 'dense_candidates': len(dense), 'other_candidates': len(rest),
                                 'seconds': round(time.time() - t0, 1)}
        reg['tiles'] = dict(sorted(reg['tiles'].items()))
        jsave(reg, os.path.join(RESULTS, 'tiles.json'))
        log('TILES', name, json.dumps(reg['segments'][name]))
    per = {}
    for t in reg['tiles'].values():
        per[t['scroll']] = per.get(t['scroll'], 0) + 1
    log('TILES total', len(reg['tiles']), json.dumps(per))


# --------------------------------------------------------------------------------------------------------- windows
def windows(names):
    reg = jload(os.path.join(RESULTS, 'windows.json')) or {'windows': {}, 'segments': {}}
    reg['rule'] = dict(W_CELLS=W_CELLS, GRID=GRID, CT_FRAC=CT_FRAC, W_PER_SEG=W_PER_SEG, W_MIN_PX=W_MIN_PX)
    S = segs()['segments']
    for name in names:
        if name in reg['segments']:
            continue
        pl = jload(os.path.join(RESULTS, 'place', name + '.json'))
        if pl is None:
            raise SystemExit(f'windows {name}: run place first')
        s = S[name]
        if pl['verdict'] != 'PLACE_OK':
            reg['segments'][name] = {'stopped': pl['line'], 'windows': 0}
            jsave(reg, os.path.join(RESULTS, 'windows.json'))
            continue
        t0 = time.time()
        lab = read_labels(s)
        cov2 = coverage(s['sv_lr'], 2)
        n = W_CELLS * CELL
        cands = []
        for (y0, x0) in grid_boxes(s, W_CELLS, cov2):
            cp = cov_px(cov2, y0, x0, n)
            ink = (sample(lab['inklabels'], pl, y0, x0, n, n) > 0) & cp
            sup = (sample(lab['supervision'], pl, y0, x0, n, n) > 0) & cp
            pos, neg = int((ink & sup).sum()), int((~ink & sup).sum())
            if pos >= W_MIN_PX and neg >= W_MIN_PX:
                cands.append((pos, y0, x0, ink, sup))
        cands.sort(key=lambda a: (-a[0], a[1], a[2]))
        chosen = []
        for c in cands:
            if all(abs(c[1] - a[1]) >= n or abs(c[2] - a[2]) >= n for a in chosen):
                chosen.append(c)
            if len(chosen) == W_PER_SEG:
                break
        os.makedirs(os.path.join(RESULTS, 'window_labels'), exist_ok=True)
        for pos, y0, x0, ink, sup in chosen:
            wid = f'{name}_w{y0}_{x0}'
            g_ink = (sample(lab['inklabels'], pl, y0, x0, n, n, corrected=False) > 0)
            g_sup = (sample(lab['supervision'], pl, y0, x0, n, n, corrected=False) > 0)
            np.savez_compressed(os.path.join(RESULTS, 'window_labels', wid + '.npz'), ink=np.packbits(ink),
                                supervision=np.packbits(sup), ink_gridindex=np.packbits(g_ink),
                                supervision_gridindex=np.packbits(g_sup), shape=np.array([n, n]))
            reg['windows'][wid] = {'segment': name, 'scroll': s['scroll'], 'origin': [y0, x0], 'px': [n, n],
                                   'dz': pl['dz'], 'labelled_ink_px': pos, 'supervised_px': int(sup.sum())}
        reg['segments'][name] = {'windows': len(chosen), 'candidates': len(cands), 'seconds': round(time.time() - t0, 1)}
        reg['windows'] = dict(sorted(reg['windows'].items()))
        jsave(reg, os.path.join(RESULTS, 'windows.json'))
        log('WINDOWS', name, json.dumps(reg['segments'][name]))
    per = {}
    for w in reg['windows'].values():
        per[w['scroll']] = per.get(w['scroll'], 0) + 1
    log('WINDOWS total', len(reg['windows']), json.dumps(per))


# --------------------------------------------------------------------------------------------------------- fetch
def load_labels(kind, name):
    d = np.load(os.path.join(RESULTS, f'{kind}_labels', name + '.npz'))
    n = int(np.prod(d['shape']))
    return (np.unpackbits(d['ink'])[:n].reshape(d['shape']).astype(bool),
            np.unpackbits(d['supervision'])[:n].reshape(d['shape']).astype(bool))


def fetch_one(kind, name, item):
    out = os.path.join(DATA, kind, name + '.npy')
    if os.path.exists(out):
        return False
    s = segs()['segments'][item['segment']]
    y0, x0 = item['origin']
    h, w = item['px']
    st = apply_dz(centre28(zcrop(s['sv_lr'], 0, y0, x0, h, w)), item['dz'])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + '.part.npy', st)
    os.replace(out + '.part.npy', out)
    return True


def fetch():
    t0 = time.time()
    for kind, f in (('windows', 'windows.json'), ('tiles', 'tiles.json')):
        reg = jload(os.path.join(RESULTS, f))
        if reg is None:
            raise SystemExit(f'{f} missing: commit the data stages first')
        items = reg[kind]
        done = 0
        for i, (n, it) in enumerate(items.items()):
            done += fetch_one(kind, n, it)
            if (i + 1) % 50 == 0 or i + 1 == len(items):
                log(f'FETCH {kind} {i + 1}/{len(items)} (new {done}) {time.time() - t0:.0f}s')


def status():
    S = segs()['segments']
    for role in ('train', 'test'):
        ns = select([], role, None if role == 'train' else True)
        pl = [jload(os.path.join(RESULTS, 'place', n + '.json')) for n in ns]
        log(role, len(ns), 'usable; placed', sum(p is not None for p in pl), 'PLACE_OK',
            sum(1 for p in pl if p and p['verdict'] == 'PLACE_OK'))
    for f, k in (('tiles.json', 'tiles'), ('windows.json', 'windows')):
        r = jload(os.path.join(RESULTS, f))
        log(f, 'missing' if r is None else f"{len(r[k])} {k}")
        if r is not None:
            have = sum(os.path.exists(os.path.join(DATA, k, n + '.npy')) for n in r[k])
            log('  fetched', have, 'of', len(r[k]))
    del S


# --------------------------------------------------------------------------------------------------------- check
def selftest():
    rng = np.random.default_rng(0)
    # best_shift recovers a known shift on a synthetic layered texture
    from scipy import ndimage
    L, S, R = NL, 96, R_COARSE + R_FINE
    vol = ndimage.gaussian_filter(rng.random((L + 8, S + 2 * R + 20, S + 2 * R + 20)), 2) * 200 + 20
    dz, dy, dx = 1, -7, 5
    lr = vol[4 + dz:4 + dz + L, 10 + dy:10 + dy + S + 2 * R, 10 + dx:10 + dx + S + 2 * R]
    hq = vol[4:4 + L, 10 + R:10 + R + S, 10 + R:10 + R + S]
    # lr[c + dz, R + d + i] = vol[4 + dz + c + dz', ...]: LR content at (c + dz, p + d) equals HR at (c, p) when the LR
    # window is shifted by (-dz, -dy, -dx) relative to vol; so the search must find (dz, dy, dx) = (-1, 7, -5)
    n0, b, z, y, x = best_shift(lr.astype(np.float32), hq.astype(np.float32), R, (8, 21))
    assert (z, y, x) == (-dz, -dy, -dx) and b > 0.99 > n0, (n0, b, z, y, x)
    # fit_shift recovers an affine field and trims an outlier
    rows = []
    for yy in range(0, 4000, 500):
        for xx in (100, 2100, 3900):
            d = np.array([1.5 + 0.002 * yy, -3 + 0.001 * xx])
            rows.append({'centre': [yy, xx], 'best': 0.9, 'dy': round(d[0]), 'dx': round(d[1])})
    rows[3]['dy'] += 15
    f = fit_shift(rows)
    assert f['used'] == len(rows) - 1 and f['rms'] < 0.6, f
    # hr_index: zero shift is the grid-index convention; a shift moves the source the other way
    pl = {'k': 4.0, 'fit': {'coef': [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]}}
    iy, ix = hr_index(pl, np.array([10.0]), np.array([10.0]), corrected=False)
    assert (iy[0], ix[0]) == (42, 42), (iy, ix)
    iy, ix = hr_index(pl, np.array([10.0]), np.array([10.0]))
    assert (iy[0], ix[0]) == (34, 42), (iy, ix)
    # apply_dz
    a = np.arange(28)[:, None, None] * np.ones((1, 2, 2))
    assert apply_dz(a, 2)[0, 0, 0] == 2 and apply_dz(a, 2)[-1, 0, 0] == 0 and apply_dz(a, -1)[1, 0, 0] == 0
    log('SELFTEST ok: best_shift', (z, y, x), f'ncc {n0:.3f} -> {b:.3f}; fit rms {f["rms"]:.2f} px (outlier trimmed)')


def check():
    selftest()
    if os.path.exists(os.path.join(RESULTS, 'segments.json')):
        S = segs()['segments']
        for sc in SCROLLS:
            ns = [n for n in select([sc])]
            if ns:
                s = S[ns[0]]
                a = zcrop(s['sv_lr'], 4, 0, 0, 8, 8)
                log('CHECK', sc, ns[0], 'LR sv level-4 read', a.shape, 'ok')


def main():
    a = sys.argv[1:]
    if not a or a[0] in ('-h', '--help'):
        print(__doc__)
        return
    cmd, rest = a[0], [x for x in a[1:] if not x.startswith('--')]
    force = '--force' in a
    if cmd == '--check':
        check()
    elif cmd == 'resolve':
        resolve()
    elif cmd == 'place':
        names = select(rest, 'train') + select(rest, 'test', True)
        todo = [n for n in names if force or not os.path.exists(os.path.join(RESULTS, 'place', n + '.json'))]
        log('PLACE', len(names), 'segments,', len(todo), 'to do')
        with ThreadPoolExecutor(int(os.environ.get('B001_WORKERS', '4'))) as ex:
            list(ex.map(lambda n: place_one(n, force), todo))
        ok = sum(1 for n in names if (jload(os.path.join(RESULTS, 'place', n + '.json')) or {}).get('verdict') == 'PLACE_OK')
        log('PLACE done:', ok, 'of', len(names), 'PLACE_OK')
    elif cmd == 'tiles':
        tiles(select(rest, 'train'))
    elif cmd == 'windows':
        windows(select(rest, 'test', True))
    elif cmd == 'fetch':
        fetch()
    elif cmd == 'status':
        status()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
