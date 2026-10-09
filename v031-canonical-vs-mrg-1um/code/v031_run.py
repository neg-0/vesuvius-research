"""V-031 driver: does the stock canonical 2.4 um model, zero-shot on 1.129 um renders (2.258 um, level 1, along the
1.129 um mesh), read held-out ink as well as the organisers' published mrg20736-1um prediction of the same segment?
PHerc1667 (6 segments, refit transform, villa #1843) and PHercParis4 (6 segments), scored on identical pixels.

Spec: experiments/v031/README.md (registered before any V-031 score). This script implements it and does not change it.
It reuses V-030's code by import (experiments/v030/v030_run.py: transforms maths, placement statistic, greedy boxes,
the 3D carry, render_crop call, pixel sets, AUC, inference) and V-020's window chooser (via v030_run.v20).

Stages (each resumable; a stage whose output exists is skipped; later stages refuse to run after a failed gate):
  resolve  CPU, bucket listing only. Exact paths per segment -> results/.../segments.json (committed).
  step0 [SEG..]   CPU. Placement of the 1.129 um mesh against the labelled 2.4 um mesh, 6 label-chosen crops, V-030's
           model-free NCC statistic. PHerc1667: the refit mesh B' = M_lsq^-1 M_pub B is required (villa #1843);
           PHercParis4: published mesh B, refit only if it fails (V-030). Prints PLACEMENT_* per segment.
  carry [SEG..]   CPU. 2.4 um labels -> 1.129 um L1 canvas by 3D correspondence (V-030 step 1), restricted to the
           part of the mesh where the 1.129 um scan has data (ROI). Prints CARRY_OK / CARRY_FAIL per segment.
  plan [SEG..]    CPU, labels and mesh only. V-030 held-out windows (40 x 80 B cells, 3 z bands per segment, <= 6 per
           band, greedy by carried-ink density). PLAN SAME / DIFFERENT against the committed windows.json; ESTIMATE.
  render [--scroll S] [names]  GPU machine (CPU + network). Each window on the 1.129 um volume, level 1, 63 layers,
           along the step-2 mesh; PHerc1667 also along the published mesh (descriptive arm).
  score [--scroll S]  GPU. Canonical zero-shot (bf16) vs mrg20736-1um per window, offset 0, identical ds8 pixels.
  rule     CPU. RULE and VERDICT lines (pooled = verdict; per scroll alongside), summary.json, RESULT.md.
  status   what is done, GPU-stage time, estimate for the rest.
  check    (--check) selftest (V-030 demo + V-031 maths), env, villa commit, every bucket URL (1 kB range reads).

usage: v031_run.py check | resolve | step0 [SEG..] [--force] | carry [SEG..] | plan [SEG..] | render [--scroll S] [names]
       | score [--scroll S] | rule | status
SEG is a short name from segments.json (e.g. 1667_w013, P4_20230702185753) or a scroll (PHerc1667, PHercParis4).
Env: V031_DATA (work folder), V031_RESULTS (default projects/vesuvius/results/v031-canonical-vs-mrg-1um),
V031_SRC (folder with render_crop.py, render_tifxyz.py, infer_crop.py, march_bench.py: the V-016 src tree),
V031_VILLA (neg-0/villa checkout at 7a5ba5a), CANON_CKPT, XSCAN_CACHE, PUZZLE_RUN_DIR, INK_DEVICE (default cuda),
V031_LOWDISK=1 (cloud: carry arrays stay in memory for `cloud`; nothing large is written). The CT chunk cache is
removed after every render; meshes, label shards and the mrg map are read from the bucket, never downloaded whole.

Desktop run: see the GPU-QUEUE item (44, or 44/45 when split per scroll) and README "Desktop run".
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
for _a, _b in (('V031_SRC', 'V030_SRC'), ('V031_VILLA', 'V030_VILLA')):
    if os.environ.get(_a):
        os.environ[_b] = os.environ[_a]
sys.path.insert(0, os.path.join(HERE, '..', 'v030'))
import v030_run as v30                                              # noqa: E402  (V-030 maths, unchanged)

v20 = v30.v20
log, jload, jsave, apply, inverse = v30.log, v30.jload, v30.jsave, v30.apply, v30.inverse
B = v30.B
SCROLLS = {
    'PHerc1667': {'short': '1667', 'um_A': 2.399, 'scan_A': '20251217075048', 'scan_B': '20260323082859',
                  'vol_A': 'PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr',
                  'vol_B': 'PHerc1667/volumes/20260323082859-1.129um-0.2m-59keV-masked.zarr',
                  'refit_required': True, 'study': 'V-031',
                  'segments': ['20240304141531-w013_20240304141531_flatboi', '20240304144031-w018_20240304144031_flatboi',
                               '20240304161941-w023_20240304161941_flatboi', '20251208130119-w028_20251208130119156_flatboi',
                               '20251212185248-w029_20251212185248662_flatboi', '20251223230000-w031_2025122323_flatboi']},
    'PHercParis4': {'short': 'P4', 'um_A': 2.400, 'scan_A': '20260411134726', 'scan_B': '20260608103018',
                    'vol_A': 'PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr',
                    'vol_B': 'PHercParis4/volumes/20260608103018-1.129um-0.2m-78keV-masked.zarr',
                    'refit_required': False, 'study': 'V-031',
                    'segments': ['20230702185753', '20231007101619', '20231012184424', '20231031143852', '20231106155351',
                                 '20231210121321']},
    # V-031b (2026-10-08, roadmap follow-up): the 8 labelled PHerc0139 segments with an mrg20736-1um map; same rule,
    # reported per scroll and pooled with V-031 as an extension (V-031's registered verdict is unchanged)
    'PHerc0139': {'short': '0139', 'um_A': 2.399, 'scan_A': '20260102150214', 'scan_B': '20260413113053',
                  'vol_A': 'PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr',
                  'vol_B': 'PHerc0139/volumes/20260413113053-1.129um-0.2m-59keV-masked.zarr',
                  'refit_required': False, 'study': 'V-031b',
                  'segments': ['20250108000005-w030_2025010818', '20260317000000-w035_2026031718',
                               '20260302000000-w039_2026030210', '20250831000000-w040_2025083102',
                               '20260108000000-w041_2026010816', '20260112000000-w043_2026011217',
                               '20260115000000-w044_2026011522', '20260126000000-w045_2026012619']},
}
STUDY_FILE = {'V-031': 'windows.json', 'V-031b': 'windows_v031b.json'}
# V-031b descriptive depth column (villa #1912, V-013/V-021): on w044 and w045 the canonical model read best 32 layers
# below the PHerc0139 March-scan mesh (V-013 renders: 2.403 um scan, level 1, oversample 2 -> 2.403 um per layer; the
# window centred at stack centre + offset, offset -32 = -76.9 um along the render normal). On the 2.258 um layer step of
# these renders that is -76.9 / 2.258 = -34.06 -> -34 layers. Those windows are rendered 63 + 2 * 34 = 131 layers deep;
# the centre 63 are exactly the 63-layer render (same layer positions), so the main scores do not change.
DEPTH = {'0139_w044': -34, '0139_w045': -34}
MRG_TAG = 'mrg20736-1um'
VILLA_SHA = v30.VILLA_SHA
UM_B0 = 1.129
UM_PX = 2 * UM_B0                        # 2.258 um render px and layer step (level 1)
CELL_A, CELL_B = 20, 10                  # px per mesh cell: A canvas (labels), B L1 canvas (renders, mrg map)
# step 0 / step 1: V-030's constants
CROPS, CROP_A, CROP_B, P_LAYERS, NCC_MIN, NEED = v30.CROPS, v30.CROP_A, v30.CROP_B, v30.P_LAYERS, v30.NCC_MIN, v30.NEED
SMOOTH = v30.SMOOTH
RES_MED_MAX, INK_FRAC_MIN, SCALE_TOL = v30.RES_MED_MAX, v30.INK_FRAC_MIN, v30.SCALE_TOL
SUB_MARGIN = 12                          # cells kept around a crop / window in written sub-meshes (render_crop uses 8)
ROI_PAD = 2                              # cells around the usable part of mesh B
SCOPE_VOX = 20.0                         # an A vertex is "in scope" if a usable B vertex lies within one A cell (3D)
# windows: V-030 held-out windows (Amendment 2)
PATCH, MAX_TEST, LAYERS, DS = v30.PATCH, v30.MAX_TEST, v30.LAYERS, v30.DS
# rule (README "Pre-registered rule")
WIN_FRAC, MEAN_DIFF_MIN, N_MIN, AUC_OK = 0.70, 0.0, 10, 0.75
EXTRA_PER_BAND = 2                       # descriptive published-mesh arm: the 2 densest windows per band (PHerc1667)
BUDGET_S = 4 * 3600
EST = {'render_min': 1.0, 'score_min': 0.25}         # per window; V-030 measured 0.75 min per 400 x 800 render
DATA = os.environ.get('V031_DATA', 'out/v031')
RESULTS = os.environ.get('V031_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v031-canonical-vs-mrg-1um'))
RUN = os.environ.get('PUZZLE_RUN_DIR', '.')
LOWDISK = bool(os.environ.get('V031_LOWDISK'))
v30.DATA = DATA


# --------------------------------------------------------------------------------------------------------- basics
def fetch(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except Exception:                                              # noqa: BLE001  (resets, IncompleteRead)
            if t == tries - 1:
                raise
            time.sleep(2 ** (t + 1))


def ls(prefix):
    """S3 listing (one level): (sub-prefixes, [(key, size)])."""
    import re
    ps, ks, tok = [], [], None
    while True:
        q = {'list-type': '2', 'prefix': prefix, 'delimiter': '/'}
        if tok:
            q['continuation-token'] = tok
        x = fetch(B + '?' + urllib.parse.urlencode(q)).decode()
        ps += [p for p in re.findall(r'<Prefix>([^<]*)</Prefix>', x) if p != prefix]
        ks += [(m.group(1), int(m.group(2))) for m in re.finditer(r'<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>', x)]
        m = re.search(r'<NextContinuationToken>([^<]*)</NextContinuationToken>', x)
        if not m:
            return ps, ks
        tok = m.group(1)


class RangeFile(io.RawIOBase):
    """Read-only file over HTTP range requests (tiff headers and tiles without downloading the file)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60)
        self.size = int(r.headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        rq = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}'})
        for t in range(5):
            try:
                d = urllib.request.urlopen(rq, timeout=120).read()
                break
            except Exception:                                          # noqa: BLE001
                if t == 4:
                    raise
                time.sleep(2 ** (t + 1))
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def tif_shape(url):
    import tifffile
    with tifffile.TiffFile(RangeFile(url)) as t:
        p = t.pages[0]
        return list(p.shape), str(p.dtype), int(p.compression), bool(p.is_tiled)


def segs_json():
    s = jload(os.path.join(RESULTS, 'segments.json'))
    if s is None:
        raise SystemExit('run resolve first (results/.../segments.json)')
    return s


def select(args):
    """Segments named by short name or scroll; all when none."""
    S = segs_json()['segments']
    if not args:
        return list(S)
    out = []
    for a in args:
        out += [k for k, v in S.items() if a in (k, v['scroll'])]
    return out


def sdir(name, *p):
    d = os.path.join(DATA, name, *p)
    os.makedirs(d if not p or '.' not in p[-1] else os.path.dirname(d), exist_ok=True)
    return d


def timing(stage, seconds):
    tp = os.path.join(DATA, 'timing.json')
    t = jload(tp, {})
    t[stage] = round(t.get(stage, 0) + seconds, 1)
    jsave(t, tp)


def timing_total():
    return {k: v for k, v in jload(os.path.join(DATA, 'timing.json'), {}).items() if k.startswith(('render', 'score'))}


# --------------------------------------------------------------------------------------------------------- resolve
def resolve(only=None):
    """Exact bucket paths for every registered segment (listing, metadata and tiff headers only). With scroll names,
    only those scrolls are (re)listed and the other entries of segments.json are kept."""
    out = jload(os.path.join(RESULTS, 'segments.json')) if only else None
    out = out or {'bucket': B, 'segments': {}, 'scrolls': {}}
    stamp = time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())
    if only:
        out.setdefault('listed_utc_by_scroll', {}).update({sc: stamp for sc in only})
    else:
        out['listed_utc'] = stamp
    for scroll, cfg in SCROLLS.items():
        if only and scroll not in only:
            continue
        vb = cfg['vol_B']
        tr = json.loads(fetch(B + vb + '/transform.json'))
        out['scrolls'][scroll] = {'vol_A': cfg['vol_A'], 'vol_B': vb, 'transform': vb + '/transform.json',
                                  'transform_fixed_volume': tr.get('fixed_volume'),
                                  'n_landmarks': len(tr['fixed_landmarks']),
                                  'vol_A_L0_zyx': json.loads(fetch(B + cfg['vol_A'] + '/0/.zarray'))['shape'],
                                  'vol_B_L0_zyx': json.loads(fetch(B + vb + '/0/.zarray'))['shape'],
                                  'um_A': cfg['um_A'], 'refit_required': cfg['refit_required']}
        for seg in cfg['segments']:
            base = f'{scroll}/segments/{seg}/'
            short = cfg['short'] + '_' + (seg.split('-')[1].split('_')[0] if '-' in seg else seg)
            meshes, _ = ls(base + 'mesh/')
            mA = [p for p in meshes if f"-on-{cfg['scan_A']}-" in p]
            mB = [p for p in meshes if f"-on-{cfg['scan_B']}-1.129um" in p]
            labroots, _ = ls(base + 'ink-labels/')
            lr = [p for p in labroots if cfg['scan_A'] in p]
            r = {'scroll': scroll, 'segment': seg, 'status': 'ok'}
            if len(mA) != 1 or len(mB) != 1 or len(lr) != 1:
                r['status'] = f'missing: mesh_A {len(mA)}, mesh_B {len(mB)}, label roots {len(lr)}'
                out['segments'][short] = r
                log('RESOLVE', short, r['status'])
                continue
            dates, _ = ls(lr[0])
            date = sorted(dates)[-1]
            lab = date + 'inklabels.zarr/0/'
            lmeta = json.loads(fetch(B + lab + 'zarr.json'))
            _, dets = ls(base + 'ink-detection/')
            mrg = [k for k, _ in dets if MRG_TAG in k and cfg['scan_B'] in k]
            svs, _ = ls(base + 'surface-volumes/')
            sv = [p for p in svs if cfg['scan_B'] in p and p.rstrip('/').endswith('-L1.zarr')]
            r.update({'mesh_A': mA[0], 'mesh_B': mB[0], 'labels': lab, 'label_dates': [d.split('/')[-2] for d in dates],
                      'label_shape': lmeta['shape'], 'mrg': mrg[0] if len(mrg) == 1 else None,
                      'sv_B': sv[0] if len(sv) == 1 else None})
            r['mesh_A_shape'] = tif_shape(B + mA[0] + 'x.tif')[0]
            r['mesh_B_shape'] = tif_shape(B + mB[0] + 'x.tif')[0]
            r['mesh_B_bbox'] = json.loads(fetch(B + mB[0] + 'meta.json')).get('bbox')
            if r['mrg']:
                r['mrg_size'] = dict(dets)[r['mrg']]
                r['mrg_shape'], r['mrg_dtype'], r['mrg_compression'], r['mrg_tiled'] = tif_shape(B + r['mrg'])
            canvas = [n * CELL_B for n in r['mesh_B_shape']]
            if not r['mrg'] or not r['sv_B']:
                r['status'] = f"missing: mrg {bool(r['mrg'])}, L1 surface volume {bool(r['sv_B'])}"
            elif r['mrg_shape'] != canvas or r['mrg_dtype'] != 'uint8':
                r['status'] = f"mrg map {r['mrg_shape']} {r['mrg_dtype']} is not the L1 canvas {canvas} uint8"
            r['canvas_B'] = canvas
            out['segments'][short] = r
            log('RESOLVE', short, r['status'], 'mesh_B', r['mesh_B_shape'], 'labels', r['label_shape'])
    jsave(out, os.path.join(RESULTS, 'segments.json'))
    return out


# --------------------------------------------------------------------------------------------------------- data
def transforms(scroll):
    """M_pub (1.129 -> labelled-scan voxels) and the least-squares refit to its landmarks (V-014), with residuals."""
    cfg = segs_json()['scrolls'][scroll]
    d = json.loads(fetch(B + cfg['transform']))
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
            'rms_px_A': {'pub': rms(Mpub), 'lsq': rms(Mlsq)}, 'loo_px_A_lsq': loo,
            'singular_values': {'pub': np.linalg.svd(Mpub[:, :3])[1].tolist(), 'lsq': np.linalg.svd(Mlsq[:, :3])[1].tolist(),
                                'physical': UM_B0 / cfg['um_A']},
            'M_pub': Mpub.tolist(), 'M_lsq': Mlsq.tolist()}
    return Mpub, Mlsq, info


def mesh(url_dir, dtype=np.float64):
    """(H, W, 3) xyz in that scan's level-0 voxels and the valid mask, read into memory (never written whole)."""
    import tifffile
    xyz = np.stack([tifffile.imread(io.BytesIO(fetch(B + url_dir + c + '.tif'))).astype(dtype) for c in 'xyz'], -1)
    return xyz, (xyz > -0.5).all(-1)


class LabCrop:
    """Labels of a sub-rectangle of the A canvas, indexed in full-canvas coordinates (callers clip to the crop)."""

    def __init__(self, a, oy, ox, shape):
        self.a, self.oy, self.ox, self.shape = a, oy, ox, shape

    def __getitem__(self, idx):
        li, lj = idx
        return self.a[np.clip(li - self.oy, 0, self.a.shape[0] - 1), np.clip(lj - self.ox, 0, self.a.shape[1] - 1)]


class Labels:
    """Level-0 human labels on the mesh-A canvas: the published zarr v3 shards copied locally (compressed, small),
    read by region so the whole canvas is never held in memory."""

    def __init__(self, s, name):
        import zarr
        meta = json.loads(fetch(B + s['labels'] + 'zarr.json'))
        d = os.path.join(DATA, name, 'labels.zarr')
        if not os.path.exists(os.path.join(d, 'zarr.json')):
            grid = [-(-n // c) for n, c in zip(meta['shape'], meta['chunk_grid']['configuration']['chunk_shape'])]
            for i in range(grid[0]):
                for j in range(grid[1]):
                    try:
                        raw = fetch(f"{B}{s['labels']}c/{i}/{j}")
                    except urllib.error.HTTPError as e:
                        if e.code in (403, 404):
                            continue
                        raise
                    os.makedirs(os.path.join(d, 'c', str(i)), exist_ok=True)
                    open(os.path.join(d, 'c', str(i), str(j)), 'wb').write(raw)
            json.dump(meta, open(os.path.join(d, 'zarr.json'), 'w'))
        self.z = zarr.open_array(d, mode='r')
        self.shape = tuple(self.z.shape)
        assert list(self.shape) == s['label_shape'], (self.shape, s['label_shape'])

    def rows(self, r0, r1, c1):
        return np.asarray(self.z[r0:r1, :c1]) > 0

    def crop(self, U, V, C):
        """The labels every carried (u, v) of this block can reach (bounding box + 1 cell)."""
        if not C.any():
            return LabCrop(np.zeros((1, 1), bool), 0, 0, self.shape)
        u, v = U[C], V[C]
        y0, y1 = max(0, int(np.floor(u.min())) * CELL_A - CELL_A), min(self.shape[0], int(np.ceil(u.max())) * CELL_A + CELL_A + 1)
        x0, x1 = max(0, int(np.floor(v.min())) * CELL_A - CELL_A), min(self.shape[1], int(np.ceil(v.max())) * CELL_A + CELL_A + 1)
        return LabCrop(np.asarray(self.z[y0:y1, x0:x1]) > 0, y0, x0, self.shape)


class ArrLabels(Labels):
    def __init__(self, a):                                         # selftest: an in-memory label array
        self.z, self.shape = a, a.shape


def a_cells(lab, A, strip=64):
    Hc, Wc = min(lab.shape[0] // CELL_A, A.shape[0] - 1), min(lab.shape[1] // CELL_A, A.shape[1] - 1)
    out = np.zeros((Hc, Wc), np.float32)
    for r in range(0, Hc, strip):
        r1 = min(Hc, r + strip)
        out[r:r1] = lab.rows(r * CELL_A, r1 * CELL_A, Wc * CELL_A).reshape(r1 - r, CELL_A, Wc, CELL_A).mean(
            (1, 3), dtype=np.float32)
    return out


def coverage(s, shape):
    """B vertices where the 1.129 um scan has data: the organisers' L1 surface volume at level 5 (32 L1 px per pixel),
    any layer non-zero (V-030 Amendment 1)."""
    url = B + s['sv_B'] + '5/'
    meta = json.loads(fetch(url + '.zarray'))
    assert meta['compressor'] is None and meta['dtype'] == '|u1' and meta['chunks'][0] == meta['shape'][0], meta
    nz, cy, cx = meta['chunks']
    mip = np.zeros(meta['shape'][1:], bool)
    for iy in range(-(-meta['shape'][1] // cy)):
        for ix in range(-(-meta['shape'][2] // cx)):
            try:
                raw = fetch(f'{url}0/{iy}/{ix}')
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    continue
                raise
            c = (np.frombuffer(raw, np.uint8).reshape(nz, cy, cx) > 0).any(0)
            h, w = mip[iy * cy:(iy + 1) * cy, ix * cx:(ix + 1) * cx].shape
            mip[iy * cy:iy * cy + h, ix * cx:ix * cx + w] = c[:h, :w]
    ii = np.clip(np.arange(shape[0]) * CELL_B // 32, 0, mip.shape[0] - 1)
    jj = np.clip(np.arange(shape[1]) * CELL_B // 32, 0, mip.shape[1] - 1)
    return mip[np.ix_(ii, jj)]


def vol_shape(vol):
    return json.loads(fetch(f'{B}{vol}/0/.zarray'))['shape']           # z, y, x


def inside(xyz, zyx):
    zs, ys, xs = zyx
    return (xyz >= 0).all(-1) & (xyz <= np.array([xs - 1, ys - 1, zs - 1])).all(-1)


def write_sub(xyz, valid, rows, cols, out, scale, margin=SUB_MARGIN):
    """Write mesh rows/cols (+ margin) as a tifxyz; returns (dir, rows, cols) relative to it."""
    R0, C0 = max(0, rows[0] - margin), max(0, cols[0] - margin)
    R1, C1 = min(valid.shape[0], rows[1] + margin + 1), min(valid.shape[1], cols[1] + margin + 1)
    v30.write_mesh(xyz[R0:R1, C0:C1], valid[R0:R1, C0:C1], out, scale)
    return out, (rows[0] - R0, rows[1] - R0), (cols[0] - C0, cols[1] - C0)


def render(mdir, vol, out, rows, cols, layers, level):
    """V-030's render_crop call; the chunk cache is removed after every render (crops and windows do not overlap)."""
    cache = os.path.join(DATA, 'cache')
    try:
        return v30.render(mdir, B + vol, out, rows, cols, layers, level, cache)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def step2_xyz(scroll, Bm, Mpub, Mlsq, which):
    if which == 'B':
        return Bm
    return apply(inverse(Mlsq), apply(Mpub, Bm))                        # B' = M_lsq^-1 M_pub B (V-014 / V-030)


# --------------------------------------------------------------------------------------------------------- step 0
def step0_one(name, force=False):
    s = segs_json()['segments'][name]
    sc = segs_json()['scrolls'][s['scroll']]
    out_p = os.path.join(RESULTS, 'step0', name + '.json')
    if os.path.exists(out_p) and not force:
        r = json.load(open(out_p))
        log('STEP0', name, r['line'], '(committed; rerun with --force to recompute)')
        return r
    if s['status'] != 'ok':
        r = {'segment': name, 'verdict': 'DROPPED', 'line': f'DROPPED: {s["status"]}', 'mesh_for_step2': None}
        jsave(r, out_p)
        log('STEP0', name, r['line'])
        return r
    t0 = time.time()
    Mpub, Mlsq, tinfo = transforms(s['scroll'])
    zyx = vol_shape(sc['vol_B'])
    A, vA = mesh(s['mesh_A'])
    cells = a_cells(Labels(s, name), A)
    Bm, vB = mesh(s['mesh_B'], np.float32)
    Bm = Bm.astype(np.float64)
    first = 'Bfit' if sc['refit_required'] else 'B'
    Bs = step2_xyz(s['scroll'], Bm, Mpub, Mlsq, first)
    cov = coverage(s, vB.shape)
    usable = vB & inside(Bs, zyx) & cov
    del Bs
    from scipy.spatial import cKDTree
    idxB = np.argwhere(usable)
    if len(idxB) == 0:
        r = {'segment': name, 'verdict': 'PLACEMENT_FAIL', 'mesh_for_step2': None,
             'line': 'PLACEMENT_FAIL: no mesh-B vertex lies inside the 1.129 um volume where it has data', 'transform': tinfo}
        jsave(r, out_p); log('STEP0', name, r['line']); return r
    treeB = cKDTree(apply(Mpub, Bm[usable]))
    dA = np.full(vA.shape, np.inf)
    dA[vA] = treeB.query(A[vA], workers=-1, distance_upper_bound=SCOPE_VOX * 2)[0]
    inA = vA & (dA <= SCOPE_VOX)
    Hc, Wc = cells.shape
    boxes = v30.greedy_boxes(cells, v30.cell_ok(inA)[:Hc, :Wc], CROP_A, CROPS)
    scope_mm2 = float((cells * v30.cell_ok(inA)[:Hc, :Wc]).sum() * (CELL_A * sc['um_A'] * 1e-3) ** 2)
    if not boxes:                                                      # nothing labelled where B has data
        r = {'segment': name, 'verdict': 'DROPPED', 'mesh_for_step2': None, 'transform': tinfo,
             'usable_B_vertices': int(usable.sum()), 'in_scope_ink_mm2_A': round(scope_mm2, 3),
             'line': f'DROPPED: no labelled ink where the 1.129 um scan has data along the mesh (in-scope ink '
                     f'{scope_mm2:.2f} mm2 of {float(cells.sum() * (CELL_A * sc["um_A"] * 1e-3) ** 2):.2f}); '
                     'no crop to test placement'}
        jsave(r, out_p); log('STEP0', name, r['line']); return r
    meshes = {'ref': (apply(Mpub, Bm), 0.1, sc['vol_A'], 0), 'B': (Bm, 0.05, sc['vol_B'], 1),
              'Bfit': (apply(inverse(Mlsq), apply(Mpub, Bm)), 0.05, sc['vol_B'], 1)}
    crops = []
    arms = ['B', 'Bfit'] if sc['refit_required'] else ['B']
    for k, (rc, dn) in enumerate(boxes):
        r0, r1, c0, c1 = rc
        d, j = treeB.query(A[(r0 + r1) // 2, (c0 + c1) // 2])
        bi, bj = (int(x) for x in idxB[j])
        hb, wb = CROP_B
        br, bc = (bi - hb // 2, bi + hb // 2), (bj - wb // 2, bj + wb // 2)
        c = {'crop': k, 'cells_A': rc, 'density': round(dn, 4), 'centre_B': [bi, bj], 'nn_dist_vox_A': round(float(d), 2),
             'rows_B': list(br), 'cols_B': list(bc)}
        inb = br[0] >= 0 and bc[0] >= 0 and br[1] < vB.shape[0] and bc[1] < vB.shape[1]
        if not inb or vB[br[0]:br[1] + 1, bc[0]:bc[1] + 1].mean() < 0.9:
            c['status'] = 'UNKNOWN: B crop under 90% valid vertices'
            crops.append(c); log('CROP', name, json.dumps(c)); continue
        st = {}
        for arm in ['ref'] + arms:
            xyz, scale, vol, level = meshes[arm]
            md, rr, cc = write_sub(np.where(vB[..., None], xyz, -1), vB, br, bc, sdir(name, 'step0', f'mesh_{arm}{k}'), scale)
            st[arm] = render(md, vol, sdir(name, 'step0', f'crop{k}_{arm}'), rr, cc, P_LAYERS, level)
            shutil.rmtree(md, ignore_errors=True)
        c['valid_frac'] = {a: round(float((x > 0).any(0).mean()), 3) for a, x in st.items()}
        if min(c['valid_frac'].values()) < 0.5:
            c['status'] = 'UNKNOWN: under 50% CT-valid pixels'
        else:
            for arm in arms:
                c['pub' if arm == 'B' else 'refit'] = v30.placement_stat(st['ref'], st[arm])
            c['status'] = 'ok'
        crops.append(c)
        log('CROP', name, json.dumps(c))
    passed = lambda arm: sum(1 for c in crops if arm in c and c[arm]['ncc0'] >= NCC_MIN)      # noqa: E731
    n_pub = passed('pub')
    if sc['refit_required']:
        n_fit = passed('refit')
        better = sum(1 for c in crops if 'refit' in c and 'pub' in c and c['refit']['ncc0'] > c['pub']['ncc0'])
        verdict = 'PLACEMENT_REFIT' if n_fit >= NEED else 'PLACEMENT_FAIL'
        line = (f'{verdict}: refit (M_lsq, required for {s["scroll"]}) NCC0 >= {NCC_MIN} on {n_fit} of {len(crops)} crops '
                f'(need {NEED}); published NCC0 >= {NCC_MIN} on {n_pub}, refit > published on {better} (descriptive)')
    else:
        verdict = 'PLACEMENT_OK' if n_pub >= NEED else None
        n_fit = better = None
        if verdict is None:                                            # V-030: refit only when the published fails
            for k, c in enumerate(crops):
                if c['status'] != 'ok':
                    continue
                md, rr, cc = write_sub(np.where(vB[..., None], meshes['ref'][0], -1), vB, c['rows_B'], c['cols_B'],
                                       sdir(name, 'step0', f'mesh_ref{k}'), 0.1)
                R = render(md, sc['vol_A'], sdir(name, 'step0', f'crop{k}_ref'), rr, cc, P_LAYERS, 0)
                md2, rr, cc = write_sub(np.where(vB[..., None], meshes['Bfit'][0], -1), vB, c['rows_B'], c['cols_B'],
                                        sdir(name, 'step0', f'mesh_Bfit{k}'), 0.05)
                X = render(md2, sc['vol_B'], sdir(name, 'step0', f'crop{k}_Bfit'), rr, cc, P_LAYERS, 1)
                shutil.rmtree(md, ignore_errors=True); shutil.rmtree(md2, ignore_errors=True)
                c['refit'] = v30.placement_stat(R, X)
                log('CROP refit', name, json.dumps(c))
            n_fit = passed('refit')
            better = sum(1 for c in crops if 'refit' in c and c['refit']['ncc0'] > c['pub']['ncc0'])
            verdict = 'PLACEMENT_REFIT' if n_fit >= NEED and better >= NEED else 'PLACEMENT_FAIL'
        line = (f'{verdict}: published transform NCC0 >= {NCC_MIN} on {n_pub} of {len(crops)} crops (need {NEED})'
                + ('' if n_fit is None else f'; refit (M_lsq) NCC0 >= {NCC_MIN} on {n_fit}, refit > published on {better}'))
    shutil.rmtree(os.path.join(DATA, name, 'step0'), ignore_errors=True)
    res = {'segment': name, 'verdict': verdict, 'line': line,
           'mesh_for_step2': {'PLACEMENT_OK': 'B', 'PLACEMENT_REFIT': 'Bfit'}.get(verdict),
           'transform': tinfo, 'crops': crops, 'in_scope_ink_mm2_A': round(scope_mm2, 3),
           'usable_B_vertices': int(usable.sum()),
           'rule': dict(CROPS=CROPS, CROP_A=CROP_A, CROP_B=CROP_B, LAYERS=P_LAYERS, NCC_MIN=NCC_MIN, NEED=NEED,
                        SMOOTH=SMOOTH, SCOPE_VOX=SCOPE_VOX, refit_required=sc['refit_required']),
           'seconds': round(time.time() - t0, 1)}
    jsave(res, out_p, os.path.join(RUN, 'payload', f'v031-step0-{name}.json'))
    log('STEP0', name, line)
    return res


def gate(name, stage):
    s0 = jload(os.path.join(RESULTS, 'step0', name + '.json'))
    if s0 is None:
        raise SystemExit(f'{stage} {name}: run step0 first')
    if s0['mesh_for_step2'] is None:
        return None, f'STOPPED_AT_PLACEMENT ({s0["line"]})'
    if stage != 'carry':
        c = jload(os.path.join(RESULTS, 'carry', name + '.json'))
        if c is None:
            raise SystemExit(f'{stage} {name}: run carry first')
        if c['verdict'] != 'CARRY_OK':
            return None, f'STOPPED_AT_CARRY ({c["line"]})'
    return s0, None


# --------------------------------------------------------------------------------------------------------- step 1
def canvas_cells(U, V, C, lab, strip=32, colstrip=1024):
    """Per B cell: carried-ink fraction of its 10 x 10 L1 pixels and whether all of them are carried. The pixel
    mapping is V-030's (render_tifxyz.upsample of (u, v), label at (round(20 u), round(20 v))), done in tiles (one
    extra vertex row and column each, so it equals the whole-canvas mapping) and never holds the full canvas.
    lab: a Labels (cropped per tile)."""
    v30.src()
    H, W = C.shape[0] - 1, C.shape[1] - 1
    ink = np.zeros((H, W), np.float32)
    full = np.zeros((H, W), bool)
    for a in range(0, H, strip):
        b = min(H, a + strip)
        for c in range(0, W, colstrip):
            d = min(W, c + colstrip)
            Ua, Va, Ca = U[a:b + 1, c:d + 1], V[a:b + 1, c:d + 1], C[a:b + 1, c:d + 1]
            lab_p, ok = pixels(Ua, Va, Ca, lab.crop(Ua, Va, Ca))
            n, m = (b - a) * CELL_B, (d - c) * CELL_B
            ink[a:b, c:d] = lab_p[:n, :m].reshape(b - a, CELL_B, d - c, CELL_B).mean((1, 3), dtype=np.float32)
            full[a:b, c:d] = ok[:n, :m].reshape(b - a, CELL_B, d - c, CELL_B).all((1, 3))
    return ink, full


def pixels(U, V, C, lab):
    """Carried labels and carried mask on the L1 pixels of a block of B vertices (corner-aligned, 10 px per cell)."""
    from render_tifxyz import upsample
    Up, ok = upsample(U, C, CELL_B, CELL_B)
    Vp, _ = upsample(V, C, CELL_B, CELL_B)
    li, lj = np.rint(Up * CELL_A).astype(np.int64), np.rint(Vp * CELL_A).astype(np.int64)
    ok &= (li >= 0) & (lj >= 0) & (li < lab.shape[0]) & (lj < lab.shape[1])
    lab_p = np.where(ok, lab[np.clip(li, 0, lab.shape[0] - 1), np.clip(lj, 0, lab.shape[1] - 1)], False)
    return lab_p, ok


def carry_planes(A, vA, Bm, vB, M, rows=128):
    """v30.carry_planes, the same per-vertex maths (v30.carry_vertices) in row chunks of mesh B, so memory stays
    bounded on meshes of tens of millions of vertices. Selftest: equal to v30.carry_planes."""
    from scipy.spatial import cKDTree
    idxA = np.argwhere(vA)
    tree = cKDTree(A[vA])
    Ju, Jv = v30.jacobians(A, vA)
    U = np.full(vB.shape, -1.0, np.float32); V = np.full(vB.shape, -1.0, np.float32)
    C = np.zeros(vB.shape, bool)
    res_all = []
    for r0 in range(0, vB.shape[0], rows):
        sl = slice(r0, min(vB.shape[0], r0 + rows))
        vb = vB[sl]
        if not vb.any():
            continue
        q = apply(M, Bm[sl][vb].astype(np.float64))
        _, k = tree.query(q, workers=-1)
        i, j = idxA[k].T
        J = np.stack([Ju[i, j], Jv[i, j]], -1)
        r = q - A[i, j]
        JtJ = np.einsum('nki,nkj->nij', J, J)
        Jtr = np.einsum('nki,nk->ni', J, r)
        good = np.isfinite(JtJ).all((1, 2)) & (np.abs(np.linalg.det(np.nan_to_num(JtJ))) > 1e-9)
        delta = np.full((len(q), 2), np.nan)
        delta[good] = np.linalg.solve(JtJ[good], Jtr[good][..., None])[..., 0]
        res = np.full(len(q), np.nan)
        res[good] = np.linalg.norm(r[good] - np.einsum('nki,ni->nk', J[good], delta[good]), axis=1)
        u, v = i + delta[:, 0], j + delta[:, 1]
        ok = np.isfinite(res) & (np.abs(delta) <= v30.DELTA_MAX).all(1) & (res <= v30.RES_MAX)
        ok &= (u >= 0) & (v >= 0) & (u <= A.shape[0] - 1) & (v <= A.shape[1] - 1)
        Uc, Vc, Cc = U[sl], V[sl], C[sl]
        Uc[vb], Vc[vb], Cc[vb] = np.where(ok, u, -1), np.where(ok, v, -1), ok
        res_all.append(res[ok])
    return U, V, C, np.concatenate(res_all) if res_all else np.zeros(0)


def roi_of(usable):
    rr, cc = np.nonzero(usable.any(1))[0], np.nonzero(usable.any(0))[0]
    H, W = usable.shape
    return [max(0, int(rr[0]) - ROI_PAD), min(H, int(rr[-1]) + ROI_PAD + 1), max(0, int(cc[0]) - ROI_PAD),
            min(W, int(cc[-1]) + ROI_PAD + 1)]


def carry_compute(name):
    """Everything later stages need, recomputed from the bucket: ROI, carried (u, v), per-cell ink, usable mask, z."""
    s = segs_json()['segments'][name]
    sc = segs_json()['scrolls'][s['scroll']]
    s0 = jload(os.path.join(RESULTS, 'step0', name + '.json'))
    Mpub, Mlsq, _ = transforms(s['scroll'])
    zyx = vol_shape(sc['vol_B'])
    A, vA = mesh(s['mesh_A'])
    Bm, vB = mesh(s['mesh_B'], np.float32)
    Bm = Bm.astype(np.float64)
    Bs = step2_xyz(s['scroll'], Bm, Mpub, Mlsq, s0['mesh_for_step2'])
    usable = vB & inside(Bs, zyx) & coverage(s, vB.shape)
    roi = roi_of(usable)
    g = (slice(roi[0], roi[1]), slice(roi[2], roi[3]))
    Br, vBr, Bsr, ur = Bm[g].copy(), vB[g].copy(), Bs[g].copy(), usable[g].copy()
    from scipy.spatial import cKDTree
    treeB = cKDTree(apply(Mpub, Bm[usable]))
    del Bm, Bs, vB, usable
    lab = Labels(s, name)
    # in-scope labelled ink on the A canvas (the denominator of carry rule 2): A cells whose 4 vertices lie within one
    # A cell (3D) of a usable B vertex
    dA = np.full(vA.shape, np.inf)
    dA[vA] = treeB.query(A[vA], workers=-1, distance_upper_bound=SCOPE_VOX * 2)[0]
    cellsA = a_cells(lab, A)
    Hc, Wc = cellsA.shape
    src_mm2 = float((cellsA * v30.cell_ok(vA & (dA <= SCOPE_VOX))[:Hc, :Wc]).sum() * (CELL_A * sc['um_A'] * 1e-3) ** 2)
    del dA, cellsA, treeB
    U, V, C, res = carry_planes(A, vA, Br, vBr, Mpub)
    sv = v30.local_scale(U, V, C)
    ink, full = canvas_cells(U, V, C, lab)
    del lab, A
    return {'roi': roi, 'U': U, 'V': V, 'C': C, 'res': res, 'sv': sv, 'ink': ink, 'full': full, 'usable': ur,
            'z': Bsr[..., 2].astype(np.float32), 'src_mm2': src_mm2, 'valid': vBr}


_CARRY = {}


def carry_one(name):
    s0, stop = gate(name, 'carry')
    if stop:
        log('CARRY', name, 'VERDICT', stop)
        return None
    t0 = time.time()
    k = carry_compute(name)
    cd = sdir(name, 'carry')
    if LOWDISK:                                         # cloud: plan reads them from memory in the same process
        _CARRY.clear()
        _CARRY[name] = k
    else:
        for f in ('U', 'V', 'ink', 'z'):
            np.save(os.path.join(cd, f + '.npy'), k[f])
        for f in ('C', 'full', 'usable', 'valid'):
            v30.save_bits(os.path.join(cd, f + '.npy'), k[f])
        json.dump(k['roi'], open(os.path.join(cd, 'roi.json'), 'w'))
    um = segs_json()['scrolls'][segs_json()['segments'][name]['scroll']]['um_A']
    scale = UM_B0 / um
    H, W = k['ink'].shape
    usable_cells = v30.cell_ok(k['usable'])[:H, :W] & k['full']
    car_mm2 = float((k['ink'] * usable_cells).sum() * CELL_B ** 2 * (UM_PX * 1e-3) ** 2)
    med_sv = [float(np.median(k['sv'][:, 0])), float(np.median(k['sv'][:, 1]))] if len(k['sv']) else [0.0, 0.0]
    med_res = float(np.median(k['res'])) if len(k['res']) else float('inf')
    src = k['src_mm2']
    rules = {
        'median_residual_vox': (med_res, med_res <= RES_MED_MAX),
        'carried_ink_frac': (car_mm2 / src if src else 0.0, src > 0 and car_mm2 / src >= INK_FRAC_MIN),
        'median_local_scale': (med_sv, all(abs(x / scale - 1) <= SCALE_TOL for x in med_sv)),
    }
    verdict = 'CARRY_OK' if all(ok for _, ok in rules.values()) else 'CARRY_FAIL'
    out = {'segment': name, 'verdict': verdict, 'transform': 'M_pub', 'mesh_for_step2': s0['mesh_for_step2'],
           'roi_cells_B': k['roi'], 'rules': {a: {'value': v, 'ok': ok} for a, (v, ok) in rules.items()},
           'thresholds': dict(DELTA_MAX=v30.DELTA_MAX, RES_MAX=v30.RES_MAX, RES_MED_MAX=RES_MED_MAX,
                              INK_FRAC_MIN=INK_FRAC_MIN, SCALE=scale, SCALE_TOL=SCALE_TOL, SCOPE_VOX=SCOPE_VOX),
           'residual_vox_p90': float(np.percentile(k['res'], 90)) if len(k['res']) else None,
           'vertices': {'roi_valid': int(k['valid'].sum()), 'carried': int(k['C'].sum()), 'usable': int(k['usable'].sum())},
           'cells': {'usable_and_carried': int(usable_cells.sum())},
           'ink_mm2': {'in_scope_A': src, 'carried_usable_B': car_mm2},
           'seconds': round(time.time() - t0, 1)}
    out['line'] = (f"{verdict}: median normal residual {med_res:.2f} vox (<= {RES_MED_MAX}), carried ink {car_mm2:.2f} of "
                   f"{src:.2f} mm2 in scope ({rules['carried_ink_frac'][0]:.3f}, >= {INK_FRAC_MIN}), median local scale "
                   f"{med_sv[0]:.4f}/{med_sv[1]:.4f} (target {scale:.4f} +-{SCALE_TOL:.0%})")
    ref_p = os.path.join(RESULTS, 'carry', name + '.json')
    ref = jload(ref_p)
    if ref is not None:
        same = (ref['verdict'] == verdict and ref['roi_cells_B'] == k['roi']
                and abs(ref['vertices']['carried'] - out['vertices']['carried']) <= 1e-3 * max(1, ref['vertices']['carried'])
                and abs(ref['ink_mm2']['carried_usable_B'] - car_mm2) <= 1e-3 * max(1.0, car_mm2))
        log('CARRY', name, 'SAME' if same else 'DIFFERENT', 'as', ref_p)
        jsave(out, os.path.join(RUN, 'payload', f'v031-carry-{name}.json'))
    else:
        jsave(out, ref_p, os.path.join(RUN, 'payload', f'v031-carry-{name}.json'))
    timing('carry', time.time() - t0)
    log('CARRY', name, out['line'])
    if verdict == 'CARRY_FAIL':
        log('CARRY', name, 'VERDICT STOPPED_AT_CARRY')
    return out


# --------------------------------------------------------------------------------------------------------- plan
def plan_one(name):
    """V-030 held-out windows on one segment: 3 z bands (ink terciles of the step-2 mesh z), up to MAX_TEST per band,
    greedy by carried-ink density over cells that are usable (in volume, with 1.129 um data) and fully carried."""
    s0, stop = gate(name, 'plan')
    if stop:
        return {'segment': name, 'stopped': stop, 'windows': []}
    if name in _CARRY:
        k = _CARRY[name]
        roi, ink, z, full, usable = k['roi'], k['ink'], k['z'], k['full'], k['usable']
    else:
        cd = os.path.join(DATA, name, 'carry')
        roi = json.load(open(os.path.join(cd, 'roi.json')))
        ink, z = np.load(os.path.join(cd, 'ink.npy')), np.load(os.path.join(cd, 'z.npy'))
        full, usable = v30.load_bits(os.path.join(cd, 'full.npy')), v30.load_bits(os.path.join(cd, 'usable.npy'))
    H, W = ink.shape
    inside_c = v30.cell_ok(usable)[:H, :W] & full
    if not inside_c.any() or not (ink * inside_c).sum():
        return {'segment': name, 'stopped': 'NO_WINDOWS (no carried ink where the 1.129 um scan has data)', 'windows': []}
    z4 = np.stack([z[:-1, :-1], z[1:, :-1], z[:-1, 1:], z[1:, 1:]], -1)[:H, :W]
    zc = z4.min(-1)
    w = ink * inside_c
    order = np.argsort(zc[inside_c])
    cw = np.cumsum(w[inside_c][order]) / w[inside_c].sum()
    edges = tuple(int(round(float(zc[inside_c][order][min(np.searchsorted(cw, q), order.size - 1)]) / 10) * 10)
                  for q in (1 / 3, 2 / 3))
    v20.PATCH_H, v20.PATCH_W = PATCH
    v20.BAND_EDGES, v20.MAX_PER, v20.REL, v20.MIN_DENS, v20.MIN_VALID = edges, MAX_TEST, 0.2, 0.02, 0.5
    in_band = inside_c & (v20.band_of(z4.min(-1)) == v20.band_of(z4.max(-1)))
    px_mm2 = (UM_PX * 1e-3) ** 2
    wins = []
    for b in range(3):
        for rc, dn in v20.choose(ink, in_band, zc, b):
            r0, r1, c0, c1 = rc
            zz = z[r0:r1 + 1, c0:c1 + 1][usable[r0:r1 + 1, c0:c1 + 1]]
            R0, C0 = r0 + roi[0], c0 + roi[2]
            wins.append({'name': f'{name}_r{R0}_c{C0}', 'segment': name, 'band': b,
                         'cells': [R0, R0 + PATCH[0], C0, C0 + PATCH[1]], 'cells_roi': rc, 'density': round(dn, 4),
                         'ink_mm2': round(float(ink[r0:r1, c0:c1][inside_c[r0:r1, c0:c1]].sum() * CELL_B ** 2 * px_mm2), 3),
                         'valid_cells': round(float(inside_c[r0:r1, c0:c1].mean()), 3),
                         'z_min': round(float(zz.min()), 1), 'z_max': round(float(zz.max()), 1),
                         'canvas_origin': [R0 * CELL_B, C0 * CELL_B], 'shape_px': [PATCH[0] * CELL_B, PATCH[1] * CELL_B]})
    band_ink = {b: round(float(w[v20.band_of(zc) == b].sum() * CELL_B ** 2 * px_mm2), 3) for b in range(3)}
    return {'segment': name, 'band_edges': list(edges), 'band_ink_mm2': band_ink, 'roi_cells_B': roi,
            'mesh_for_step2': s0['mesh_for_step2'], 'windows': wins}


def study_of(name):
    return SCROLLS[segs_json()['segments'][name]['scroll']]['study']


def plan(names, write=True):
    for st in STUDY_FILE:
        ns = [n for n in names if study_of(n) == st]
        if ns:
            plan_study(ns, STUDY_FILE[st], write)


def plan_study(names, fname, write=True):
    reg = os.path.join(RESULTS, fname)
    pj = os.path.join(DATA, fname)
    have = jload(pj) or {'segments': {}}
    for n in names:
        if n not in have['segments']:
            have['segments'][n] = json.loads(json.dumps(plan_one(n)))
    have['rule'] = json.loads(json.dumps(dict(PATCH=PATCH, MAX_TEST=MAX_TEST, REL=0.2, MIN_DENS=0.02, MIN_VALID=0.5,
                                              LAYERS=LAYERS, LEVEL=1, UM_PX=UM_PX, SMOOTH=SMOOTH, DS=DS,
                                              NEAR_PX=v30.NEAR_PX)))
    if fname != 'windows.json':
        have['depth_offsets_layers'] = DEPTH
    jsave(have, pj)
    r = jload(reg)
    if r is not None and all(n in r['segments'] for n in names):
        same = all(r['segments'][n] == have['segments'][n] for n in names) and r['rule'] == have['rule'] \
            and r.get('depth_offsets_layers') == have.get('depth_offsets_layers')
        log('PLAN', 'SAME' if same else 'DIFFERENT', 'as', reg, f'({len(names)} segments compared)')
    elif write:                                          # registration: add segments not yet in the committed plan
        r = r or {k: v for k, v in have.items() if k != 'segments'} | {'segments': {}}
        assert r['rule'] == have['rule'], 'plan rule changed'
        r['segments'].update({n: have['segments'][n] for n in names if n not in r['segments']})
        r['segments'] = dict(sorted(r['segments'].items()))
        jsave(r, reg)
        log('PLAN written to', reg)
    for n in names:
        p = have['segments'][n]
        if p.get('stopped'):
            log('PLAN', n, 'no windows:', p['stopped'])
        else:
            nb = [sum(1 for w in p['windows'] if w['band'] == b) for b in range(3)]
            log('PLAN', n, f"{len(p['windows'])} windows {nb}, band edges {p['band_edges']}, "
                f"ink {sum(w['ink_mm2'] for w in p['windows']):.2f} mm2")
    estimate(have)
    return have


def windows(pl, scroll=None, extra=False):
    """Planned windows; with extra, only the EXTRA_PER_BAND densest of each band of each segment (descriptive arm)."""
    S = segs_json()['segments']
    out = []
    for n, p in pl['segments'].items():
        if scroll not in (None, S[n]['scroll']):
            continue
        for w in p['windows']:
            k = sum(1 for v in p['windows'] if v['band'] == w['band'] and p['windows'].index(v) < p['windows'].index(w))
            if not extra or k < EXTRA_PER_BAND:
                out.append(w)
    return out


def estimate(pl, scroll=None):
    """Main arm (rule) and extra arm (descriptive, PHerc1667) GPU-machine minutes per scroll."""
    out = {}
    per = EST['render_min'] + EST['score_min']
    for sc in SCROLLS:
        if scroll not in (None, sc):
            continue
        ws = windows(pl, sc)
        segs = sorted({w['segment'] for w in ws})
        nd = sum(1 for w in ws if w['segment'] in DEPTH)              # deep renders (131 layers) + one more map
        main = len(ws) * per + nd * (1.1 * EST['render_min'] + EST['score_min'])
        extra = len(windows(pl, sc, True)) * per if SCROLLS[sc]['refit_required'] else 0.0
        out[sc] = {'windows': len(ws), 'segments': len(segs), 'main_min': round(main), 'extra_min': round(extra)}
        log(f'ESTIMATE {sc}: {len(ws)} windows on {len(segs)} segments; main ~{main / 60:.1f} h'
            + (f', extra (published-mesh arm, descriptive) ~{extra / 60:.1f} h' if extra else '') + ' (cap 4 h per queue item)')
    tot = sum(v['main_min'] + v['extra_min'] for v in out.values())
    log(f'ESTIMATE total ~{tot / 60:.1f} h for the scrolls above; ' + ('one queue item' if tot <= 240 else 'one queue item per scroll'))
    return out


# --------------------------------------------------------------------------------------------------------- render
def committed_plan():
    """Scores need the committed windows files, unchanged in git (README: fixed before any score). windows.json (V-031)
    is required; windows_v031b.json (V-031b) is added when present."""
    root = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
    pl = {'segments': {}}
    for st, fname in STUDY_FILE.items():
        reg = os.path.join(RESULTS, fname)
        q = jload(reg)
        if q is None:
            if st == 'V-031':
                raise SystemExit(f'{fname} is not committed; run plan and commit it before render/score')
            continue
        rel = os.path.relpath(reg, root)
        stt = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--', rel], capture_output=True, text=True).stdout
        tracked = subprocess.run(['git', '-C', root, 'ls-files', '--error-unmatch', rel], capture_output=True).returncode == 0
        if not tracked or stt.strip():
            raise SystemExit(f'{rel} must be committed and unchanged before render/score (git status: {stt.strip() or "untracked"})')
        pl['segments'].update(q['segments'])
    return pl


def all_plans():
    pl = {'segments': {}}
    for fname in STUDY_FILE.values():
        pl['segments'].update((jload(os.path.join(RESULTS, fname)) or {'segments': {}})['segments'])
    return pl


_MESH = {}


def window_mesh(w, which):
    """A tifxyz of the window (+ SUB_MARGIN cells) cut from the step-2 mesh (B or B') or the published mesh B; the
    full mesh is read from the bucket once per segment and kept in memory. Returns (dir, rows, cols) for render_crop."""
    name = w['segment']
    if (name, which) not in _MESH:
        _MESH.clear()
        s = segs_json()['segments'][name]
        Bm, vB = mesh(s['mesh_B'], np.float32)
        if which == 'Bfit':
            Mpub, Mlsq, _ = transforms(s['scroll'])
            T = inverse(Mlsq)[:, :3] @ Mpub[:, :3], inverse(Mlsq)[:, :3] @ Mpub[:, 3] + inverse(Mlsq)[:, 3]
            for r0 in range(0, vB.shape[0], 256):                       # B' = M_lsq^-1 M_pub B, in place, by rows
                Bm[r0:r0 + 256] = (Bm[r0:r0 + 256].astype(np.float64) @ T[0].T + T[1]).astype(np.float32)
        _MESH[(name, which)] = (Bm, vB)
    Bm, vB = _MESH[(name, which)]
    R0, R1, C0, C1 = w['cells']
    out = os.path.join(DATA, name, 'win', f"mesh_{w['name']}_{which}")
    return write_sub(Bm, vB, (R0, R1), (C0, C1), out, 0.05)


def window_labels(w, lab_cache):
    """Carried labels and carried mask on the window's L1 pixels, from the segment's carried (u, v)."""
    wd = sdir(w['segment'], 'win')
    lp, mp = os.path.join(wd, w['name'] + '_lab.png'), os.path.join(wd, w['name'] + '_carried.npy')
    from PIL import Image
    if not os.path.exists(lp):
        v30.src()
        n = w['segment']
        if n not in lab_cache:
            lab_cache.clear()
            lab_cache[n] = Labels(segs_json()['segments'][n], n)
        cd = os.path.join(DATA, n, 'carry')
        U, V = np.load(os.path.join(cd, 'U.npy'), mmap_mode='r'), np.load(os.path.join(cd, 'V.npy'), mmap_mode='r')
        C = v30.load_bits(os.path.join(cd, 'C.npy'))
        r0, r1, c0, c1 = w['cells_roi']
        g = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        Ug, Vg = np.asarray(U[g]), np.asarray(V[g])
        lab_p, ok = pixels(Ug, Vg, C[g], lab_cache[n].crop(Ug, Vg, C[g]))
        h, wd_ = w['shape_px']
        Image.fromarray(lab_p[:h, :wd_].astype(np.uint8) * 255).save(lp)
        v30.save_bits(mp, ok[:h, :wd_])
    return np.asarray(Image.open(lp)) >= 128, v30.load_bits(mp)


def arms_of(name, extra=False):
    """Main arm: the canonical model along the step-2 mesh. Extra arm (descriptive, PHerc1667 only): the canonical model
    along the published mesh B, the mesh the organisers' map was made along."""
    s0 = jload(os.path.join(RESULTS, 'step0', name + '.json'))
    if not extra:
        return {'canonical': s0['mesh_for_step2']}
    return {'canonical_pubmesh': 'B'} if s0['mesh_for_step2'] == 'Bfit' else {}


def render_stage(scroll, names, extra=False):
    pl = committed_plan()
    todo = [w for w in windows(pl, scroll, extra) if (not names or w['name'] in names) and arms_of(w['segment'], extra)]
    rows = jload(os.path.join(RESULTS, 'scores.json'), {})
    lab_cache = {}
    for i, w in enumerate(todo):
        n = w['segment']
        if not os.path.exists(os.path.join(DATA, n, 'carry', 'roi.json')):
            carry_one(n)                                    # recompute the carry on this machine (CARRY SAME expected)
        sc = segs_json()['scrolls'][segs_json()['segments'][n]['scroll']]
        lab, cm = window_labels(w, lab_cache)
        for arm, which in arms_of(n, extra).items():
            out = os.path.join(DATA, n, 'win', f"{w['name']}_{arm}")
            info = jload(out + '.v031.json')
            if info is None and arm not in rows.get(w['name'], {}):
                t0 = time.time()
                md, rr, cc = window_mesh(w, which)
                deep = DEPTH.get(n) if arm == 'canonical' else None
                ct = render(md, sc['vol_B'], out, rr, cc, LAYERS + 2 * abs(deep) if deep else LAYERS, 1)
                shutil.rmtree(md, ignore_errors=True)
                assert list(ct.shape[1:]) == w['shape_px'], (ct.shape, w['shape_px'])
                unc = (ct != 0).any(0) & ~cm
                ct[:, unc] = 0                              # uncarried CT is not scored (V-030)
                np.save(out + '.npy', ct)
                info = {'uncarried_ct_px': int(unc.sum()), 'ct_valid_px': int((ct != 0).any(0).sum()), 'mesh': which,
                        'layers': int(ct.shape[0]), 'seconds': round(time.time() - t0, 1)}
                jsave(info, out + '.v031.json')
                timing(f"render_{sc['vol_B'].split('/')[0]}{'_extra' if extra else ''}", time.time() - t0)
            log(f'[{i + 1}/{len(todo)}]', w['name'], arm, json.dumps(info))


# --------------------------------------------------------------------------------------------------------- score
_MRG = {}


def mrg_crop(w):
    """The organisers' mrg20736-1um map on the window (same L1 canvas), 0-1, read tile by tile from the bucket."""
    out = os.path.join(DATA, w['segment'], 'win', w['name'] + '_mrg.npy')
    if os.path.exists(out):
        return np.load(out)
    import tifffile
    import zarr
    s = segs_json()['segments'][w['segment']]
    if w['segment'] not in _MRG:
        _MRG.clear()
        t = tifffile.TiffFile(RangeFile(B + s['mrg']))
        _MRG[w['segment']] = (t, zarr.open(t.pages[0].aszarr(), mode='r'))
    t, z = _MRG[w['segment']]
    assert list(z.shape) == s['canvas_B'] and z.dtype == np.uint8, (z.shape, z.dtype)
    y, x = w['canvas_origin']
    h, wd = w['shape_px']
    a = np.asarray(z[y:y + h, x:x + wd]).astype(np.float32) / 255.0
    np.save(out, a)
    return a


def layers(stack, offset):
    """The 62 layers the model reads: of a 63-layer render, layers 0..61 (V-030); of a deeper render (2k more layers),
    the same positions shifted by `offset` layers along the render normal (negative = towards -normal)."""
    k = (stack.shape[0] - LAYERS) // 2
    assert k >= abs(offset), (stack.shape, offset)
    return stack[k + offset:k + offset + 62]


def score_stage(scroll, extra=False):
    """Main: canonical (step-2 mesh) and mrg20736-1um AUC on the same ds8 pixels (offset 0). Extra: canonical along the
    published mesh, on the main arm's pixels (saved as sets.npz)."""
    pl = committed_plan()
    d = v30.infer_mod()
    rows_p = os.path.join(RESULTS, 'scores.json')
    rows = jload(rows_p, {})
    lab_cache = {}
    for w in windows(pl, scroll, extra):
        n = w['segment']
        arms = arms_of(n, extra)
        if not arms or (w['name'] in rows and all(a in rows[w['name']] for a in arms)):
            continue
        if extra and w['name'] not in rows:
            raise SystemExit(f"{w['name']}: score the main arm first")
        t0 = time.time()
        base = os.path.join(DATA, n, 'win', w['name'])
        stacks = {a: base + f'_{a}.npy' for a in arms}
        missing = [a for a, p in stacks.items() if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"{w['name']}: not rendered ({missing}); run render{' --extra' if extra else ''} first")
        if extra:
            z = np.load(base + '_sets.npz')
            sets, r = (z['pos'], z['neg']), rows[w['name']]
        else:
            lab, cm = window_labels(w, lab_cache)
            sets = v30.pixel_sets(lab, cm, layers(np.load(stacks['canonical'], mmap_mode='r'), 0))
            np.savez_compressed(base + '_sets.npz', pos=sets[0], neg=sets[1])
            r = {'segment': n, 'scroll': segs_json()['segments'][n]['scroll'], 'band': w['band'],
                 'pos_ds8': int(sets[0].sum()), 'neg_ds8': int(sets[1].sum())}
        for a, p in stacks.items():
            m = d.infer(os.environ['CANON_CKPT'], layers(np.load(p, mmap_mode='r'), 0), base + f'_{a}_ink.npy')
            r[a] = v30.auc_on(m, sets)
            if a == 'canonical' and DEPTH.get(n):              # V-031b descriptive depth column (#1912)
                o = DEPTH[n]
                m = d.infer(os.environ['CANON_CKPT'], layers(np.load(p, mmap_mode='r'), o), base + f'_{a}_ink_o{o:+d}.npy')
                r[f'canonical_depth{o:+d}'] = v30.auc_on(m, sets)
        if not extra:
            r['mrg20736_1um'] = v30.auc_on(mrg_crop(w), sets)
        r['seconds' + ('_extra' if extra else '')] = round(time.time() - t0, 1)
        rows[w['name']] = json.loads(json.dumps(r).replace('NaN', 'null'))
        jsave(rows, rows_p)
        timing(f"score_{r['scroll']}{'_extra' if extra else ''}", time.time() - t0)
        log('WINDOW', w['name'], ' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}'
                                          for k, v in r.items() if k in list(arms) + ['mrg20736_1um', 'pos_ds8', 'neg_ds8']))
        if not os.environ.get('V031_KEEP_STACKS'):
            for p in stacks.values():
                os.remove(p)


# --------------------------------------------------------------------------------------------------------- rule
def group_rule(rows):
    ok = [r for r in rows if r.get('canonical') is not None and r.get('mrg20736_1um') is not None]
    n = len(ok)
    diff = [r['canonical'] - r['mrg20736_1um'] for r in ok]
    wins = sum(x >= 0 for x in diff)
    need = int(np.ceil(WIN_FRAC * n)) if n else 0
    mean = float(np.mean(diff)) if diff else None
    if n < N_MIN:
        vd, why = 'UNKNOWN', f'{n} scored windows (need >= {N_MIN})'
    elif wins >= need and mean >= MEAN_DIFF_MIN:
        vd, why = 'PASS', f'canonical >= mrg20736-1um on {wins} of {n} windows (need {need}), mean diff {mean:+.4f} (>= 0)'
    else:
        vd, why = 'FAIL', f'canonical >= mrg20736-1um on {wins} of {n} windows (need {need}), mean diff {mean:+.4f} (need >= 0)'
    return {'verdict': vd, 'reason': why, 'n_scored': n, 'n_rows': len(rows), 'wins': wins, 'need': need,
            'mean_diff': mean, 'mean_canonical': float(np.mean([r['canonical'] for r in ok])) if ok else None,
            'mean_mrg': float(np.mean([r['mrg20736_1um'] for r in ok])) if ok else None}


def rule_stage():
    """Per scroll; pooled over V-031's scrolls (PHerc1667 + PHercParis4) = V-031's registered verdict; pooled over all
    three scrolls (with PHerc0139) = the V-031b extension, reported separately."""
    pl = all_plans()
    rows = jload(os.path.join(RESULTS, 'scores.json'), {})
    planned = {w['name']: w for w in windows(pl)}
    S = segs_json()['segments']
    by = {sc: [dict(name=k, **r) for k, r in rows.items() if r['scroll'] == sc] for sc in SCROLLS}
    per = {}
    for sc, rs in by.items():
        todo = [k for k, w in planned.items() if S[w['segment']]['scroll'] == sc and k not in rows]
        per[sc] = group_rule(rs)
        per[sc]['study'] = SCROLLS[sc]['study']
        if todo:
            per[sc].update(verdict='UNKNOWN', reason=f'{len(todo)} planned windows not scored yet')

    def pool(scs):
        g = group_rule([r for sc in scs for r in by[sc]])
        if any(per[sc]['reason'].endswith('not scored yet') for sc in scs):
            g.update(verdict='UNKNOWN', reason='not every planned window is scored')
        return g
    v031 = [sc for sc in SCROLLS if SCROLLS[sc]['study'] == 'V-031']
    pooled, ext = pool(v031), pool(list(SCROLLS))
    allrows = [r for rs in by.values() for r in rs]
    cols = ['canonical', 'mrg20736_1um', 'canonical_pubmesh'] + sorted({f'canonical_depth{o:+d}' for o in DEPTH.values()})
    low = {a: sorted(r['name'] for r in allrows if r.get(a) is not None and r[a] < AUC_OK) for a in cols}
    used = sum(timing_total().values())
    out = {'verdict': pooled['verdict'], 'pooled': pooled, 'pooled_v031b_extension': ext, 'per_scroll': per,
           'below_auc_ok_0.75': low, 'rows': allrows, 'timing_s': timing_total(), 'gpu_stage_h': round(used / 3600, 2),
           'depth_offsets_layers': DEPTH,
           'rule': {'WIN_FRAC': WIN_FRAC, 'MEAN_DIFF_MIN': MEAN_DIFF_MIN, 'N_MIN': N_MIN, 'AUC_OK': AUC_OK,
                    'combine': 'V-031 verdict = pooled over PHerc1667 + PHercParis4; per-scroll results alongside; '
                               'V-031b extension = pooled over all three scrolls, reported separately'},
           'step0': {n: jload(os.path.join(RESULTS, 'step0', n + '.json'), {}).get('line') for n in pl['segments']},
           'carry': {n: jload(os.path.join(RESULTS, 'carry', n + '.json'), {}).get('line') for n in pl['segments']}}
    jsave(out, os.path.join(RESULTS, 'summary.json'), os.path.join(RUN, 'payload', 'v031-summary.json'))
    lines = []
    for sc, p in per.items():
        lines.append(f"RULE {sc} ({p['study']}): {p['reason']}; mean canonical {fmt(p['mean_canonical'])} vs "
                     f"mrg20736-1um {fmt(p['mean_mrg'])} -> {p['verdict']}")
    lines.append(f"RULE pooled V-031 ({' + '.join(v031)}): {pooled['reason']}; mean canonical "
                 f"{fmt(pooled['mean_canonical'])} vs mrg20736-1um {fmt(pooled['mean_mrg'])}")
    lines.append(f"VERDICT {pooled['verdict']} - {pooled['reason']} (V-031 registered verdict, pooled; per scroll: "
                 + ', '.join(f"{sc} {per[sc]['verdict']}" for sc in v031) + ')')
    lines.append(f"RULE pooled V-031b extension (all {len(SCROLLS)} scrolls): {ext['reason']}; mean canonical "
                 f"{fmt(ext['mean_canonical'])} vs mrg20736-1um {fmt(ext['mean_mrg'])}")
    lines.append(f"VERDICT_V031B_EXTENSION {ext['verdict']} - {ext['reason']} (pooled over all scrolls; PHerc0139 alone: "
                 f"{per['PHerc0139']['verdict']}; not V-031's verdict)")
    for ln in lines:
        log(ln)
    write_result(out, lines, planned)
    return out


def fmt(x):
    return 'n/a' if x is None else f'{x:.4f}'


def write_result(out, lines, planned):
    """RESULT.md skeleton with the RULE/VERDICT lines and the per-window table (the desktop adds the prose)."""
    p = os.path.join(RESULTS, 'RESULT.md')
    head = ['# V-031: canonical 2.4 µm model zero-shot vs mrg20736-1um on 1.129 µm scans: ' + out['verdict'], '',
            'Spec: `experiments/v031/README.md`. Driver: `experiments/v031/v031_run.py`. Lines printed by `rule`:', '']
    head += [f'- `{ln}`' for ln in lines]
    head += ['', f"GPU-stage time {out['gpu_stage_h']} h.", '', '## Per window', '',
             '| window | scroll | band | canonical | mrg20736-1um | canonical − mrg | canonical on published mesh (descr.) '
             '| canonical at depth offset (descr., V-031b) | pos / neg ds8 px |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in sorted(out['rows'], key=lambda r: r['name']):
        c, m = r.get('canonical'), r.get('mrg20736_1um')
        head.append(f"| {r['name']} | {r['scroll']} | {r['band']} | {fmt(c)} | {fmt(m)} | "
                    f"{'n/a' if c is None or m is None else f'{c - m:+.4f}'} | {fmt(r.get('canonical_pubmesh'))} | "
                    + ' '.join(f'{k[15:]} layers: {fmt(v)}' for k, v in r.items() if k.startswith('canonical_depth'))
                    + f" | {r['pos_ds8']} / {r['neg_ds8']} |")
    head += ['', '## Below --auc-ok 0.75 (villa scan_diagnosis would advise adapting; descriptive)', '']
    for a, ns in out['below_auc_ok_0.75'].items():
        head.append(f"- {a}: {len(ns)} windows" + (': ' + ', '.join(ns) if ns else ''))
    head += ['', '## Steps 0 and 1', '']
    for n in sorted(out['step0']):
        head.append(f"- {n}: `{out['step0'][n]}`; `{out['carry'][n]}`")
    old = open(p).read() if os.path.exists(p) else ''
    keep = old.split('<!-- prose -->', 1)[1] if '<!-- prose -->' in old else '\n'
    open(p, 'w').write('\n'.join(head) + '\n\n<!-- prose -->' + keep)


# --------------------------------------------------------------------------------------------------------- status/check
def status():
    S = jload(os.path.join(RESULTS, 'segments.json'))
    if not S:
        log('resolve not run'); return
    for n in S['segments']:
        s0 = jload(os.path.join(RESULTS, 'step0', n + '.json'))
        c = jload(os.path.join(RESULTS, 'carry', n + '.json'))
        log(n, '|', s0['line'] if s0 else 'step0 not run', '|', c['line'] if c else 'carry not run')
    pl = all_plans()
    if not pl['segments']:
        log('plan not committed'); return
    rows = jload(os.path.join(RESULTS, 'scores.json'), {})
    for sc in SCROLLS:
        ws = windows(pl, sc)
        rend = sum(os.path.exists(os.path.join(DATA, w['segment'], 'win', w['name'] + '_canonical.v031.json')) for w in ws)
        sco = sum(w['name'] in rows for w in ws)
        log(f'{sc}: {len(ws)} windows, rendered {rend}, scored {sco}')
    log(f'GPU-stage time {sum(timing_total().values()) / 3600:.2f} h; ' + json.dumps(timing_total()))


def selftest():
    v30.demo()
    # the strip-wise per-cell carry equals the whole-block pixel carry
    v30.src()
    rng = np.random.default_rng(1)
    H, W = 23, 37
    U = (np.arange(H)[:, None] * 0.47 + 1.0 + 0 * np.arange(W)[None]).astype(np.float32)
    V = (np.arange(W)[None] * 0.47 + 2.0 + 0 * np.arange(H)[:, None]).astype(np.float32)
    C = np.ones((H, W), bool); C[5, 7] = False
    lab = rng.random((H * 12, W * 12)) > 0.6
    ink, full = canvas_cells(U, V, C, ArrLabels(lab), strip=4, colstrip=7)
    lp, ok = pixels(U, V, C, lab)
    lp1, _ = pixels(U, V, C, ArrLabels(lab).crop(U, V, C))
    assert (lp1 == lp).all(), 'label crop'
    h, w = (H - 1) * CELL_B, (W - 1) * CELL_B
    assert np.allclose(ink, lp[:h, :w].reshape(H - 1, CELL_B, W - 1, CELL_B).mean((1, 3))), 'strip carry'
    assert (full == ok[:h, :w].reshape(H - 1, CELL_B, W - 1, CELL_B).all((1, 3))).all(), 'strip carried mask'
    lp2, ok2 = pixels(U[3:9, 4:12], V[3:9, 4:12], C[3:9, 4:12], lab)       # a window block equals the canvas crop
    assert (lp2[:50, :70] == lp[30:80, 40:110]).all() and (ok2[:50, :70] == ok[30:80, 40:110]).all(), 'window block'
    # chunked carry == V-030 carry (synthetic meshes as in v30.demo)
    Ha, Wa = 40, 60
    ii, jj = np.meshgrid(np.arange(Ha, dtype=float), np.arange(Wa, dtype=float), indexing='ij')
    A = np.stack([1000 + 20 * jj, 2000 + 20 * ii * 0.8, 3000 + 20 * ii * 0.6 + 0.01 * jj ** 2], -1)
    M = np.array([[0.47, 0, 0, 500.0], [0, 0.47, 0, 600.0], [0, 0, 0.47, 700.0]])
    bi, bj = np.meshgrid(np.arange(60, dtype=float), np.arange(100, dtype=float), indexing='ij')
    ut, vt = 2 + bi * 0.47, 3 + bj * 0.47
    Bm = apply(inverse(M), np.stack([1000 + 20 * vt, 2000 + 20 * ut * 0.8, 3000 + 20 * ut * 0.6 + 0.01 * vt ** 2], -1))
    vB = (ut <= Ha - 1) & (vt <= Wa - 1)
    a, b = v30.carry_planes(A, np.ones((Ha, Wa), bool), Bm, vB, M), carry_planes(A, np.ones((Ha, Wa), bool), Bm, vB, M, rows=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), 'chunked carry differs from V-030 carry'
    # deep stacks: main layers equal the 63-layer render's, the depth window is shifted by the offset (V-031b)
    dd, ss = np.arange(131)[:, None, None] - 65, np.arange(63)[:, None, None] - 31
    assert (layers(dd, 0) == layers(ss, 0)).all() and layers(dd, -34)[0, 0, 0] == -65, 'deep stack layers'
    # rule
    mk = lambda c, m: {'canonical': c, 'mrg20736_1um': m}                                       # noqa: E731
    assert group_rule([mk(0.9, 0.8)] * 7 + [mk(0.7, 0.8)] * 3)['verdict'] == 'PASS'              # 7/10, mean +0.04
    assert group_rule([mk(0.9, 0.8)] * 6 + [mk(0.7, 0.8)] * 4)['verdict'] == 'FAIL'              # 6/10
    assert group_rule([mk(0.81, 0.8)] * 7 + [mk(0.5, 0.8)] * 3)['verdict'] == 'FAIL'             # mean < 0
    assert group_rule([mk(0.9, 0.8)] * 9)['verdict'] == 'UNKNOWN'
    assert group_rule([mk(0.8, 0.8)] * 10)['verdict'] == 'PASS'                                  # ties count, mean 0
    log('SELFTEST ok (V-030 demo, strip carry = block carry, window block, rule)')


def check():
    bad = []

    def ok(name, cond, note=''):
        log('CHECK', 'ok  ' if cond else 'FAIL', name, note)
        if not cond:
            bad.append(name)

    if os.environ.get('V030_SRC'):
        try:
            selftest(); ok('selftest', True)
        except Exception as e:                                             # noqa: BLE001
            ok('selftest', False, repr(e))
        for f in ('render_crop.py', 'render_tifxyz.py', 'infer_crop.py', 'march_bench.py'):
            ok(f'V031_SRC/{f}', os.path.exists(os.path.join(os.environ['V030_SRC'], f)))
    else:
        log('CHECK skip selftest and V031_SRC (unset)')
    S = jload(os.path.join(RESULTS, 'segments.json'))
    if S is None:
        ok('segments.json', False, 'run resolve')
    else:
        for sc, c in S['scrolls'].items():
            for k in ('vol_A', 'vol_B'):
                probe(ok, f'{sc} {k}', B + c[k] + '/0/.zarray')
            probe(ok, f'{sc} transform', B + c['transform'])
        for n, s in S['segments'].items():
            if s['status'] != 'ok':
                log('CHECK info', n, 'not usable:', s['status']); continue
            for k, u in (('mesh_A', s['mesh_A'] + 'x.tif'), ('mesh_B', s['mesh_B'] + 'x.tif'),
                         ('labels', s['labels'] + 'zarr.json'), ('mrg', s['mrg']), ('sv_B', s['sv_B'] + '5/.zarray')):
                probe(ok, f'{n} {k}', B + u)
    for d, what in (('step0', 'PLACEMENT'), ('carry', 'CARRY')):
        for f in sorted(os.listdir(os.path.join(RESULTS, d))) if os.path.isdir(os.path.join(RESULTS, d)) else []:
            log('CHECK info', d, f[:-5], jload(os.path.join(RESULTS, d, f))['line'])
    w = all_plans()
    log('CHECK info windows', f"{len(windows(w))} windows committed")
    v = os.environ.get('V030_VILLA')
    if v:
        head = subprocess.run(['git', '-C', v, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        ok('villa commit', head == VILLA_SHA, head)
        ok('villa optimized_inference', os.path.exists(os.path.join(v, 'ink-detection', 'optimized_inference',
                                                                     'model_resnet3d_3d_decoder.py')))
    else:
        log('CHECK skip villa (V031_VILLA unset)')
    ck = os.environ.get('CANON_CKPT')
    ok('CANON_CKPT', bool(ck) and os.path.exists(ck), str(ck)) if ck else log('CHECK skip CANON_CKPT (unset)')
    try:
        import torch
        if os.environ.get('INK_DEVICE', 'cuda') == 'cuda':
            ok('cuda', torch.cuda.is_available(), torch.__version__)
    except ImportError:
        log('CHECK skip cuda (no torch in this python)')
    for mod in ('zarr', 'cv2', 'tifffile', 'imagecodecs', 'PIL'):
        try:
            m = __import__(mod)
            ok(f'import {mod}', True, getattr(m, '__version__', ''))
        except ImportError as e:
            ok(f'import {mod}', False, repr(e))
    log('CHECK OK' if not bad else f'CHECK FAIL {bad}')
    return not bad


def probe(ok, name, u):
    try:
        rq = urllib.request.Request(u, headers={'Range': 'bytes=0-1023'})
        with urllib.request.urlopen(rq, timeout=60) as r:
            ok(f'bucket {name}', r.status in (200, 206), str(r.status))
    except Exception as e:                                                 # noqa: BLE001
        ok(f'bucket {name}', False, f'{e} {u}')


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd in ('check', '--check'):
        sys.exit(0 if check() else 1)
    os.makedirs(DATA, exist_ok=True)
    scroll = None
    if '--scroll' in args:
        i = args.index('--scroll'); scroll = args[i + 1]; args = args[:i] + args[i + 2:]
    force = '--force' in args
    args = [a for a in args if a != '--force']
    if cmd == 'resolve':
        resolve(args or None)
    elif cmd == 'step0':
        for n in select(args):
            step0_one(n, force)
    elif cmd == 'carry':
        for n in select(args):
            carry_one(n)
    elif cmd == 'plan':
        plan(select(args))
    elif cmd == 'cloud':                                 # steps 0, 1 and plan per segment in one process (LOWDISK)
        for n in select(args):
            step0_one(n, force)
            if gate(n, 'carry')[0] is not None:
                carry_one(n)
            plan([n])
            _CARRY.clear()
            shutil.rmtree(os.path.join(DATA, n), ignore_errors=True)
    elif cmd == 'render':
        render_stage(scroll, [a for a in args if a != '--extra'], '--extra' in args)
    elif cmd == 'score':
        score_stage(scroll, '--extra' in args)
    elif cmd == 'rule':
        rule_stage()
    elif cmd == 'status':
        status()
    elif cmd == 'selftest':
        selftest()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
