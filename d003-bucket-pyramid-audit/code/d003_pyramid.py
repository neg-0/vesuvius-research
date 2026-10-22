"""D-003: pyramid audit of every published surface volume in the open-data bucket (CPU, network only, no GPU, no training).

Context: ScrollPrize/villa#2107 found surface volumes on Hugging Face whose level 2 holds only the first 17 of 65 depth
slices while the metadata claims full depth. This audits the same defect classes on the open-data bucket we read from:
  META   every level's .zarray exists; XY shape of level k+1 is ceil(level k / 2) (+-1); depth is the same at every level.
  CHUNK  the centre chunks probed at every level exist (no HTTP 404).
  DEPTH  fraction of depth layers with any non-zero voxel in a probed chunk, per level (full-depth bug = low fraction at
         some level while level 0 is full).
  XLEVEL correlation of level 1 with the 2x2 mean of level 0 over the same window (misregistered or empty pyramid levels).
Probe positions are fixed fractions of the level-0 extent, first one whose level-0 chunk is non-zero.

usage: d003_pyramid.py list              (-> $D003_OUT/volumes.json, the sorted list of surface-volume zarr roots)
       d003_pyramid.py audit [N]         (audit the first N volumes (default all); resumable; -> $D003_OUT/rows.jsonl)
       d003_pyramid.py rule              (-> ../results/{summary.json,volumes.csv,RESULT.md})
       d003_pyramid.py selftest
Env: D003_OUT (default ../results/work, not committed), D003_WORKERS (default 8).
"""
import csv
import json
import os
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bucket as v31                                                     # noqa: E402

RES = os.path.join(HERE, '..', 'results')
WORK = os.environ.get('D003_OUT', os.path.join(RES, 'work'))
WORKERS = int(os.environ.get('D003_WORKERS', '8'))
PROBES = ((0.5, 0.5), (0.35, 0.35), (0.65, 0.65), (0.5, 0.35), (0.35, 0.5), (0.2, 0.2), (0.8, 0.8))
DEPTH_MIN = 0.9          # DEPTH flag: fewer than 90% of layers non-zero at a level whose level-0 probe has >= 90%
XCORR_MIN = 0.9          # XLEVEL flag: level-1 vs pooled level-0 correlation below this


def list_volumes():
    out = []
    ps, _ = v31.ls('')
    for sc in ps:
        if not sc.startswith('PHerc'):
            continue
        segs, _ = v31.ls(sc + 'segments/')
        for sg in segs:
            svs, _ = v31.ls(sg + 'surface-volumes/')
            out += [p.rstrip('/') for p in svs if p.rstrip('/').endswith('.zarr')]
    return sorted(out)


def get_json(path):
    return json.loads(v31.fetch(v31.B + path))


def chunk(root, lv, meta, iy, ix):
    """Decode chunk (all depth, iy, ix) of level lv -> (depth, h, w) array or None if the key is missing (404)."""
    import numcodecs
    sh, ch = meta['shape'], meta['chunks']
    try:
        raw = v31.fetch(f'{v31.B}{root}/{lv}/0/{iy}/{ix}')
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    c = numcodecs.get_codec(meta['compressor']) if meta.get('compressor') else None
    a = np.frombuffer(c.decode(raw) if c else raw, dtype=meta['dtype'])
    return a.reshape(ch[0], ch[1], ch[2])[:sh[0]]


def layer_frac(a):
    return float((a.reshape(a.shape[0], -1).max(1) > 0).mean())


def audit_one(root):
    r = {'root': root, 'flags': []}
    try:
        lvl, meta = 0, {}
        while True:
            try:
                meta[lvl] = get_json(f'{root}/{lvl}/.zarray')
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    break
                raise
            lvl += 1
        r['levels'] = lvl
        if lvl == 0:
            r['flags'].append('META_NO_LEVELS')
            return r
        r['shapes'] = [meta[k]['shape'] for k in range(lvl)]
        zs = {}
        try:
            for d in get_json(f'{root}/.zattrs')['multiscales'][0]['datasets']:
                zs[int(d['path'])] = float(d['coordinateTransformations'][0]['scale'][0])
        except Exception:                                                # noqa: BLE001
            r['flags'].append('META_NO_SCALES')
        r['zscale'] = [zs.get(k) for k in range(lvl)]
        for k in range(1, lvl):
            a, b = meta[k - 1]['shape'], meta[k]['shape']
            zr = (zs.get(k, 1.0) / zs.get(0, 1.0)) if zs else 1.0   # declared z downsampling vs level 0
            want = -(-meta[0]['shape'][0] // max(int(round(zr)), 1))
            if abs(b[0] - want) > 1:
                r['flags'].append(f'META_DEPTH_L{k}')
            if any(abs(b[i] - -(-a[i] // 2)) > 1 for i in (1, 2)):
                r['flags'].append(f'META_XY_L{k}')
        m0 = meta[0]
        ch0 = m0['chunks']
        ny0, nx0 = -(-m0['shape'][1] // ch0[1]), -(-m0['shape'][2] // ch0[2])
        pick = None
        for fy, fx in PROBES:
            iy, ix = min(int(fy * ny0), ny0 - 1), min(int(fx * nx0), nx0 - 1)
            a0 = chunk(root, 0, m0, iy, ix)
            if a0 is not None and a0.max() > 0:
                pick = (iy, ix, a0)
                break
        if pick is None:
            best = None
            for pre in v31.ls(f'{root}/0/0/')[0]:
                for key, size in v31.ls(pre)[1]:
                    if best is None or size > best[0]:
                        best = (size, key)
            if best is not None:
                iy, ix = (int(t) for t in best[1].split('/')[-2:])
                a0 = chunk(root, 0, m0, iy, ix)
                if a0 is not None and a0.max() > 0:
                    pick = (iy, ix, a0)
                    r['probe_fallback'] = True
        if pick is None:
            r['flags'].append('PROBE_EMPTY')
            return r
        iy, ix, a0 = pick
        r['probe'] = [iy, ix]
        fr = {0: layer_frac(a0)}
        for k in range(1, lvl):
            mk = meta[k]
            ck = mk['chunks']
            s = 2 ** k
            ky, kx = (iy * ch0[1] // s) // ck[1], (ix * ch0[2] // s) // ck[2]
            ak = chunk(root, k, mk, ky, kx)
            if ak is None:
                r['flags'].append(f'CHUNK_MISSING_L{k}')
                continue
            fr[k] = layer_frac(ak)
        r['layer_frac'] = fr
        if fr[0] >= DEPTH_MIN:
            r['flags'] += [f'DEPTH_L{k}' for k, f in fr.items() if f < DEPTH_MIN]
        if lvl > 1 and ('CHUNK_MISSING_L1' not in r['flags']):
            m1, c1 = meta[1], meta[1]['chunks']
            sy, sx = iy * ch0[1] // 2, ix * ch0[2] // 2          # level-1 pixel origin of the level-0 chunk
            y1, x1 = sy // c1[1], sx // c1[2]
            a1 = chunk(root, 1, m1, y1, x1)
            if a1 is not None:
                oy, ox = sy - y1 * c1[1], sx - x1 * c1[2]
                h, w = ch0[1] // 2, ch0[2] // 2
                if oy + h <= a1.shape[1] and ox + w <= a1.shape[2]:
                    p0 = a0[:, :h * 2, :w * 2].astype(np.float32).reshape(a0.shape[0], h, 2, w, 2).mean((2, 4))
                    p1 = a1[:, oy:oy + h, ox:ox + w].astype(np.float32)
                    d = min(p0.shape[0], p1.shape[0])
                    x, y = p0[:d].ravel(), p1[:d].ravel()
                    r['xcorr'] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
                    if r['xcorr'] is not None and r['xcorr'] < XCORR_MIN:
                        r['flags'].append('XLEVEL_L1')
    except Exception as e:                                               # noqa: BLE001
        r['flags'].append('ERROR')
        r['error'] = f'{type(e).__name__}: {e}'[:200]
    return r


def cmd_list():
    os.makedirs(WORK, exist_ok=True)
    vols = list_volumes()
    json.dump(vols, open(os.path.join(WORK, 'volumes.json'), 'w'))
    print('LIST volumes', len(vols))


def cmd_audit(n=None):
    vols = json.load(open(os.path.join(WORK, 'volumes.json')))[:n]
    path = os.path.join(WORK, 'rows.jsonl')
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)['root'] for l in open(path)}
    todo = [x for x in vols if x not in done]
    print('AUDIT todo', len(todo), 'done', len(done))
    with ThreadPoolExecutor(WORKERS) as ex, open(path, 'a') as f:
        for i, r in enumerate(ex.map(audit_one, todo), 1):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if i % 25 == 0:
                print('AUDIT', i, 'of', len(todo), flush=True)


def cmd_rule():
    rows = list({r['root']: r for r in map(json.loads, open(os.path.join(WORK, 'rows.jsonl')))}.values())
    assert len(rows) == len(json.load(open(os.path.join(WORK, 'volumes.json')))), 'audit incomplete'
    rows.sort(key=lambda r: r['root'])
    os.makedirs(RES, exist_ok=True)
    flagged = [r for r in rows if r['flags']]
    by = {}
    for r in flagged:
        for fl in r['flags']:
            by.setdefault(fl.split('_L')[0] if fl.startswith(('DEPTH', 'CHUNK', 'META_D', 'META_X')) else fl, []).append(r['root'])
    S = {'volumes': len(rows), 'flagged': len(flagged),
         'flag_counts': {k: len(v) for k, v in sorted(by.items())},
         'depth_flagged': sorted(r['root'] for r in rows if any(f.startswith('DEPTH') for f in r['flags'])),
         'errors': sorted(r['root'] for r in rows if 'ERROR' in r['flags'])}
    S['verdict'] = 'BUCKET_CLEAN' if not S['depth_flagged'] and not any(
        f.startswith(('META', 'CHUNK', 'XLEVEL')) for r in rows for f in r['flags']) and not S['errors'] else 'BUCKET_DEFECTS'
    json.dump(S, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)
    with open(os.path.join(RES, 'volumes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['root', 'levels', 'flags', 'layer_frac_by_level', 'xcorr_L0_L1'])
        for r in rows:
            w.writerow([r['root'], r.get('levels'), ' '.join(r['flags']),
                        ' '.join(f'{k}:{v:.2f}' for k, v in r.get('layer_frac', {}).items()),
                        '' if r.get('xcorr') is None else f"{r['xcorr']:.3f}"])
    L = [f"# D-003 surface-volume pyramid audit of the open-data bucket", '',
         f"VERDICT {S['verdict']}: {S['volumes']} volumes audited, {S['flagged']} flagged.", '',
         'Flag counts: ' + (', '.join(f'{k} {n}' for k, n in S['flag_counts'].items()) or 'none'), '']
    for k, v in sorted(by.items()):
        L += [f'## {k} ({len(v)})'] + [f'- {x}' for x in sorted(set(v))[:60]] + ['']
    open(os.path.join(RES, 'RESULT.md'), 'w').write('\n'.join(L))
    print('RULE', S['verdict'], S['volumes'], 'flagged', S['flagged'], S['flag_counts'])


def selftest():
    a = np.zeros((10, 4, 4), np.uint8)
    a[:3] = 1
    assert layer_frac(a) == 0.3
    r = audit_one('PHerc0841/segments/__does_not_exist__/surface-volumes/x.zarr')
    assert 'META_NO_LEVELS' in r['flags'], r
    print('SELFTEST ok')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['list']:
        cmd_list()
    elif a and a[0] == 'audit':
        cmd_audit(int(a[1]) if len(a) > 1 else None)
    elif a == ['rule']:
        cmd_rule()
    elif a == ['selftest']:
        selftest()
    else:
        sys.exit(__doc__)
