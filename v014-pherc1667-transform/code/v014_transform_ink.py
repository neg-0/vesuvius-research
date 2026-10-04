"""V-014: PHerc1667 1.129 um transform (villa #1843). For each labelled segment, render the same
2.399-grid window three ways and score the canonical ink model against the human labels:
  R  2.399 mesh on the 2.399 volume (level 0)
  P  M_pub^-1 . p on the 1.129 volume (level 1, oversample 2)  -- the published-equivalent mesh
  C  M_lsq^-1 . p on the 1.129 volume (level 1, oversample 2)  -- refit to the published landmarks
Runs inside the pinned cross-scan-ink-transfer src tree (imports march_bench for down/eval_map).
usage: v014_transform_ink.py <segment> [...]
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

import numpy as np
import tifffile
import zarr
from scipy import ndimage

import march_bench as mb

B = mb.B
VOL_2399 = f'{B}/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr'
VOL_1129 = f'{B}/PHerc1667/volumes/20260323082859-1.129um-0.2m-59keV-masked.zarr'
OUT = os.environ.get('V014_DATA', 'out/v014')
RUN = os.environ.get('PUZZLE_RUN_DIR', '.')
WIN_H, WIN_W, DS = mb.WIN_H, mb.WIN_W, mb.DS
OFFS = [-40, -20, 0, 20, 40]
MIN_DENS_FRAC = 0.20


def transforms():
    d = json.loads(urllib.request.urlopen(VOL_1129 + '/transform.json').read())
    Mpub = np.array(d['transformation_matrix'], float)
    Mv, F = np.array(d['moving_landmarks'], float), np.array(d['fixed_landmarks'], float)
    Mlsq = np.linalg.lstsq(np.c_[Mv, np.ones(len(Mv))], F, rcond=None)[0].T
    res = {k: float(np.sqrt((np.linalg.norm(np.c_[Mv, np.ones(len(Mv))] @ M.T - F, axis=1) ** 2).mean()))
           for k, M in (('pub', Mpub), ('lsq', Mlsq))}
    return Mpub, Mlsq, {'M_pub': Mpub.tolist(), 'M_lsq': Mlsq.tolist(), 'landmark_rms_px_2399': res}


def inv_apply(M, xyz):
    A = np.r_[M, [[0, 0, 0, 1]]]
    Ai = np.linalg.inv(A)
    return xyz @ Ai[:3, :3].T + Ai[:3, 3]


def labels(seg, d):
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix=PHerc1667/segments/{seg}/ink-labels/2.399um-volume-20251217075048/').read().decode()
    lz = sorted(set(re.findall(r'<Key>([^<]*inklabels\.zarr)/0/zarr\.json</Key>', lst)))[-1]
    root = os.path.join(d, 'inklabels.zarr')
    for f in ('zarr.json', '0/zarr.json', '0/c/0/0'):
        mb.get(f'{B}/{lz}/{f}', os.path.join(root, f))
    return zarr.open_array(os.path.join(root, '0'), mode='r')[:] > 0, lz


def mesh2399(seg, d):
    md = os.path.join(d, 'mesh_2399')
    for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
        mb.get(f'{B}/PHerc1667/segments/{seg}/mesh/{seg.split("-")[0]}-on-20251217075048-2.399um.tifxyz/{f}', os.path.join(md, f))
    xyz = np.stack([tifffile.imread(os.path.join(md, c + '.tif')).astype(np.float64) for c in 'xyz'], -1)
    return md, xyz, (xyz > -0.5).all(-1)


def write_mesh(md_src, xyz, valid, out):
    os.makedirs(out, exist_ok=True)
    for i, c in enumerate('xyz'):
        tifffile.imwrite(os.path.join(out, c + '.tif'), np.where(valid, xyz[..., i], -1).astype(np.float32))
    meta = json.load(open(os.path.join(md_src, 'meta.json')))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'))


def shape_l0(url):
    return json.loads(urllib.request.urlopen(url + '/0/.zarray').read())['shape']  # z, y, x


def run(seg, Mpub, Mlsq):
    d = os.path.join(OUT, seg); os.makedirs(d, exist_ok=True)
    rp = os.path.join(d, 'result.json')
    if os.path.exists(rp):
        return json.load(open(rp))
    lab, lz = labels(seg, d)
    md, xyz, valid = mesh2399(seg, d)
    zs, ys, xs = shape_l0(VOL_1129)
    P, C = inv_apply(Mpub, xyz), inv_apply(Mlsq, xyz)

    def inside(Q):
        return (Q[..., 0] >= 0) & (Q[..., 0] <= xs - 1) & (Q[..., 1] >= 0) & (Q[..., 1] <= ys - 1) & (Q[..., 2] >= 0) & (Q[..., 2] <= zs - 1)
    ok = ~valid | (inside(P) & inside(C))                  # invalid vertices do not block a cell
    Hc, Wc = min(lab.shape[0] // 20, xyz.shape[0]), min(lab.shape[1] // 20, xyz.shape[1])
    cells = lab[:Hc * 20, :Wc * 20].reshape(Hc, 20, Wc, 20).mean((1, 3), dtype=np.float32)
    dens = ndimage.uniform_filter(cells, (WIN_H, WIN_W), mode='constant')
    bad = ndimage.maximum_filter((~ok[:Hc, :Wc]).astype(np.uint8), (WIN_H, WIN_W), mode='constant', cval=1) > 0
    # window centre (cy, cx) allowed only if the whole window [cy-WIN_H//2, +WIN_H) is ok; maximum_filter centres match
    res = {'segment': seg, 'labels': lz, 'grid': list(xyz.shape[:2]), 'label_grid': list(lab.shape),
           'frac_valid_inside_both': float((inside(P) & inside(C))[valid].mean())}
    if Hc < WIN_H or Wc < WIN_W:
        res['status'] = 'UNKNOWN: segment smaller than window'
    else:
        dfree = dens.max()
        dm = np.where(bad, -1, dens)
        cy, cx = np.unravel_index(np.argmax(dm), dm.shape)
        res['density_unrestricted'] = float(dfree); res['density_window'] = float(dm[cy, cx])
        if dm[cy, cx] < MIN_DENS_FRAC * dfree or dm[cy, cx] <= 0:
            res['status'] = 'UNKNOWN: no in-volume labelled window'
    if 'status' in res:
        json.dump(res, open(rp, 'w'), indent=1); return res
    r0, c0 = int(np.clip(cy - WIN_H // 2, 0, Hc - WIN_H)), int(np.clip(cx - WIN_W // 2, 0, Wc - WIN_W))
    assert ok[r0:r0 + WIN_H, c0:c0 + WIN_W].all()
    res['window_cells'] = [r0, r0 + WIN_H, c0, c0 + WIN_W]
    w = (slice(r0, r0 + WIN_H), slice(c0, c0 + WIN_W))
    vw = valid[w]
    disp = np.linalg.norm(P[w] - C[w], axis=-1)[vw] * 1.129
    res['displacement_um'] = {'median': float(np.median(disp)), 'p90': float(np.percentile(disp, 90)), 'max': float(disp.max())}
    lab8 = mb.down(lab); e0 = (r0 * 20 // DS, c0 * 20 // DS)
    arms = {'R': (md, VOL_2399, ['--level', '0', '--oversample', '1']),
            'P': (os.path.join(d, 'mesh_P'), VOL_1129, ['--level', '1', '--oversample', '2']),
            'C': (os.path.join(d, 'mesh_C'), VOL_1129, ['--level', '1', '--oversample', '2'])}
    write_mesh(md, P, valid, arms['P'][0]); write_mesh(md, C, valid, arms['C'][0])
    if os.environ.get('V014_DRYRUN'):
        res['status'] = 'dryrun'
        return res
    for arm, (mdir, vol, extra) in arms.items():
        base = os.path.join(d, arm)
        if not all(os.path.exists(base + f'_ink_o{o:+d}.npy') for o in OFFS):
            if not os.path.exists(base + '.npy'):
                subprocess.check_call([mb.PY, os.path.join(mb.HERE, 'render_crop.py'), mdir, vol, base,
                                       '--rows', str(r0), str(r0 + WIN_H), '--cols', str(c0), str(c0 + WIN_W),
                                       '--layers', '161', '--smooth-normals', '1.0', '--cache-chunks', '1500'] + extra,
                                      stdout=subprocess.DEVNULL)
            subprocess.check_call([mb.PY, os.path.join(mb.HERE, 'infer_crop.py'), base, '--offsets'] + [str(o) for o in OFFS],
                                  stdout=subprocess.DEVNULL)
        res[arm] = {str(o): mb.eval_map(np.load(base + f'_ink_o{o:+d}.npy'), lab8, e0) for o in OFFS}
        rj = base + '.json'
        if os.path.exists(rj):
            res[arm]['render'] = json.load(open(rj))
        if os.path.exists(base + '.npy'):
            os.remove(base + '.npy')                     # keep ink maps, drop the big stack
    res['status'] = 'ok'
    json.dump(res, open(rp, 'w'), indent=1)
    return res


def main():
    Mpub, Mlsq, tinfo = transforms()
    os.makedirs(os.path.join(RUN, 'payload'), exist_ok=True)
    summary = {'transforms': tinfo, 'segments': {}}
    for seg in sys.argv[1:]:
        r = run(seg, Mpub, Mlsq)
        summary['segments'][seg] = r
        if r['status'] == 'dryrun':
            print(seg, 'window', r['window_cells'], 'disp_um', {k: round(v, 1) for k, v in r['displacement_um'].items()},
                  'dens', round(r['density_window'], 3), '/', round(r['density_unrestricted'], 3), flush=True)
        elif r['status'] == 'ok':
            print(seg, ' '.join(f"{a} {r[a]['0']['auc']:.4f}" for a in 'RPC'), 'disp_med_um', round(r['displacement_um']['median'], 1), flush=True)
        else:
            print(seg, r['status'], flush=True)
        json.dump(summary, open(os.path.join(RUN, 'payload', 'v014.json'), 'w'), indent=1)
    ok = {s: r for s, r in summary['segments'].items() if r['status'] == 'ok' and not any(np.isnan(r[a]['0']['auc']) for a in 'PC')}
    n = len(ok)
    gain = [r['C']['0']['auc'] - r['P']['0']['auc'] for r in ok.values()]
    wins = sum(g > 0 for g in gain); losses = sum(g < 0 for g in gain)
    if n < 4:
        verdict = 'UNKNOWN'
    elif np.mean(gain) >= 0.02 and wins >= n - 1 and losses < 2:
        verdict = 'PASS'
    else:
        verdict = 'FAIL'
    summary['verdict'] = {'computable': n, 'mean_gain_C_minus_P': float(np.mean(gain)) if gain else None,
                          'wins': int(wins), 'losses': int(losses), 'verdict': verdict}
    json.dump(summary, open(os.path.join(RUN, 'payload', 'v014.json'), 'w'), indent=1)
    print('VERDICT', summary['verdict'], flush=True)


if __name__ == '__main__':
    main()
