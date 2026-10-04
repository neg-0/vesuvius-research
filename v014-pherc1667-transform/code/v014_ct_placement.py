"""V-014 addendum: model-free placement check. For each segment, render the central 8x16 grid cells of
the V-014 window as R (2.399 mesh on 2.399 volume), P (M_pub^-1) and C (M_lsq^-1) on the 1.129 volume,
and measure the normalised cross-correlation of the CT against R.
Runs inside the pinned cross-scan-ink-transfer src tree next to v014_transform_ink.py (reuses its
window selection with V014_DRYRUN=1).
usage: v014_ct_placement.py <segment> [...]
"""
import json
import os
import subprocess
import sys

import numpy as np

os.environ['V014_DRYRUN'] = '1'
import v014_transform_ink as v  # noqa: E402

CH, CW, LAYERS, SLAB = 8, 16, 61, 21


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def main():
    Mpub, Mlsq, tinfo = v.transforms()
    out = {'transforms': tinfo, 'segments': {}}
    for seg in sys.argv[1:]:
        r = v.run(seg, Mpub, Mlsq)
        if r['status'] != 'dryrun':
            out['segments'][seg] = {'status': r['status']}
            continue
        r0, r1, c0, c1 = r['window_cells']
        rr, cc = (r0 + r1) // 2 - CH // 2, (c0 + c1) // 2 - CW // 2
        d = os.path.join(v.OUT, seg)
        arms = {'R': (os.path.join(d, 'mesh_2399'), v.VOL_2399, ['--level', '0', '--oversample', '1']),
                'P': (os.path.join(d, 'mesh_P'), v.VOL_1129, ['--level', '1', '--oversample', '2']),
                'C': (os.path.join(d, 'mesh_C'), v.VOL_1129, ['--level', '1', '--oversample', '2'])}
        vol = {}
        for a, (mdir, url, extra) in arms.items():
            base = os.path.join(d, f'place_{a}')
            if not os.path.exists(base + '.npy'):
                subprocess.check_call([v.mb.PY, os.path.join(v.mb.HERE, 'render_crop.py'), mdir, url, base,
                                       '--rows', str(rr), str(rr + CH), '--cols', str(cc), str(cc + CW),
                                       '--layers', str(LAYERS), '--smooth-normals', '1.0'] + extra, stdout=subprocess.DEVNULL)
            vol[a] = np.load(base + '.npy').astype(np.float32)
        R = vol['R']; L0 = (LAYERS - SLAB) // 2; m = 12
        H, W = R.shape[1:]
        ref = R[L0:L0 + SLAB, m:H - m, m:W - m]
        res = {'crop_cells': [rr, rr + CH, cc, cc + CW], 'valid_frac_R': float((R > 0).any(0).mean())}
        for a in 'PC':
            X = vol[a]
            z = ncc(ref, X[L0:L0 + SLAB, m:H - m, m:W - m])
            best = (-2.0, 0, 0, 0)
            for dz in range(-20, 21):
                for dy in range(-m, m + 1, 2):
                    for dx in range(-m, m + 1, 2):
                        s = ncc(ref, X[L0 + dz:L0 + dz + SLAB, m + dy:H - m + dy, m + dx:W - m + dx])
                        if s > best[0]:
                            best = (s, dz, dy, dx)
            res[a] = {'ncc0': z, 'best_ncc': best[0], 'best_shift_layers_y_x': list(best[1:]),
                      'valid_frac': float((X > 0).any(0).mean())}
        res['status'] = 'ok'
        out['segments'][seg] = res
        print(seg, 'NCC0 P %.3f C %.3f | best P %.3f %s C %.3f %s' % (res['P']['ncc0'], res['C']['ncc0'], res['P']['best_ncc'],
              res['P']['best_shift_layers_y_x'], res['C']['best_ncc'], res['C']['best_shift_layers_y_x']), flush=True)
        os.makedirs(os.path.join(v.RUN, 'payload'), exist_ok=True)
        json.dump(out, open(os.path.join(v.RUN, 'payload', 'v014-ct-placement.json'), 'w'), indent=1)
    ok = [s for s in out['segments'].values() if s['status'] == 'ok']
    fail = any(s['P']['ncc0'] >= s['C']['ncc0'] for s in ok)
    strong = sum(s['C']['ncc0'] >= 0.4 for s in ok)
    verdict = 'UNKNOWN' if len(ok) < 6 and not fail else ('FAIL' if fail else ('PASS' if strong >= 5 else 'FAIL'))
    out['verdict'] = {'computable': len(ok), 'C_ge_0.4': strong, 'any_P_ge_C': fail, 'verdict': verdict}
    json.dump(out, open(os.path.join(v.RUN, 'payload', 'v014-ct-placement.json'), 'w'), indent=1)
    print('VERDICT', out['verdict'], flush=True)


if __name__ == '__main__':
    main()
