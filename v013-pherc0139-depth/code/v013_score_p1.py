"""V-013 cheap check: CT-only predictor P1 vs reporter's label-best sub-window depths."""
import json, os, sys
import numpy as np
from scipy import ndimage
DATA = os.environ['MARCH_BENCH_DATA']
best = json.load(open('../../ext/cross-scan-ink-transfer/results/C2_march_scan_local_depth.json'))
WIN = np.array([-40, -20, 0, 20, 40])
rows, out = [], {}
for seg, v in best.items():
    p = os.path.join(DATA, seg, 'march_flat.npy')
    if not os.path.exists(p):
        out[seg] = 'missing'; continue
    F = np.load(p, mmap_mode='r'); L, H, W = F.shape; c = (L - 1) // 2
    preds = []
    for j in range(4):
        prof = np.asarray(F[:, ::4, j * W // 4:(j + 1) * W // 4:4], np.float32).mean((1, 2))
        prof = ndimage.gaussian_filter1d(prof, 3.0)
        ks = np.arange(L) - c; m = (ks >= -48) & (ks <= 48)
        k = int(ks[m][np.argmax(prof[m])]); preds.append(k)
        lab = v['subwindow_best_depth_um'][j]
        if lab is None:
            continue
        labl = int(round(lab / 2.4))
        snap = int(WIN[np.argmin(np.abs(WIN - k))])
        rows.append((seg, j, k, snap, labl, snap == labl, labl == 0))
    out[seg] = preds
tot = len(rows); p1 = sum(r[5] for r in rows); c0 = sum(r[6] for r in rows)
hold = [r for r in rows if 'w044' not in r[0]]
res = {'scorable': tot, 'p1_match': p1, 'const0_match': c0,
       'holdout_scorable': len(hold), 'holdout_p1': sum(r[5] for r in hold), 'holdout_const0': sum(r[6] for r in hold),
       'predictions_layers': out, 'rows': rows}
for r in rows: print(r[0][-18:], r[1], 'pred', r[2], 'snap', r[3], 'label', r[4], 'OK' if r[5] else '')
print({k: v for k, v in res.items() if k not in ('rows', 'predictions_layers')})
os.makedirs(os.environ.get('PUZZLE_RUN_DIR', '.') + '/payload', exist_ok=True)
json.dump(res, open(os.environ.get('PUZZLE_RUN_DIR', '.') + '/payload/v013-p1.json', 'w'), indent=1)
