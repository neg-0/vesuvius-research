"""V-013 label-free selection on the fine sweep (rules R1-R4 in the protocol).
Scores with the reporter's eval_map logic applied to stored ds8 maps (eval_map itself
downsamples by 8 first, so this is the same computation up to float16 storage)."""
import glob, json, os, sys
import numpy as np
from scipy import ndimage

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'sweep')
REP = json.load(open(sys.argv[1]))  # reporter's C2_march_scan_8_segments.json
OFFS = list(range(-48, 49, 4))


def auc(score, pos, neg):
    sp, sn = score[pos], score[neg]
    if sp.size < 50 or sn.size < 50:
        return float('nan')
    rng = np.random.default_rng(0)
    sp = rng.choice(sp, min(sp.size, 200000), replace=False); sn = rng.choice(sn, min(sn.size, 200000), replace=False)
    allv = np.concatenate([sp, sn]); rk = allv.argsort(kind='mergesort').argsort() + 1
    return float((rk[:sp.size].sum() - sp.size * (sp.size + 1) / 2) / (sp.size * sn.size))


def eval_ds8(t, lab8, e0, R_=48, DS=8):
    h, w = t.shape; ey, ex = e0; best = (-2, 0, 0)
    for dy in range(-R_, R_ + 1, 2):
        for dx in range(-R_, R_ + 1, 2):
            y, x = ey + dy, ex + dx
            if y < 0 or x < 0 or y + h > lab8.shape[0] or x + w > lab8.shape[1]:
                continue
            L = lab8[y:y + h, x:x + w]
            if L.std() == 0:
                continue
            a, b = t - t.mean(), L - L.mean()
            r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
            if r > best[0]:
                best = (r, dy, dx)
    r, dy, dx = best
    L = lab8[ey + dy:ey + dy + h, ex + dx:ex + dx + w]
    near = ndimage.distance_transform_edt(L < 0.5) <= 2.0 / (2.399e-3 * DS)
    return auc(t, L >= 0.5, near & (L < 0.05))


out = {}
for f in sorted(glob.glob(f'{R}/*_ds8_maps.npz')):
    seg = os.path.basename(f).replace('_ds8_maps.npz', '')
    z = np.load(f); lab8 = z['lab8_window'].astype(np.float32); e0 = tuple(int(v) for v in z['e0'])
    M = np.stack([z[f'o{o:+d}'].astype(np.float32) for o in OFFS])
    frac = (M > 0.5).mean((1, 2)); mean = M.mean((1, 2))
    r1 = OFFS[int(np.argmax(frac))]; r2 = OFFS[int(np.argmax(mean))]
    W = M.shape[2]; comp4 = np.zeros_like(M[0]); r4 = []
    for j in range(4):
        sl = slice(j * W // 4, (j + 1) * W // 4)
        k = int(np.argmax((M[:, :, sl] > 0.5).mean((1, 2)))); r4.append(OFFS[k]); comp4[:, sl] = M[k, :, sl]
    rep = REP[seg]
    out[seg] = {
        'R1': {'offset': r1, 'auc': eval_ds8(M[OFFS.index(r1)], lab8, e0)},
        'R2': {'offset': r2, 'auc': eval_ds8(M[OFFS.index(r2)], lab8, e0)},
        'R3': {'auc': eval_ds8(M.max(0), lab8, e0)},
        'R4': {'offsets': r4, 'auc': eval_ds8(comp4, lab8, e0)},
        'oracle': max(eval_ds8(M[i], lab8, e0) for i in range(len(OFFS))),
        'reporter_labelfree': rep['march_pick_conf']['auc'], 'reporter_asis': rep['march_mesh_as_is']['auc'],
        'reporter_ref2399': rep['reference_2399']['auc'],
    }
    print(seg[-18:], {k: (round(v['auc'], 4) if isinstance(v, dict) else round(v, 4)) for k, v in out[seg].items()}, 'R1 off', r1, flush=True)
mean = {k: float(np.mean([out[s][k]['auc'] if isinstance(out[s][k], dict) else out[s][k] for s in out])) for k in ('R1', 'R2', 'R3', 'R4', 'oracle', 'reporter_labelfree', 'reporter_asis', 'reporter_ref2399')}
print('MEAN', {k: round(v, 4) for k, v in mean.items()})
json.dump({'per_segment': out, 'mean': mean}, open(os.path.join(os.environ.get('PUZZLE_RUN_DIR', '.'), 'v013-select.json'), 'w'), indent=1)
