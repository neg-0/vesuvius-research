"""B-001 read and rule: the stock ink_9um model (S0) and the B-001 fine-tune (REAL) on the held-out scrolls' labelled
test windows, then the pre-registered rule.

usage: b001_read.py read <ckpt dir>       (GPU desktop; resumable. <ckpt dir>/b001_real.pth from b001_train.py)
       b001_read.py rule                  (CPU: AUCs, RULE / VERDICT lines, results/b001/{scores,summary}.json, RESULT.md)
       b001_read.py --check               (CPU: AUC and rule maths on synthetic maps)

Read (fixed in experiments/b001/README.md): per window, the LR surface-volume crop from b001_data.py fetch (28 layers,
dz applied) through villa infer_single_zarr with --overlap 0.5 --blend-mode hann --batch-size 4 (v026_read's flags),
forward and reverse, for S0 (V026_CKPT, sha256 checked) and REAL; REAL also on the layer-shuffled stack
(v026_read.shuffled_layers, default_rng(0).permutation(17)), forward. Maps: $B001_DATA/read/<arm>/<window>_<dir>.png
(uint8, ds1). Scoring: pixels CT-valid (centre layer > 0) and supervised, positives = labelled ink; a window is scored
with >= 200 positives and >= 200 negatives. Rule: README "Pre-registered rule".
Env: V026_VILLA_SRC, V026_CKPT, B001_DATA, B001_RESULTS, V026_AMP.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', 'v026'), HERE):
    sys.path.insert(0, p)
import b001_data as bd                                              # noqa: E402

MIN_PX = 200
GAIN_MIN, WIN_FRAC, N_MIN, SCROLL_MIN_N = 0.05, 0.70, 20, 3
CONTROL_TOL = 0.10
MAPS = [('S0', 'fwd'), ('S0', 'rev'), ('REAL', 'fwd'), ('REAL', 'rev'), ('REAL_shuf', 'fwd')]


def committed(path):
    root = subprocess.run(['git', '-C', HERE, 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip()
    rel = os.path.relpath(path, root)
    st = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--', rel], capture_output=True, text=True).stdout
    tracked = subprocess.run(['git', '-C', root, 'ls-files', '--error-unmatch', rel], capture_output=True).returncode == 0
    if not tracked or st.strip():
        raise SystemExit(f'{rel} must be committed and unchanged before read/rule (git status: {st.strip() or "untracked"})')


def windows():
    p = os.path.join(bd.RESULTS, 'windows.json')
    committed(p)
    return json.load(open(p))['windows']


def map_path(arm, wid, d):
    return os.path.join(bd.DATA, 'read', arm, f'{wid}_{d}.png')


def read(ckpt_dir):
    """B-001: S0 and REAL, forward and reverse, plus REAL on the layer-shuffled stack (forward)."""
    real = os.path.join(ckpt_dir, 'b001_real.pth')
    read_arms([('S0', os.environ['V026_CKPT'], ('fwd', 'rev'), False), ('REAL', real, ('fwd', 'rev'), True)])


def infer_map(model, stack, direction, tmp, tag):
    """One villa infer_single_zarr pass (v026_read's flags) of a (layers, H, W) uint8 stack -> (H, W) uint8 map.
    model = v026_read.load_model()[:4]. Shared with experiments/b001g."""
    import tifffile
    from pathlib import Path
    import v026_read as vr
    inf, args, cm, dev = model
    zp = os.path.join(tmp, f'{tag}.zarr')
    shutil.rmtree(zp, ignore_errors=True)
    vr.write_zarr(zp, stack)
    tif = Path(tmp) / f'{tag}.tif'
    inf.infer_single_zarr(args=args, input_zarr=zp, configured_model=cm, device=dev, output_tiff=tif,
                          layer_direction=direction)
    im = tifffile.imread(str(tif))
    assert im.shape == stack.shape[1:] and im.dtype == np.uint8, (im.shape, im.dtype)
    os.remove(tif)
    shutil.rmtree(zp, ignore_errors=True)
    return im


def read_arms(arms, only=None):
    """arms: (name, checkpoint, directions, shuffled control). Maps -> map_path(name | name_shuf, window, direction);
    resumable (existing maps are skipped). only: optional window ids to read. Shared with b001r and b001s."""
    import cv2
    import v026_read as vr
    W = windows()
    base = os.environ['V026_CKPT']
    assert vr.sha256(base) == vr.CKPT_SHA256, 'unexpected S0 checkpoint'
    t0 = time.time()
    for arm, ckpt, dirs, shuf in arms:
        want = [(arm, d) for d in dirs] + ([(arm + '_shuf', 'fwd')] if shuf else [])
        todo = [w for w in W if (only is None or w in only) and not all(os.path.exists(map_path(a, w, d)) for a, d in want)]
        if not todo:
            continue
        os.environ['V026_CKPT'] = ckpt
        try:
            inf, args, cm, dev, info = vr.load_model()
        finally:
            os.environ['V026_CKPT'] = base
        print('model', arm, json.dumps({k: info[k] for k in ('checkpoint', 'input_depth', 'amp', 'device')}), flush=True)
        for i, wid in enumerate(todo):
            ct = np.load(os.path.join(bd.DATA, 'windows', wid + '.npy'))
            assert ct.shape[0] == bd.NL, ct.shape
            order, _ = vr.shuffled_layers(ct.shape[0], inf, cm.input_depth)
            jobs = [(arm, ct, {'fwd': 'forward', 'rev': 'reverse'}[d], d) for d in dirs]
            if shuf:
                jobs.append((arm + '_shuf', ct[order], 'forward', 'fwd'))
            tmp = os.path.join(bd.DATA, 'read', 'tmp')
            os.makedirs(tmp, exist_ok=True)
            for a, stack, direction, d in jobs:
                png = map_path(a, wid, d)
                if os.path.exists(png):
                    continue
                os.makedirs(os.path.dirname(png), exist_ok=True)
                im = infer_map((inf, args, cm, dev), stack, direction, tmp, f'{a}_{d}')
                cv2.imwrite(png + '.tmp.png', im)
                os.replace(png + '.tmp.png', png)
            print(f'{arm} {i + 1}/{len(todo)} {wid} {time.time() - t0:.0f}s', flush=True)   # no AUC on stdout
    print('READ done', len(W), 'windows', f'{time.time() - t0:.0f}s', flush=True)


def auc(score, pos, mask):
    from scipy.stats import rankdata
    s, y = score[mask].astype(np.float64), pos[mask]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < MIN_PX or n0 < MIN_PX:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rule_lines(rows):
    """rows: scored windows with 'scroll', 'S0', 'REAL'. Returns (verdict, lines)."""
    n = len(rows)
    gains = np.array([r['REAL'] - r['S0'] for r in rows]) if rows else np.zeros(0)
    wins = int((gains > 0).sum())
    lines = [f'RULE mean paired gain {gains.mean() if n else float("nan"):+.4f} (need >= +{GAIN_MIN}); REAL > S0 on {wins} '
             f'of {n} windows (need {WIN_FRAC:.0%}); n {n} (need {N_MIN})']
    per_ok = True
    for sc in sorted({r['scroll'] for r in rows}):
        g = np.array([r['REAL'] - r['S0'] for r in rows if r['scroll'] == sc])
        cond = len(g) < SCROLL_MIN_N or g.mean() >= 0
        per_ok &= cond
        lines.append(f'RULE {sc}: {len(g)} windows, mean gain {g.mean():+.4f}'
                     + ('' if len(g) >= SCROLL_MIN_N else f' (under {SCROLL_MIN_N} windows: not in condition 3)')
                     + ('' if cond else ' (< 0: condition 3 fails)'))
    if n < N_MIN:
        return 'UNKNOWN', lines
    ok = gains.mean() >= GAIN_MIN and wins >= WIN_FRAC * n and per_ok
    return ('PASS' if ok else 'FAIL'), lines


def rule():
    import cv2
    W = windows()
    rows, allrows, shuf = [], [], {'s': [], 'y': []}
    missing = [(a, w, d) for w in W for a, d in MAPS if not os.path.exists(map_path(a, w, d))]
    for wid, w in W.items():
        ct = np.load(os.path.join(bd.DATA, 'windows', wid + '.npy'))
        valid = ct[bd.NL // 2] > 0
        ink, sup = bd.load_labels('window', wid)
        lab = np.load(os.path.join(bd.RESULTS, 'window_labels', wid + '.npz'))
        nn = int(np.prod(lab['shape']))
        g_ink = np.unpackbits(lab['ink_gridindex'])[:nn].reshape(lab['shape']).astype(bool)
        g_sup = np.unpackbits(lab['supervision_gridindex'])[:nn].reshape(lab['shape']).astype(bool)
        m = valid & sup
        r = {'window': wid, 'scroll': w['scroll'], 'segment': w['segment'], 'pos_px': int((ink & m).sum()),
             'neg_px': int((~ink & m).sum())}
        maps = {}
        for a, d in MAPS:
            pth = map_path(a, wid, d)
            if os.path.exists(pth):
                maps[(a, d)] = cv2.imread(pth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        for (a, d), mp in maps.items():
            r[f'{a}_{d}'] = auc(mp, ink, m)
        if ('S0', 'fwd') in maps and ('S0', 'rev') in maps:
            r['S0_max'] = auc(np.maximum(maps[('S0', 'fwd')], maps[('S0', 'rev')]), ink, m)
            r['S0_fwd_gridindex_labels'] = auc(maps[('S0', 'fwd')], g_ink, valid & g_sup)
        if ('REAL', 'fwd') in maps and ('REAL', 'rev') in maps:
            r['REAL_max'] = auc(np.maximum(maps[('REAL', 'fwd')], maps[('REAL', 'rev')]), ink, m)
        r['scored'] = r.get('S0_fwd') is not None and r.get('REAL_fwd') is not None
        allrows.append(r)
        if r['scored']:
            rows.append({'scroll': w['scroll'], 'S0': r['S0_fwd'], 'REAL': r['REAL_fwd']})
            if ('REAL_shuf', 'fwd') in maps:
                shuf['s'].append(maps[('REAL_shuf', 'fwd')][m])
                shuf['y'].append(ink[m])
    verdict, lines = rule_lines(rows)
    ctrl = None
    if shuf['s']:
        s, y = np.concatenate(shuf['s']), np.concatenate(shuf['y'])
        ctrl = auc(s, y, np.ones_like(y))
    lines.append(f'CONTROL REAL shuffled-layer pooled AUC {ctrl if ctrl is None else round(ctrl, 4)} '
                 f'(need 0.5 +- {CONTROL_TOL})')
    if missing or ctrl is None or abs(ctrl - 0.5) > CONTROL_TOL:
        verdict = 'INVALID'
        if missing:
            lines.append(f'INVALID: {len(missing)} maps missing, e.g. {missing[0]}')
    lines.append(f'VERDICT {verdict}')
    for ln in lines:
        print(ln, flush=True)
    desc = {}
    for key in ('S0_fwd', 'REAL_fwd', 'S0_rev', 'REAL_rev', 'S0_max', 'REAL_max', 'S0_fwd_gridindex_labels'):
        v = [r[key] for r in allrows if r['scored'] and r.get(key) is not None]
        desc[key] = round(float(np.mean(v)), 4) if v else None
    summary = {'verdict': verdict, 'lines': lines, 'n_windows': len(W), 'n_scored': len(rows),
               'mean_auc_scored': desc, 'control_shuffled_auc': ctrl}
    bd.jsave(allrows, os.path.join(bd.RESULTS, 'scores.json'))
    bd.jsave(summary, os.path.join(bd.RESULTS, 'summary.json'))
    with open(os.path.join(bd.RESULTS, 'RESULT.md'), 'w') as fh:
        fh.write(f'# B-001: real-scan distillation for the eligible ~9 µm scan types: {verdict}\n\n'
                 'Spec: `experiments/b001/README.md`. Lines printed by `b001_read.py rule`:\n\n')
        fh.writelines(f'- `{ln}`\n' for ln in lines)
        fh.write('\n## Mean AUC over scored windows (descriptive)\n\n| map | mean AUC |\n| --- | --- |\n')
        fh.writelines(f'| {k} | {v} |\n' for k, v in desc.items())
        fh.write('\n## Per window\n\n| window | scroll | S0 fwd | REAL fwd | gain | S0 fwd, grid-index labels | REAL shuffled |'
                 ' pos / neg px |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        f4 = lambda x: '' if x is None else f'{x:.4f}'                                     # noqa: E731
        for r in allrows:
            g = (r['REAL_fwd'] - r['S0_fwd']) if r['scored'] else None
            fh.write(f"| {r['window']} | {r['scroll']} | {f4(r.get('S0_fwd'))} | {f4(r.get('REAL_fwd'))} | "
                     f"{'' if g is None else f'{g:+.4f}'} | {f4(r.get('S0_fwd_gridindex_labels'))} | "
                     f"{f4(r.get('REAL_shuf_fwd'))} | {r['pos_px']} / {r['neg_px']} |\n")
    return verdict


def check():
    rng = np.random.default_rng(0)
    y = np.zeros((100, 100), bool)
    y[:, :40] = True
    s = rng.random((100, 100)) + 0.5 * y
    a = auc(s, y, np.ones_like(y))
    assert 0.8 < a < 0.95, a
    assert abs(auc(rng.random((100, 100)), y, np.ones_like(y)) - 0.5) < 0.03
    assert auc(s, y, np.zeros_like(y)) is None
    rows = [{'scroll': 'A' if i < 12 else 'B', 'S0': 0.7, 'REAL': 0.78 if i % 5 else 0.69} for i in range(24)]
    v, lines = rule_lines(rows)
    assert v == 'PASS', lines
    rows[0]['REAL'] = 0.2
    rows[1]['REAL'] = 0.2
    v2, lines2 = rule_lines([r for r in rows if r['scroll'] == 'A'] * 2)
    assert v2 == 'FAIL', lines2
    assert rule_lines(rows[:5])[0] == 'UNKNOWN'
    print('CHECK OK: auc', round(a, 4), '; rule PASS / FAIL / UNKNOWN cases', flush=True)


def main():
    a = sys.argv[1:]
    if a == ['--check']:
        check()
    elif len(a) == 2 and a[0] == 'read':
        read(a[1])
    elif a == ['rule']:
        rule()
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
