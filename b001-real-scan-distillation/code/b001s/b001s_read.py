"""B-001s read and rule: each held-out scroll's windows read with the fold model that never saw that scroll, compared
with B-001's two-seed reference. Spec: experiments/b001s/README.md (registered before any B-001s model output).

usage: b001s_read.py read <ckpt root>     (GPU; <ckpt root>/<scroll>/b001s.pth per fold; resumable)
       b001s_read.py rule                 (CPU -> results/b001s/{scores,summary}.json, RESULT.md)
       b001s_read.py --check              (CPU: rule maths on synthetic rows)
Env: V026_VILLA_SRC, V026_CKPT, B001_DATA, B001_RESULTS (= results/b001s).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', 'v026'), os.path.join(HERE, '..', 'b001')):
    sys.path.insert(0, p)
import b001_data as bd                                              # noqa: E402
import b001_read as br                                              # noqa: E402

RES = os.path.join(HERE, '..', '..', 'results')
FOLDS = ('PHerc0841', 'PHerc0009B', 'PHerc0500P2')
REPRO_TOL = 0.005
MARGIN, FRAC = 0.02, 0.60


def arm(sc):
    return 'F_' + sc


def read(root):
    W = br.windows()
    br.read_arms([('S0', os.environ['V026_CKPT'], ('fwd',), False)])
    for sc in FOLDS:
        only = {w for w, v in W.items() if v['scroll'] == sc}
        br.read_arms([(arm(sc), os.path.join(root, sc, 'b001s.pth'), ('fwd', 'rev'), True)], only=only)


def scale_lines(d):
    """d: per-window AUC(B-001s) - reference (mean of B-001 REAL and R1)."""
    n = len(d)
    m = float(np.mean(d)) if n else float('nan')
    up, down = int((d > 0).sum()), int((d < 0).sum())
    line = (f'SCALE B-001s - B-001 (two-seed mean) mean paired gain {m:+.4f}; higher on {up} of {n}, lower on {down} '
            f'of {n} (margin {MARGIN}, fraction {FRAC:.0%})')
    if m >= MARGIN and up >= FRAC * n:
        out = 'SCALE_HELPS'
    elif m <= -MARGIN and down >= FRAC * n:
        out = 'SCALE_HURTS'
    else:
        out = 'NO_CLEAR_DIFFERENCE'
    return out, [line, f'SCALE_OUTCOME {out}']


def rule():
    import cv2
    W = br.windows()
    b1 = {r['window']: r for r in json.load(open(os.path.join(RES, 'b001', 'scores.json')))}
    b1r = {r['window']: r for r in json.load(open(os.path.join(RES, 'b001r', 'scores.json')))}
    allrows, shuf, missing = [], {sc: ([], []) for sc in FOLDS}, []
    for wid, w in W.items():
        sc = w['scroll']
        need = [('S0', 'fwd'), (arm(sc), 'fwd'), (arm(sc), 'rev'), (arm(sc) + '_shuf', 'fwd')]
        missing += [(a, wid, d) for a, d in need if not os.path.exists(br.map_path(a, wid, d))]
        ct = np.load(os.path.join(bd.DATA, 'windows', wid + '.npy'))
        ink, sup = bd.load_labels('window', wid)
        m = (ct[bd.NL // 2] > 0) & sup
        r = {'window': wid, 'scroll': sc, 'B001_S0_fwd': b1[wid]['S0_fwd'], 'B001_REAL_fwd': b1[wid]['REAL_fwd'],
             'B001_R1_fwd': b1r[wid].get('R1_fwd')}
        for a, d in need:
            pth = br.map_path(a, wid, d)
            if not os.path.exists(pth):
                continue
            mp = cv2.imread(pth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            key = 'S0_fwd' if a == 'S0' else ('S_shuf_fwd' if a.endswith('_shuf') else f'S_{d}')
            r[key] = br.auc(mp, ink, m)
            if a.endswith('_shuf') and b1[wid]['scored']:
                shuf[sc][0].append(mp[m])
                shuf[sc][1].append(ink[m])
        r['scored'] = bool(b1[wid]['scored']) and r.get('S_fwd') is not None and r.get('B001_R1_fwd') is not None
        allrows.append(r)
    sc_rows = [r for r in allrows if r['scored']]
    lines = []
    diffs = [abs(r['S0_fwd'] - r['B001_S0_fwd']) for r in sc_rows if r.get('S0_fwd') is not None]
    repro_ok = len(diffs) == len(sc_rows) and bool(diffs) and max(diffs) <= REPRO_TOL
    lines.append(f'REPRO S0 forward AUC vs B-001 committed: max abs diff {max(diffs) if diffs else float("nan"):.5f} '
                 f'on {len(diffs)} windows (tol {REPRO_TOL}) -> {"OK" if repro_ok else "FAIL"}')
    v_s0, l_s0 = br.rule_lines([{'scroll': r['scroll'], 'S0': r['B001_S0_fwd'], 'REAL': r['S_fwd']} for r in sc_rows])
    lines += ['VS_S0 ' + ln for ln in l_s0]
    ctrl_ok = True
    for sc in FOLDS:
        c = br.auc(np.concatenate(shuf[sc][0]), np.concatenate(shuf[sc][1]), np.ones(sum(len(x) for x in shuf[sc][1]), bool)) \
            if shuf[sc][0] else None
        ok = c is not None and abs(c - 0.5) <= br.CONTROL_TOL
        ctrl_ok &= ok
        lines.append(f'CONTROL fold {sc} shuffled-layer pooled AUC {None if c is None else round(c, 4)} (need 0.5 +- {br.CONTROL_TOL})')
    lines.append(f'VS_S0_VERDICT {v_s0}')
    ref = np.array([(r['B001_REAL_fwd'] + r['B001_R1_fwd']) / 2 for r in sc_rows])
    s = np.array([r['S_fwd'] for r in sc_rows])
    out, s_lines = scale_lines(s - ref)
    lines += s_lines
    for sc in FOLDS:
        k = np.array([r['scroll'] == sc for r in sc_rows])
        if k.any():
            lines.append(f'SCALE {sc}: {int(k.sum())} windows, mean B-001s {s[k].mean():.4f} vs B-001 two-seed {ref[k].mean():.4f} '
                         f'({(s[k] - ref[k]).mean():+.4f})')
    if missing or not repro_ok or not ctrl_ok:
        out = v_s0 = 'INVALID'
        lines.append(f'INVALID: repro {"OK" if repro_ok else "FAIL"}, controls {"OK" if ctrl_ok else "FAIL"}, '
                     f'{len(missing)} maps missing' + (f', e.g. {missing[0]}' if missing else ''))
    lines.append(f'VERDICT {out} (vs stock: {v_s0})')
    desc = {k: round(float(np.mean([r[k] for r in sc_rows if r.get(k) is not None])), 4)
            for k in ('B001_S0_fwd', 'B001_REAL_fwd', 'B001_R1_fwd', 'S_fwd', 'S_rev', 'S0_fwd')} if sc_rows else {}
    for ln in lines:
        print(ln, flush=True)
    print('DESCR', json.dumps(desc), flush=True)
    summary = {'verdict': out, 'vs_s0': v_s0, 'lines': lines, 'n_scored': len(sc_rows), 'descriptive': desc}
    bd.jsave(allrows, os.path.join(bd.RESULTS, 'scores.json'))
    bd.jsave(summary, os.path.join(bd.RESULTS, 'summary.json'))
    with open(os.path.join(bd.RESULTS, 'RESULT.md'), 'w') as fh:
        fh.write(f'# B-001s: scaled, leave-one-scroll-out real-scan distillation: {out} (vs stock: {v_s0})\n\n'
                 'Spec: `experiments/b001s/README.md`. Lines printed by `b001s_read.py rule`:\n\n')
        fh.writelines(f'- `{ln}`\n' for ln in lines)
        fh.write('\n## Descriptive mean AUC\n\n```json\n' + json.dumps(desc, indent=1) + '\n```\n')
        fh.write('\n## Per window (forward AUC)\n\n| window | scroll | S0 | B-001 REAL | B-001 R1 | B-001s |\n'
                 '| --- | --- | --- | --- | --- | --- |\n')
        f4 = lambda x: '' if x is None else f'{x:.4f}'                                     # noqa: E731
        for r in allrows:
            fh.write(f"| {r['window']} | {r['scroll']} | {f4(r['B001_S0_fwd'])} | {f4(r['B001_REAL_fwd'])} | "
                     f"{f4(r['B001_R1_fwd'])} | {f4(r.get('S_fwd'))} |\n")
    return out


def check():
    assert scale_lines(np.array([0.03] * 40 + [-0.01] * 9))[0] == 'SCALE_HELPS'
    assert scale_lines(np.array([-0.03] * 40 + [0.01] * 9))[0] == 'SCALE_HURTS'
    assert scale_lines(np.array([0.01] * 49))[0] == 'NO_CLEAR_DIFFERENCE'
    assert scale_lines(np.array([0.10] * 20 + [-0.02] * 29))[0] == 'NO_CLEAR_DIFFERENCE'
    print('CHECK OK: SCALE outcomes SCALE_HELPS / SCALE_HURTS / NO_CLEAR_DIFFERENCE', flush=True)


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
