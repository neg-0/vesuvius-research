"""B-001r read and rule: B-001's seed replication (arm R1) and simulated-input ablation (arm SIM) on B-001's 49 held-out
windows. Spec: experiments/b001r/README.md (registered before any B-001r model output).

usage: b001r_read.py read <ckpt root>     (GPU; <ckpt root>/r1/b001_r1.pth and <ckpt root>/sim/b001_sim.pth; resumable)
       b001r_read.py rule                 (CPU: REPRO, R1, SIM lines -> results/b001r/{scores,summary}.json, RESULT.md)
       b001r_read.py --check              (CPU: rule maths on synthetic rows)

Maps (b001_read.read_arms, same flags as B-001): S0 forward (re-read on this machine for the REPRO check against
B-001's committed S0 AUCs); R1 and SIM forward and reverse plus their layer-shuffled control (forward).
Env: V026_VILLA_SRC, V026_CKPT, B001_DATA, B001_RESULTS (B-001's results; read only), B001R_RESULTS.
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

OUT = os.environ.get('B001R_RESULTS', os.path.join(HERE, '..', '..', 'results', 'b001r'))
REPRO_TOL = 0.005
SIM_MARGIN, SIM_FRAC = 0.02, 0.60
ARMS = {'R1': ('r1', 'b001_r1.pth'), 'SIM': ('sim', 'b001_sim.pth')}


def read(root):
    br.read_arms([('S0', os.environ['V026_CKPT'], ('fwd',), False)]
                 + [(a, os.path.join(root, d, f), ('fwd', 'rev'), True) for a, (d, f) in ARMS.items()])


def sim_lines(d):
    """d: per-window AUC(REAL, B-001) - AUC(SIM). Returns (outcome, lines)."""
    n = len(d)
    m = float(np.mean(d)) if n else float('nan')
    up, down = int((d > 0).sum()), int((d < 0).sum())
    line = (f'SIM REAL - SIM mean paired gain {m:+.4f}; REAL > SIM on {up} of {n}, SIM > REAL on {down} of {n} '
            f'(margin {SIM_MARGIN}, fraction {SIM_FRAC:.0%})')
    if m >= SIM_MARGIN and up >= SIM_FRAC * n:
        out = 'REAL_INPUT_MATTERS'
    elif m <= -SIM_MARGIN and down >= SIM_FRAC * n:
        out = 'SIM_BETTER'
    else:
        out = 'NO_CLEAR_DIFFERENCE'
    return out, [line, f'SIM_OUTCOME {out}']


def control(maps_s, maps_y):
    if not maps_s:
        return None
    s, y = np.concatenate(maps_s), np.concatenate(maps_y)
    return br.auc(s, y, np.ones_like(y))


def rule():
    import cv2
    W = br.windows()
    b001 = {r['window']: r for r in json.load(open(os.path.join(bd.RESULTS, 'scores.json')))}
    maps_needed = [('S0', 'fwd')] + [(a, d) for a in ARMS for d in ('fwd', 'rev')] + [(a + '_shuf', 'fwd') for a in ARMS]
    missing = [(a, w, d) for w in W for a, d in maps_needed if not os.path.exists(br.map_path(a, w, d))]
    allrows, shuf = [], {a: ([], []) for a in ARMS}
    for wid, w in W.items():
        ct = np.load(os.path.join(bd.DATA, 'windows', wid + '.npy'))
        ink, sup = bd.load_labels('window', wid)
        m = (ct[bd.NL // 2] > 0) & sup
        r = {'window': wid, 'scroll': w['scroll'], 'B001_S0_fwd': b001[wid]['S0_fwd'], 'B001_REAL_fwd': b001[wid]['REAL_fwd']}
        for a, d in maps_needed:
            pth = br.map_path(a, wid, d)
            if not os.path.exists(pth):
                continue
            mp = cv2.imread(pth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            r[f'{a}_{d}'] = br.auc(mp, ink, m)
            if a.endswith('_shuf') and b001[wid]['scored']:
                shuf[a[:-5]][0].append(mp[m])
                shuf[a[:-5]][1].append(ink[m])
        r['scored'] = bool(b001[wid]['scored']) and all(r.get(f'{a}_fwd') is not None for a in ('S0',) + tuple(ARMS))
        allrows.append(r)
    sc = [r for r in allrows if r['scored']]
    lines = []
    diffs = [abs(r['S0_fwd'] - r['B001_S0_fwd']) for r in sc]
    repro_ok = bool(diffs) and max(diffs) <= REPRO_TOL
    lines.append(f'REPRO S0 forward AUC vs B-001 committed: max abs diff {max(diffs) if diffs else float("nan"):.5f} '
                 f'on {len(diffs)} windows (tol {REPRO_TOL}) -> {"OK" if repro_ok else "FAIL"}')
    # R1: B-001's rule with the seed-4501 model in place of REAL (S0 = B-001's committed S0, as registered)
    r1_verdict, r1_lines = br.rule_lines([{'scroll': r['scroll'], 'S0': r['B001_S0_fwd'], 'REAL': r['R1_fwd']} for r in sc])
    lines += ['R1 ' + ln for ln in r1_lines]
    ctrl = {a: control(*shuf[a]) for a in ARMS}
    for a in ARMS:
        lines.append(f'CONTROL {a} shuffled-layer pooled AUC {None if ctrl[a] is None else round(ctrl[a], 4)} '
                     f'(need 0.5 +- {br.CONTROL_TOL})')
    ctrl_ok = {a: ctrl[a] is not None and abs(ctrl[a] - 0.5) <= br.CONTROL_TOL for a in ARMS}
    if not ctrl_ok['R1']:
        r1_verdict = 'INVALID'
    lines.append(f'R1_VERDICT {r1_verdict} -> B-001 {"REPLICATED" if r1_verdict == "PASS" else "NOT_REPLICATED" if r1_verdict == "FAIL" else r1_verdict}')
    sim_out, s_lines = sim_lines(np.array([r['B001_REAL_fwd'] - r['SIM_fwd'] for r in sc]))
    if not ctrl_ok['SIM']:
        s_lines[-1] = 'SIM_OUTCOME INVALID (control)'
        sim_out = 'INVALID'
    lines += s_lines
    if missing or not repro_ok:
        lines.append(f'INVALID: {"REPRO FAIL" if not repro_ok else ""} {len(missing)} maps missing'
                     + (f', e.g. {missing[0]}' if missing else ''))
        r1_verdict = sim_out = 'INVALID'
    # descriptive
    desc = {}
    for k in ('B001_S0_fwd', 'B001_REAL_fwd', 'S0_fwd', 'R1_fwd', 'R1_rev', 'SIM_fwd', 'SIM_rev'):
        v = [r[k] for r in sc if r.get(k) is not None]
        desc[k] = round(float(np.mean(v)), 4) if v else None
    if sc:
        g_real = np.array([r['B001_REAL_fwd'] - r['B001_S0_fwd'] for r in sc])
        g_r1 = np.array([r['R1_fwd'] - r['B001_S0_fwd'] for r in sc])
        desc['gain_correlation_REAL_R1'] = round(float(np.corrcoef(g_real, g_r1)[0, 1]), 4)
        _, sim_vs_s0 = br.rule_lines([{'scroll': r['scroll'], 'S0': r['B001_S0_fwd'], 'REAL': r['SIM_fwd']} for r in sc])
        desc['SIM_vs_S0_B001_rule'] = sim_vs_s0
    for ln in lines:
        print(ln, flush=True)
    print('DESCR', json.dumps(desc), flush=True)
    os.makedirs(OUT, exist_ok=True)
    summary = {'r1_verdict': r1_verdict, 'sim_outcome': sim_out, 'lines': lines, 'n_scored': len(sc), 'descriptive': desc,
               'control_shuffled_auc': ctrl}
    bd.jsave(allrows, os.path.join(OUT, 'scores.json'))
    bd.jsave(summary, os.path.join(OUT, 'summary.json'))
    with open(os.path.join(OUT, 'RESULT.md'), 'w') as fh:
        fh.write(f'# B-001r: seed replication {r1_verdict}, simulated-input ablation {sim_out}\n\n'
                 'Spec: `experiments/b001r/README.md`. Lines printed by `b001r_read.py rule`:\n\n')
        fh.writelines(f'- `{ln}`\n' for ln in lines)
        fh.write('\n## Descriptive\n\n```json\n' + json.dumps(desc, indent=1) + '\n```\n')
        fh.write('\n## Per window (forward AUC)\n\n| window | scroll | S0 (B-001) | REAL (B-001) | R1 | SIM | S0 re-read |\n'
                 '| --- | --- | --- | --- | --- | --- | --- |\n')
        f4 = lambda x: '' if x is None else f'{x:.4f}'                                     # noqa: E731
        for r in allrows:
            fh.write(f"| {r['window']} | {r['scroll']} | {f4(r['B001_S0_fwd'])} | {f4(r['B001_REAL_fwd'])} | "
                     f"{f4(r.get('R1_fwd'))} | {f4(r.get('SIM_fwd'))} | {f4(r.get('S0_fwd'))} |\n")
    return r1_verdict, sim_out


def check():
    d = np.array([0.03] * 40 + [-0.01] * 9)
    assert sim_lines(d)[0] == 'REAL_INPUT_MATTERS', sim_lines(d)
    assert sim_lines(-d)[0] == 'SIM_BETTER'
    assert sim_lines(np.array([0.01] * 49))[0] == 'NO_CLEAR_DIFFERENCE'
    assert sim_lines(np.array([0.10] * 20 + [-0.02] * 29))[0] == 'NO_CLEAR_DIFFERENCE'    # mean ok, fraction not
    print('CHECK OK: SIM outcomes REAL_INPUT_MATTERS / SIM_BETTER / NO_CLEAR_DIFFERENCE', flush=True)


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
