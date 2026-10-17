"""V-033 driver: can a label-free statistic of the model's own ink maps choose the depth offset per window?

Spec: experiments/v033/README.md (registered before any V-033 statistic was computed). This script implements it and
does not change it. CPU only (numpy), no GPU, no network. It re-reads the nine ink maps per window that V-032 kept on
the desktop; nothing is rendered or inferred.

Stages:
  stats    Per window and offset: ds8 block-mean of the ink map, the std / top-5% mean / mean over the fully carried
           blocks, and the offset each chooser picks (argmax). Opens NO labels, NO AUCs and NOT sweep.json. Writes
           results/v033-depth-chooser/stats.json, which must be committed before `rule`.
  rule     Joins stats.json (committed, unchanged in git) with V-032's committed sweep.json: AUC at the chosen offset.
           Prints RULE / DESCR lines, writes summary.json and RESULT.md.
  check    selftest + windows present + maps present.

usage: v033_chooser.py check | stats | rule | selftest
Env: V033_MAPS (V-032's work folder: <seg>/win/<window>_sweep_ink_o<+/-N>.npy and <window>_carried.npy; i.e. the
V031_DATA used for item 47), V032_RESULTS (default projects/vesuvius/results/v032-depth-sweep), V033_RESULTS
(default projects/vesuvius/results/v033-depth-chooser).
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
V032 = os.environ.get('V032_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v032-depth-sweep'))
RESULTS = os.environ.get('V033_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v033-depth-chooser'))
MAPS = os.environ.get('V033_MAPS', '')
OFFSETS = [-32, -24, -16, -8, 0, 8, 16, 24, 32]
DS = 8
MIN_BLOCKS = 100                  # fewer fully carried ds8 blocks than this: the chooser keeps offset 0 (recorded)
TOP_FRAC = 0.05
CHOOSERS = ('std', 'top5', 'mean')
PRIMARY = 'std'
GAIN_FRAC = 0.5                   # R1: chosen-offset gain on L >= 0.5 x oracle gain on L
HARM_MAX = -0.01                  # R2: mean gain on W >= -0.01
WIN_MIN = 8                       # R3: chosen beats offset 0 on >= 8 of the 12 L windows


def log(*a):
    print(*a, flush=True)


def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def jsave(obj, p):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    json.dump(obj, open(p + '.part', 'w'), indent=1)
    os.replace(p + '.part', p)


def load_bits(path):
    shp = json.load(open(path + '.shape.json'))
    return np.unpackbits(np.load(path))[:shp[0] * shp[1]].reshape(shp).astype(bool)


def down(a, ds=DS):
    h, w = a.shape[0] // ds * ds, a.shape[1] // ds * ds
    return a[:h, :w].astype(np.float32).reshape(h // ds, ds, w // ds, ds).mean((1, 3), dtype=np.float32)


def pick(values):
    """argmax over the offsets; ties go to the smaller |offset|, then the more negative one."""
    return max(zip(OFFSETS, values), key=lambda x: (x[1], -abs(x[0]), -x[0]))[0]


def window_stats(maps, carried):
    """maps: {offset: 2-D ink map}; carried: bool mask of the same shape. Label-free."""
    V = down(carried.astype(np.float32)) == 1.0
    out = {'blocks': int(V.sum()), 'std': {}, 'top5': {}, 'mean': {}}
    for o in OFFSETS:
        m = np.asarray(maps[o])
        assert m.shape == carried.shape, (o, m.shape, carried.shape)
        assert np.isfinite(m).all(), f'non-finite ink map at offset {o}'
        M = down(m)[V] if V.sum() else np.zeros(0, np.float32)
        if V.sum() < MIN_BLOCKS:
            out['std'][str(o)] = out['top5'][str(o)] = out['mean'][str(o)] = None
            continue
        k = max(1, int(math.ceil(TOP_FRAC * M.size)))
        out['std'][str(o)] = float(M.std())
        out['top5'][str(o)] = float(np.partition(M, M.size - k)[M.size - k:].mean())
        out['mean'][str(o)] = float(M.mean())
    out['chosen'] = {}
    for c in CHOOSERS:
        v = [out[c][str(o)] for o in OFFSETS]
        out['chosen'][c] = 0 if any(x is None for x in v) else pick(v)
    return out


def sel():
    s = jload(os.path.join(V032, 'windows.json'))
    if s is None:
        raise SystemExit('V-032 windows.json not found')
    return s


def stats_stage():
    s = sel()
    names = s['sets']['L'] + s['sets']['W'] + s['sets']['D']
    res = jload(os.path.join(RESULTS, 'stats.json'), {'windows': {}})
    for n in names:
        if n in res['windows']:
            continue
        seg = s['source'][n]['segment']
        d = os.path.join(MAPS, seg, 'win')
        maps = {}
        for o in OFFSETS:
            p = os.path.join(d, f'{n}_sweep_ink_o{o:+d}.npy')
            if not os.path.exists(p):
                raise SystemExit(f'missing ink map {p}; stop and report')
            maps[o] = np.load(p)
        cp = os.path.join(d, f'{n}_carried.npy')
        if not os.path.exists(cp):
            raise SystemExit(f'missing carried mask {cp}; stop and report')
        res['windows'][n] = window_stats(maps, load_bits(cp))
        log('STATS', n, 'blocks', res['windows'][n]['blocks'], 'chosen', res['windows'][n]['chosen'])
    res['offsets_layers'] = OFFSETS
    res['choosers'] = list(CHOOSERS)
    jsave(res, os.path.join(RESULTS, 'stats.json'))
    log(f"STATS written for {len(res['windows'])} windows; commit stats.json before running rule")


def committed(path):
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    stt = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--', rel], capture_output=True, text=True).stdout
    tracked = subprocess.run(['git', '-C', ROOT, 'ls-files', '--error-unmatch', rel], capture_output=True).returncode == 0
    if not tracked or stt.strip():
        raise SystemExit(f'{rel} must be committed and unchanged (git status: {stt.strip() or "untracked"})')


def fmt(x):
    return 'n/a' if x is None or x != x else f'{x:.4f}'


def gains(rows, st, names, chooser):
    out = []
    for n in names:
        r = rows.get(n)
        if r is None or n not in st['windows']:
            return None
        o = st['windows'][n]['chosen'][chooser]
        a, a0 = r['auc'][str(o)], r['auc']['0']
        if a is None or a0 is None:
            return None
        out.append(a - a0)
    return out


def evaluate(rows, st, s):
    L, W, D = s['sets']['L'], s['sets']['W'], s['sets']['D']
    ev = {'choosers': {}}
    orc = []
    for n in L:
        r = rows.get(n)
        if r is None or r['best_auc'] is None or r['auc']['0'] is None:
            orc = None
            break
        orc.append(r['best_auc'] - r['auc']['0'])
    ev['oracle_gain_L'] = float(np.mean(orc)) if orc else None
    for c in CHOOSERS:
        gl, gw, gd = gains(rows, st, L, c), gains(rows, st, W, c), gains(rows, st, D, c)
        e = {'gain_L': None if gl is None else float(np.mean(gl)), 'gain_W': None if gw is None else float(np.mean(gw)),
             'gain_D': None if gd is None else float(np.mean(gd)),
             'wins_L': None if gl is None else int(sum(x > 0 for x in gl)),
             'chosen_L': [st['windows'][n]['chosen'][c] for n in L if n in st['windows']],
             'chosen_W': [st['windows'][n]['chosen'][c] for n in W if n in st['windows']],
             'chosen_D': [st['windows'][n]['chosen'][c] for n in D if n in st['windows']],
             'matches_oracle_offset': sum(st['windows'][n]['chosen'][c] == rows[n]['best_offset'] for n in L + W + D
                                          if n in st['windows'] and n in rows)}
        if None in (e['gain_L'], e['gain_W'], ev['oracle_gain_L']):
            e['verdict'], e['reason'] = 'UNKNOWN', 'a planned window is missing or has an undefined AUC'
        else:
            r1 = e['gain_L'] >= GAIN_FRAC * ev['oracle_gain_L']
            r2 = e['gain_W'] >= HARM_MAX
            r3 = e['wins_L'] >= WIN_MIN
            e['reason'] = (f"mean gain on loss windows {e['gain_L']:+.4f} (need >= {GAIN_FRAC} x oracle "
                           f"{ev['oracle_gain_L']:+.4f} = {GAIN_FRAC * ev['oracle_gain_L']:+.4f}); mean gain on control windows "
                           f"{e['gain_W']:+.4f} (need >= {HARM_MAX}); chosen beats offset 0 on {e['wins_L']} of {len(L)} loss windows "
                           f"(need >= {WIN_MIN})")
            e['verdict'] = 'PASS' if r1 and r2 and r3 else 'FAIL'
        ev['choosers'][c] = e
    base = {}
    for k, names in (('L', L), ('W', W), ('D', D)):
        g = [np.mean([rows[n]['auc'][str(o)] - rows[n]['auc']['0'] for o in OFFSETS]) for n in names if n in rows]
        base[k] = float(np.mean(g)) if g else None
    ev['random_offset_gain'] = base
    for k, names in (('L', L), ('W', W), ('D', D)):
        rs = [rows[n] for n in names if n in rows and rows[n]['auc']['0'] is not None]
        ev['mean_auc_' + k] = {'at_0': float(np.mean([r['auc']['0'] for r in rs])) if rs else None,
                               'oracle': float(np.mean([r['best_auc'] for r in rs])) if rs else None,
                               'mrg': float(np.mean([r['mrg20736_1um'] for r in rs])) if rs else None}
    ev['verdict'], ev['reason'] = ev['choosers'][PRIMARY]['verdict'], ev['choosers'][PRIMARY]['reason']
    return ev


def rule_stage():
    committed(os.path.join(RESULTS, 'stats.json'))
    committed(os.path.join(V032, 'sweep.json'))
    st = jload(os.path.join(RESULTS, 'stats.json'))
    rows = jload(os.path.join(V032, 'sweep.json'))
    s = sel()
    ev = evaluate(rows, st, s)
    p = ev['choosers'][PRIMARY]
    lines = [f"RULE label-free depth chooser (V-033, primary: {PRIMARY} of the ds8 ink map): {ev['reason']} -> {ev['verdict']}"]
    for c in CHOOSERS:
        e = ev['choosers'][c]
        lines.append(f"DESCR chooser {c}: gain L {fmt(e['gain_L'])}, W {fmt(e['gain_W'])}, D {fmt(e['gain_D'])}; chosen beats 0 on "
                     f"{e['wins_L']} of {len(s['sets']['L'])} L windows; equals the oracle offset on {e['matches_oracle_offset']} of 31; "
                     f"chosen offsets L {e['chosen_L']}")
    lines.append(f"DESCR oracle gain L {fmt(ev['oracle_gain_L'])}; random-offset gain L {fmt(ev['random_offset_gain']['L'])}, "
                 f"W {fmt(ev['random_offset_gain']['W'])}, D {fmt(ev['random_offset_gain']['D'])}")
    for k, nm in (('L', 'loss'), ('W', 'control'), ('D', 'PHerc0139 w044/w045')):
        a = ev['mean_auc_' + k]
        lines.append(f"DESCR {nm} set mean AUC: at 0 {fmt(a['at_0'])}, oracle {fmt(a['oracle'])}, mrg20736-1um {fmt(a['mrg'])}; "
                     f"primary chooser gives {fmt(None if a['at_0'] is None or p['gain_' + k] is None else a['at_0'] + p['gain_' + k])}")
    for ln in lines:
        log(ln)
    jsave({'verdict': ev['verdict'], 'evaluation': ev, 'lines': lines}, os.path.join(RESULTS, 'summary.json'))
    write_result(os.path.join(RESULTS, 'RESULT.md'), lines, ev)
    return ev


def write_result(path, lines, ev):
    keep = ''
    if os.path.exists(path):
        t = open(path).read()
        keep = t.split('<!-- prose -->', 1)[1] if '<!-- prose -->' in t else ''
    head = [f"# V-033: can a label-free statistic choose the depth offset per window? {ev['verdict']}", '',
            'Spec: `experiments/v033/README.md`. Driver: `experiments/v033/v033_chooser.py`. Lines printed by `rule`:', '']
    head += [f'- `{ln}`' for ln in lines]
    open(path, 'w').write('\n'.join(head) + '\n\n<!-- prose -->' + (keep or '\n'))


def selftest():
    rng = np.random.default_rng(3)
    H, W_ = 80, 160
    carried = np.ones((H, W_), bool); carried[:8] = False
    base = rng.random((H, W_)).astype(np.float32) * 0.2
    maps = {o: base * (1.0 + 0.1 * (o == 16)) + (0.5 * (np.indices((H, W_)).sum(0) % 2) if o == 16 else 0) for o in OFFSETS}
    st = window_stats(maps, carried)
    assert st['blocks'] == (H // DS - 1) * (W_ // DS), st['blocks']
    assert st['chosen']['std'] == 0 or st['chosen']['std'] in OFFSETS
    # ds8 block means wash out a pixel checkerboard: build a block-scale pattern instead
    pat = np.kron(rng.random((H // DS, W_ // DS)), np.ones((DS, DS))).astype(np.float32)
    maps = {o: base * 0.01 + (pat if o == -8 else 0.5 * pat) for o in OFFSETS}
    st = window_stats(maps, carried)
    assert st['chosen']['std'] == -8 and st['chosen']['top5'] == -8 and st['chosen']['mean'] == -8, st['chosen']
    assert pick([1, 1, 1, 1, 1, 1, 1, 1, 1]) == 0 and pick([0, 0, 0, 0, 0, 0, 0, 0, 1]) == 32
    assert pick([1, 0, 0, 0, 0, 0, 0, 0, 1]) == -32                       # equal |offset|: the more negative
    small = np.zeros((H, W_), bool); small[:16, :40] = True               # 10 blocks < MIN_BLOCKS: keep offset 0
    assert window_stats(maps, small)['chosen'] == {c: 0 for c in CHOOSERS}
    # bits round trip
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'm.npy')
        np.save(p, np.packbits(carried, axis=None)); json.dump(list(carried.shape), open(p + '.shape.json', 'w'))
        assert (load_bits(p) == carried).all()
    # rule
    s = {'sets': {'L': [f'l{i}' for i in range(12)], 'W': [f'w{i}' for i in range(12)], 'D': ['d0']}}

    def row(a0, best, mrg, bo=16, chosen_auc=None):
        au = {str(o): a0 - 0.02 for o in OFFSETS}; au['0'] = a0; au[str(bo)] = best
        return {'auc': au, 'best_auc': best, 'best_offset': bo, 'mrg20736_1um': mrg}
    rows = {f'l{i}': row(0.7, 0.8, 0.9) for i in range(12)} | {f'w{i}': row(0.9, 0.92, 0.85) for i in range(12)} | {'d0': row(0.7, 0.8, 0.9)}

    def stfor(offs_L, offs_W):
        w = {f'l{i}': {'chosen': {c: offs_L[i] for c in CHOOSERS}} for i in range(12)}
        w |= {f'w{i}': {'chosen': {c: offs_W[i] for c in CHOOSERS}} for i in range(12)}
        w['d0'] = {'chosen': {c: 16 for c in CHOOSERS}}
        return {'windows': w}
    ev = evaluate(rows, stfor([16] * 12, [0] * 12), s)
    assert ev['verdict'] == 'PASS' and abs(ev['oracle_gain_L'] - 0.1) < 1e-9, ev['reason']
    assert evaluate(rows, stfor([16] * 12, [16] * 12), s)['verdict'] == 'PASS'                   # W gain +0.02, no harm
    assert evaluate(rows, stfor([0] * 12, [0] * 12), s)['verdict'] == 'FAIL'                     # no gain
    assert evaluate(rows, stfor([16] * 4 + [0] * 8, [0] * 12), s)['verdict'] == 'FAIL'           # gain 0.033 < 0.05, 4 wins
    assert evaluate(rows, stfor([16] * 12, [8] * 12), s)['verdict'] == 'FAIL'                    # harm: W -0.02 < -0.01
    assert evaluate({k: v for k, v in rows.items() if k != 'l3'}, stfor([16] * 12, [0] * 12), s)['verdict'] == 'UNKNOWN'
    log('SELFTEST ok (window stats, chooser ties, small windows, bit masks, rule)')


def check():
    selftest()
    s = sel()
    names = s['sets']['L'] + s['sets']['W'] + s['sets']['D']
    ok = True
    log('CHECK', 'ok  ', f'{len(names)} windows in V-032 windows.json')
    if not MAPS:
        log('CHECK FAIL V033_MAPS is not set'); return False
    miss = [f"{n} o{o:+d}" for n in names for o in OFFSETS
            if not os.path.exists(os.path.join(MAPS, s['source'][n]['segment'], 'win', f'{n}_sweep_ink_o{o:+d}.npy'))]
    miss += [n + ' carried' for n in names
             if not os.path.exists(os.path.join(MAPS, s['source'][n]['segment'], 'win', f'{n}_carried.npy'))]
    log('CHECK', 'ok  ' if not miss else 'FAIL', f'all {len(names) * len(OFFSETS)} ink maps and {len(names)} carried masks present',
        miss[:4])
    ok = not miss
    log('CHECK OK' if ok else 'CHECK FAIL')
    return ok


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd in ('check', '--check'):
        sys.exit(0 if check() else 1)
    if cmd == 'stats':
        stats_stage()
    elif cmd == 'rule':
        rule_stage()
    elif cmd == 'selftest':
        selftest()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
