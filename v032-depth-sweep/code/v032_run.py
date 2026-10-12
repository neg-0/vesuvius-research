"""V-032 driver: can a depth offset alone explain why the stock canonical 2.4 um model loses to the organisers'
mrg20736-1um map on PHercParis4? A +-32 layer (+-72 um) oracle depth sweep on 12 loss windows against a 12-window
control of winning windows (PHercParis4), plus a descriptive sweep on the 7 PHerc0139 w044/w045 windows.

Spec: experiments/v032/README.md (registered before any V-032 score). This script implements it and does not change it.
It reuses V-031's code by import (experiments/v031/v031_run.py: committed plan, window mesh, labels, render, pixel sets,
mrg crop, layer slicing; and through it V-030's maths and inference). Nothing is trained.

Stages:
  select   CPU. Chooses the windows from the committed V-031 scores.json and writes results/v032-depth-sweep/windows.json
           (committed before any sweep). With the file present it prints SELECTION SAME / DIFFERENT instead.
  run [--limit N]  GPU machine. Per window: render 127 layers along the step-2 mesh, score the canonical model at the nine
           offsets on V-031's ds8 pixels, score mrg20736-1um on the same pixels, save the row (sweep.json), delete the
           stack. Resumable (rows already in sweep.json are skipped). Stops with REPRO FAIL if offset 0 does not
           reproduce V-031's canonical AUC on one of the first 3 windows.
  rule     CPU. RULE / VERDICT lines, summary.json, RESULT.md.
  status   rows done, GPU-stage time, estimate for the rest.
  check    selftest + selection + env (same inputs as V-031's check, no bucket probes beyond V-031's).

usage: v032_run.py check | select | run [--limit N] | rule | status | selftest
Env: as V-031 (V031_DATA = a NEW work folder for V-032, V031_SRC, V031_VILLA, CANON_CKPT, XSCAN_CACHE, PUZZLE_RUN_DIR,
INK_DEVICE) plus V032_RESULTS (default projects/vesuvius/results/v032-depth-sweep). V031_RESULTS keeps its default (the
committed V-031 step0 / windows / scores files are read from there).
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v031'))
import v031_run as v31                                              # noqa: E402  (V-031 code, unchanged)

v30 = v31.v30
log, jload, jsave = v31.log, v31.jload, v31.jsave
RESULTS = os.environ.get('V032_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v032-depth-sweep'))
SCORES = os.path.join(v31.RESULTS, 'scores.json')

OFFSETS = [-32, -24, -16, -8, 0, 8, 16, 24, 32]        # layers of 2.258 um along the render normal (negative = -normal)
MAXOFF = max(OFFSETS)
DEEP_LAYERS = v31.LAYERS + 2 * MAXOFF                  # 127; the centre 63 are exactly V-031's 63-layer render
N_LOSS = N_WIN = 2                                     # per PHercParis4 segment
P4 = 'PHercParis4'
DEPTH_SEGS = ('0139_w044', '0139_w045')                # the V-031b depth-column segments (descriptive set D)
GAP_TOL = 0.03                                         # "gap closed": best-offset AUC >= mrg AUC - 0.03
CLOSE_FRAC = 0.5                                       # ... on >= ceil(0.5 n) loss windows
GAIN_MARGIN = 0.05                                     # median gain (loss set) - median gain (win set) >= 0.05
REPRO_TOL = 0.005
REPRO_N = 3
EST = {'render_min': 2.0, 'offset_min': 0.22, 'carry_min_p4': 5.0, 'carry_min_0139': 2.0}
BUDGET_S = 4 * 3600
OUT = os.path.join(RESULTS, 'sweep.json')


# ------------------------------------------------------------------------------------------------------- selection
def choose(scores):
    """Deterministic from V-031's scores: per PHercParis4 segment the N_LOSS windows with the lowest canonical - mrg
    (set L) and the N_WIN with the highest (set W); set D = every scored window of the two PHerc0139 depth segments."""
    by = {}
    for name, r in scores.items():
        if r['scroll'] == P4 and r.get('canonical') is not None and r.get('mrg20736_1um') is not None:
            by.setdefault(r['segment'], []).append((r['canonical'] - r['mrg20736_1um'], name))
    sets = {'L': [], 'W': [], 'D': []}
    for seg in sorted(by):
        v = sorted(by[seg])
        assert len(v) >= N_LOSS + N_WIN, (seg, len(v))
        sets['L'] += [n for _, n in v[:N_LOSS]]
        sets['W'] += [n for _, n in v[-N_WIN:]]
    sets['D'] = sorted(n for n, r in scores.items() if r['segment'] in DEPTH_SEGS)
    assert not set(sets['L']) & set(sets['W'])
    return sets


def build_selection(scores):
    sets = choose(scores)
    keep = ('segment', 'scroll', 'band', 'pos_ds8', 'neg_ds8', 'canonical', 'mrg20736_1um')
    names = sets['L'] + sets['W'] + sets['D']
    return {'hypothesis': 'V-032', 'offsets_layers': OFFSETS, 'deep_layers': DEEP_LAYERS,
            'rule': {'GAP_TOL': GAP_TOL, 'CLOSE_FRAC': CLOSE_FRAC, 'GAIN_MARGIN': GAIN_MARGIN, 'REPRO_TOL': REPRO_TOL,
                     'N_LOSS': N_LOSS, 'N_WIN': N_WIN},
            'scores_sha256': hashlib.sha256(open(SCORES, 'rb').read()).hexdigest(),
            'sets': sets, 'source': {n: {k: scores[n].get(k) for k in keep} for n in names}}


def select_stage():
    scores = jload(SCORES)
    sel = build_selection(scores)
    p = os.path.join(RESULTS, 'windows.json')
    old = jload(p)
    if old is None:
        jsave(sel, p)
        log('SELECTION written to', p)
    else:
        log('SELECTION', 'SAME' if old == json.loads(json.dumps(sel)) else 'DIFFERENT', 'as', p)
    s = sel['sets']
    log(f"SELECTION L {len(s['L'])} loss windows, W {len(s['W'])} control windows (PHercParis4), D {len(s['D'])} (PHerc0139)")
    estimate(sel)
    return sel


def estimate(sel):
    s = sel['sets']
    per = EST['render_min'] + len(OFFSETS) * EST['offset_min']
    n = len(s['L']) + len(s['W']) + len(s['D'])
    segs = {sel['source'][k]['segment'] for k in sel['source']}
    carry = sum(EST['carry_min_p4'] if x.startswith('P4_') else EST['carry_min_0139'] for x in segs)
    tot = n * per + carry
    log(f'ESTIMATE {n} windows x {per:.1f} min + carry {carry:.0f} min = ~{tot / 60:.1f} h (cap {BUDGET_S // 3600} h)')
    return tot


def committed_selection():
    """windows.json must be committed and unchanged in git before any sweep (V-031's rule for its windows files)."""
    p = os.path.join(RESULTS, 'windows.json')
    sel = jload(p)
    if sel is None:
        raise SystemExit('windows.json is not committed; run select and commit it before run')
    root = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
    rel = os.path.relpath(os.path.abspath(p), root)
    stt = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--', rel], capture_output=True, text=True).stdout
    tracked = subprocess.run(['git', '-C', root, 'ls-files', '--error-unmatch', rel], capture_output=True).returncode == 0
    if not tracked or stt.strip():
        raise SystemExit(f'{rel} must be committed and unchanged before run (git status: {stt.strip() or "untracked"})')
    return sel


def order(sel):
    """Run order: PHercParis4 segment by segment (L then W), then the PHerc0139 set; a budget cut-off leaves whole
    segments done."""
    s, src = sel['sets'], sel['source']
    p4 = sorted(s['L'] + s['W'], key=lambda n: (src[n]['segment'], 0 if n in s['L'] else 1, n))
    return p4 + list(s['D'])


# ------------------------------------------------------------------------------------------------------- run
def best_offset(auc):
    """argmax over the defined offsets; ties go to the smaller |offset|, then the more negative one."""
    d = [(o, a) for o, a in auc.items() if a is not None and a == a]
    if not d:
        return None, None
    o, a = max(d, key=lambda x: (x[1], -abs(x[0]), -x[0]))
    return o, a


def run_stage(limit=None):
    sel = committed_selection()
    pl = v31.committed_plan()
    W = {w['name']: w for w in v31.windows(pl)}
    rows = jload(OUT, {})
    d = v30.infer_mod()
    ck = os.environ['CANON_CKPT']
    lab_cache, done = {}, 0
    for i, name in enumerate(order(sel)):
        if name in rows:
            continue
        if limit is not None and done >= limit:
            break
        if sum(v for k, v in v31.timing_total().items() if k.startswith('score_v032')) > BUDGET_S:
            log('BUDGET reached: remaining windows stay unscored (rule -> UNKNOWN)')
            break
        w, n, src = W[name], W[name]['segment'], sel['source'][name]
        t0 = time.time()
        if not os.path.exists(os.path.join(v31.DATA, n, 'carry', 'roi.json')):
            v31.carry_one(n)
        S = v31.segs_json()
        sc = S['scrolls'][S['segments'][n]['scroll']]
        lab, cm = v31.window_labels(w, lab_cache)
        which = v31.arms_of(n)['canonical']
        md, rr, cc = v31.window_mesh(w, which)
        out = os.path.join(v31.DATA, n, 'win', f'{name}_sweep')
        ct = v31.render(md, sc['vol_B'], out, rr, cc, DEEP_LAYERS, 1)
        shutil.rmtree(md, ignore_errors=True)
        assert list(ct.shape[1:]) == w['shape_px'] and ct.shape[0] == DEEP_LAYERS, (ct.shape, w['shape_px'])
        unc = (ct != 0).any(0) & ~cm
        ct[:, unc] = 0                                          # uncarried CT is not scored (V-030)
        sets = v30.pixel_sets(lab, cm, v31.layers(ct, 0))
        auc = {}
        for o in OFFSETS:
            m = d.infer(ck, v31.layers(ct, o), out + f'_ink_o{o:+d}.npy')
            a = v30.auc_on(m, sets)
            auc[o] = None if a is None or a != a else float(a)
        mrg = v30.auc_on(v31.mrg_crop(w), sets)
        mrg = None if mrg is None or mrg != mrg else float(mrg)
        bo, ba = best_offset(auc)
        r = {'segment': n, 'scroll': S['segments'][n]['scroll'], 'set': next(k for k, v in sel['sets'].items() if name in v),
             'pos_ds8': int(sets[0].sum()), 'neg_ds8': int(sets[1].sum()), 'auc': {str(o): a for o, a in auc.items()},
             'mrg20736_1um': mrg, 'best_offset': bo, 'best_auc': ba,
             'repro': {'canonical0_minus_v031': None if auc[0] is None or src['canonical'] is None else auc[0] - src['canonical'],
                       'mrg_minus_v031': None if mrg is None or src['mrg20736_1um'] is None else mrg - src['mrg20736_1um'],
                       'pos_same': r_same(sets[0].sum(), src['pos_ds8']), 'neg_same': r_same(sets[1].sum(), src['neg_ds8'])},
             'seconds': round(time.time() - t0, 1)}
        rows[name] = r
        jsave(rows, OUT)
        v31.timing('score_v032', time.time() - t0)
        for suffix in ('.npy', '.json'):
            if os.path.exists(out + suffix):
                os.remove(out + suffix)
        done += 1
        log('WINDOW', f'[{i + 1}/{len(order(sel))}]', name, r['set'], 'auc0', fmt(auc[0]), 'mrg', fmt(mrg), 'best', bo, fmt(ba),
            'repro', fmt(r['repro']['canonical0_minus_v031']), r['seconds'], 's')
        if len(rows) <= REPRO_N:
            dv = r['repro']['canonical0_minus_v031']
            if dv is not None and abs(dv) > REPRO_TOL or not (r['repro']['pos_same'] and r['repro']['neg_same']):
                raise SystemExit(f'REPRO FAIL {name}: offset-0 canonical differs from V-031 by {dv}, '
                                 f"pixel sets same: {r['repro']['pos_same']} {r['repro']['neg_same']}. Stop and report.")


def r_same(a, b):
    return b is not None and int(a) == int(b)


def fmt(x):
    return 'n/a' if x is None or x != x else f'{x:.4f}'


# ------------------------------------------------------------------------------------------------------- rule
def median(v):
    return float(np.median(v)) if len(v) else None


def evaluate(rows, sel):
    """PASS (depth explains, oracle upper bound) iff gap closed on >= ceil(0.5 n) loss windows AND
    median gain(L) - median gain(W) >= GAIN_MARGIN; UNKNOWN if any planned L or W window is unscored or has an
    undefined AUC; FAIL otherwise."""
    s = sel['sets']
    out = {}
    for k in ('L', 'W', 'D'):
        rs = [rows[n] for n in s[k] if n in rows]
        out[k] = {'planned': len(s[k]), 'scored': len(rs)}
        good = [r for r in rs if r['best_auc'] is not None and r['auc']['0'] is not None and r['mrg20736_1um'] is not None]
        out[k]['defined'] = len(good)
        out[k]['gain'] = [r['best_auc'] - r['auc']['0'] for r in good]
        out[k]['median_gain'] = median(out[k]['gain'])
        out[k]['best_offsets'] = [r['best_offset'] for r in good]
        out[k]['closed'] = [bool(r['best_auc'] >= r['mrg20736_1um'] - GAP_TOL) for r in good]
        out[k]['mean_auc0'] = float(np.mean([r['auc']['0'] for r in good])) if good else None
        out[k]['mean_best'] = float(np.mean([r['best_auc'] for r in good])) if good else None
        out[k]['mean_mrg'] = float(np.mean([r['mrg20736_1um'] for r in good])) if good else None
    L, Wn = out['L'], out['W']
    if L['defined'] < L['planned'] or Wn['defined'] < Wn['planned'] or L['planned'] == 0:
        vd, why = 'UNKNOWN', (f"L {L['defined']}/{L['planned']} and W {Wn['defined']}/{Wn['planned']} windows scored with "
                              'defined AUCs (all planned windows needed)')
    else:
        need = math.ceil(CLOSE_FRAC * L['planned'])
        nclosed = sum(L['closed'])
        dm = L['median_gain'] - Wn['median_gain']
        r1, r2 = nclosed >= need, dm >= GAIN_MARGIN
        why = (f"gap closed (best-offset AUC >= mrg - {GAP_TOL}) on {nclosed} of {L['planned']} loss windows (need {need}); "
               f"median gain {L['median_gain']:+.4f} (loss) - {Wn['median_gain']:+.4f} (control) = {dm:+.4f} (need >= {GAIN_MARGIN})")
        vd = 'PASS' if r1 and r2 else 'FAIL'
    out['verdict'], out['reason'] = vd, why
    D = out['D']
    D['best_offset_le_-24'] = sum(o is not None and o <= -24 for o in D['best_offsets'])
    return out


def rule_stage():
    sel = jload(os.path.join(RESULTS, 'windows.json'))
    rows = jload(OUT, {})
    ev = evaluate(rows, sel)
    L, Wn, D = ev['L'], ev['W'], ev['D']
    lines = [f"RULE depth explains PHercParis4 losses (V-032): {ev['reason']} -> {ev['verdict']}",
             f"DESCR loss set: mean AUC at 0 {fmt(L['mean_auc0'])}, at best offset {fmt(L['mean_best'])}, mrg {fmt(L['mean_mrg'])}; "
             f"best offsets {L['best_offsets']}",
             f"DESCR control set: mean AUC at 0 {fmt(Wn['mean_auc0'])}, at best offset {fmt(Wn['mean_best'])}, mrg {fmt(Wn['mean_mrg'])}; "
             f"best offsets {Wn['best_offsets']}",
             f"DESCR PHerc0139 w044/w045 ({D['scored']}/{D['planned']} scored): mean AUC at 0 {fmt(D['mean_auc0'])}, at best offset "
             f"{fmt(D['mean_best'])}, mrg {fmt(D['mean_mrg'])}; best offsets {D['best_offsets']}; "
             f"{D.get('best_offset_le_-24')} with best offset <= -24 (#1912 predicts -34)"]
    rp = [r['repro']['canonical0_minus_v031'] for r in rows.values() if r['repro']['canonical0_minus_v031'] is not None]
    lines.append(f"REPRO offset-0 canonical vs V-031 on {len(rp)} windows: max |diff| {max(map(abs, rp)) if rp else float('nan'):.5f}; "
                 f"pixel sets same on {sum(r['repro']['pos_same'] and r['repro']['neg_same'] for r in rows.values())} of {len(rows)}")
    for ln in lines:
        log(ln)
    used = sum(v for k, v in v31.timing_total().items() if k.startswith('score_v032'))
    jsave({'verdict': ev['verdict'], 'evaluation': ev, 'lines': lines, 'gpu_stage_h': round(used / 3600, 2),
           'rows': rows, 'offsets_layers': OFFSETS}, os.path.join(RESULTS, 'summary.json'))
    write_result(os.path.join(RESULTS, 'RESULT.md'), lines, sel, rows, ev)
    return ev


def write_result(path, lines, sel, rows, ev):
    keep = ''
    if os.path.exists(path):
        t = open(path).read()
        keep = t.split('<!-- prose -->', 1)[1] if '<!-- prose -->' in t else ''
    head = ['# V-032: does a depth offset explain the PHercParis4 losses to mrg20736-1um? ' + ev['verdict'], '',
            'Spec: `experiments/v032/README.md`. Driver: `experiments/v032/v032_run.py`. Lines printed by `rule`:', '']
    head += [f'- `{ln}`' for ln in lines] + ['', '## Per window (canonical AUC at each offset, layers of 2.258 um)', '',
                                               '| window | set | mrg | ' + ' | '.join(f'{o:+d}' for o in OFFSETS) +
                                               ' | best offset | best - at 0 | gap closed |',
                                               '| --- | --- | --- | ' + ' | '.join('---' for _ in OFFSETS) + ' | --- | --- | --- |']
    for k in ('L', 'W', 'D'):
        for n in sel['sets'][k]:
            r = rows.get(n)
            if not r:
                head.append(f'| {n} | {k} | not scored |' + ' |' * (len(OFFSETS) + 3))
                continue
            g = '' if r['best_auc'] is None or r['auc']['0'] is None else f"{r['best_auc'] - r['auc']['0']:+.4f}"
            c = '' if r['best_auc'] is None or r['mrg20736_1um'] is None else ('yes' if r['best_auc'] >= r['mrg20736_1um'] - GAP_TOL else 'no')
            head.append(f"| {n} | {k} | {fmt(r['mrg20736_1um'])} | " + ' | '.join(fmt(r['auc'][str(o)]) for o in OFFSETS) +
                        f" | {r['best_offset']} | {g} | {c} |")
    open(path, 'w').write('\n'.join(head) + '\n\n<!-- prose -->' + (keep or '\n'))


def status():
    sel = jload(os.path.join(RESULTS, 'windows.json'))
    if not sel:
        log('select not run'); return
    rows = jload(OUT, {})
    for k in ('L', 'W', 'D'):
        log(f"set {k}: {sum(n in rows for n in sel['sets'][k])} of {len(sel['sets'][k])} scored")
    used = sum(v for kk, v in v31.timing_total().items() if kk.startswith('score_v032'))
    log(f'GPU-stage time {used / 3600:.2f} h')


# ------------------------------------------------------------------------------------------------------- selftest
def selftest():
    mk = lambda seg, sc, c, m: {'segment': seg, 'scroll': sc, 'band': 0, 'pos_ds8': 100, 'neg_ds8': 100,   # noqa: E731
                                'canonical': c, 'mrg20736_1um': m}
    scores = {}
    for si, seg in enumerate(('P4_a', 'P4_b')):
        for j in range(6):
            scores[f'{seg}_w{j}'] = mk(seg, P4, 0.5 + 0.05 * j, 0.7)              # diff -0.2 .. +0.05
        scores[f'{seg}_undef'] = mk(seg, P4, None, 0.7)
    scores['0139_w044_x'] = mk('0139_w044', 'PHerc0139', 0.4, 0.9)
    scores['0139_w041_x'] = mk('0139_w041', 'PHerc0139', 0.4, 0.9)
    s = choose(scores)
    assert s['L'] == ['P4_a_w0', 'P4_a_w1', 'P4_b_w0', 'P4_b_w1'], s['L']
    assert s['W'] == ['P4_a_w4', 'P4_a_w5', 'P4_b_w4', 'P4_b_w5'], s['W']
    assert s['D'] == ['0139_w044_x'], s['D']
    assert best_offset({-8: 0.6, 0: 0.6, 8: 0.6, 16: None}) == (0, 0.6)              # tie -> smaller |offset|
    assert best_offset({-8: 0.7, 8: 0.7}) == (-8, 0.7)                               # then more negative
    assert best_offset({0: None}) == (None, None)
    # stack slicing: the centre 63 layers of the deep render are the 63-layer render, offsets shift by whole layers
    deep = np.arange(DEEP_LAYERS)[:, None, None] * np.ones((1, 2, 2))
    assert v31.layers(deep, 0)[0, 0, 0] == MAXOFF and v31.layers(deep, -MAXOFF)[0, 0, 0] == 0
    assert v31.layers(deep, MAXOFF)[-1, 0, 0] == DEEP_LAYERS - 2 and v31.layers(deep, 8).shape[0] == 62
    # rule
    sel = {'sets': {'L': [f'l{i}' for i in range(4)], 'W': [f'w{i}' for i in range(4)], 'D': []}}

    def row(a0, best, mrg, off=-8):
        au = {str(o): a0 for o in OFFSETS}; au[str(off)] = best
        return {'auc': au, 'best_auc': max(best, a0), 'best_offset': off, 'mrg20736_1um': mrg}
    rows = {f'l{i}': row(0.6, 0.88, 0.9) for i in range(4)} | {f'w{i}': row(0.9, 0.91, 0.85) for i in range(4)}
    ev = evaluate(rows, sel)
    assert ev['verdict'] == 'PASS', ev['reason']                                         # 4/4 closed, 0.25 - 0.01 >= 0.05
    rows = {f'l{i}': row(0.6, 0.88, 0.9) for i in range(4)} | {f"w{i}": row(0.65, 0.9, 0.85) for i in range(4)}
    assert evaluate(rows, sel)['verdict'] == 'FAIL'                                      # control gains as much (noise)
    rows = {f'l{i}': row(0.6, 0.65, 0.9) for i in range(4)} | {f'w{i}': row(0.9, 0.91, 0.85) for i in range(4)}
    assert evaluate(rows, sel)['verdict'] == 'FAIL'                                      # gap not closed
    rows = {f'l{i}': row(0.6, 0.88, 0.9) for i in range(2)} | {f'l{i}': row(0.6, 0.65, 0.9) for i in range(2, 4)} \
        | {f'w{i}': row(0.9, 0.91, 0.85) for i in range(4)}
    assert evaluate(rows, sel)['verdict'] == 'PASS'                                      # 2 of 4 closed = ceil(0.5 * 4)
    del rows['w3']
    assert evaluate(rows, sel)['verdict'] == 'UNKNOWN'                                   # a planned window unscored
    est = estimate({'sets': {'L': ['a'] * 12, 'W': ['b'] * 12, 'D': ['c'] * 7},
                    'source': {'x': {'segment': 'P4_1'}, 'y': {'segment': '0139_w044'}}})
    assert 2.0 < est / 60 < 3.5, est
    log('SELFTEST ok (selection, best offset, layer slicing, rule, estimate)')


def check():
    selftest()
    sel = jload(os.path.join(RESULTS, 'windows.json'))
    ok = sel is not None
    log('CHECK', 'ok  ' if ok else 'FAIL', 'windows.json committed' if ok else 'windows.json missing: run select')
    if ok:
        log('CHECK', 'ok  ' if sel == json.loads(json.dumps(build_selection(jload(SCORES)))) else 'FAIL',
            'selection equals the one rebuilt from the committed V-031 scores')
        ok = sel == json.loads(json.dumps(build_selection(jload(SCORES))))
        pl = v31.all_plans()
        have = {w['name'] for w in v31.windows(pl)}
        miss = [n for k in sel['sets'].values() for n in k if n not in have]
        log('CHECK', 'ok  ' if not miss else 'FAIL', 'every window is in the committed V-031 plans', miss[:3])
        ok = ok and not miss
    log('CHECK delegating to V-031 check (env, villa commit, CKPT, imports, bucket URLs)')
    ok = v31.check() and ok
    log('CHECK OK' if ok else 'CHECK FAIL')
    return ok


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd in ('check', '--check'):
        sys.exit(0 if check() else 1)
    os.makedirs(v31.DATA, exist_ok=True)
    if cmd == 'select':
        select_stage()
    elif cmd == 'run':
        run_stage(int(args[args.index('--limit') + 1]) if '--limit' in args else None)
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
