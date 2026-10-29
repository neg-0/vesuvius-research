"""V-037 driver: does V-034's verdict survive a second training seed?

Spec: experiments/v037/README.md (registered before any V-037 training or score). It is V-034 again with
`finetune_on_scan.py --seed 1` (V-034 used the default seed 0) and nothing else changed: `v034_run.py` (unchanged,
imported) implements every stage; this file swaps the results folder, checkpoint folders, time accounting and the seed,
and adds `compare` (seed 0 vs seed 1). The plan is V-034's committed plan.json, copied byte for byte; `check` verifies it.

usage: v037_run.py check | prep [names] | train K | eval K | rule | compare | status | selftest
Env: as V-034 plus V037_RESULTS (default projects/vesuvius/results/v037-p4-seed1) and V034_RESULTS (default
projects/vesuvius/results/v034-p4-finetune, read by `compare` and `check` only).
"""
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v034'))
import v034_run as v34  # noqa: E402

v31, np = v34.v31, v34.np
SEED = 1
PREFIX = 'v037_'
RESULTS = os.environ.get('V037_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v037-p4-seed1'))
V34_RESULTS = os.environ.get('V034_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v034-p4-finetune'))
TEXT = [("fine-tune on PHercParis4 (V-034)", "fine-tune on PHercParis4, seed 1 (V-037)"),
        ("V-034: V-016's fine-tune on PHercParis4's other segments", "V-037: V-034 again with training seed 1"),
        ('experiments/v034/README.md', 'experiments/v037/README.md'), ('v034_run.py', 'v037_run.py')]


def retext(s):
    for a, b in TEXT:
        s = s.replace(a, b)
    return s


def timing_total():
    return {k: v for k, v in v31.jload(os.path.join(v31.DATA, 'timing.json'), {}).items() if k.startswith(PREFIX)}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def wrap_infer(mod):
    """V-034's evaluate caches the fine-tuned map as <window>_ft_ink.npy in the shared work folder and infer() returns an
    existing file without running the model; V-037's maps get their own name so they can never be V-034's."""
    if getattr(mod.infer, 'v037', False):
        return
    orig = mod.infer

    def infer(model_path, stack, out):
        if out.endswith('_ft_ink.npy'):
            out = out[:-len('_ft_ink.npy')] + f'_s{SEED}_ft_ink.npy'
        return orig(model_path, stack, out)
    infer.v037 = True
    mod.infer = infer


def configure():
    v34.RESULTS = RESULTS
    v34.ROWS = os.path.join(RESULTS, 'eval_rows.json')
    v34.cdir = lambda k: os.path.join(v31.DATA, f'ckpt_seed{SEED}_fold{k}')
    v34.timing_total = timing_total
    if not getattr(v31.timing, 'v037', False):
        orig = v31.timing

        def timing(key, s):
            return orig(key.replace('v034_', PREFIX, 1) if key.startswith('v034_') else key, s)
        timing.v037 = True
        v31.timing = timing
    if not getattr(v34.train_cmd, 'v037', False):
        orig_cmd = v34.train_cmd

        def train_cmd(*a, **k):
            return orig_cmd(*a, **k) + ['--seed', str(SEED)]
        train_cmd.v037 = True
        v34.train_cmd = train_cmd
    if not getattr(v34.log, 'v037', False):
        o = v34.log

        def log(*a):
            o(*[retext(x) if isinstance(x, str) else x for x in a])
        log.v037 = True
        v34.log = log
    if not getattr(v34.v30.infer_mod, 'v037', False):
        orig_im = v34.v30.infer_mod

        def infer_mod():
            m = orig_im()
            wrap_infer(m)
            return m
        infer_mod.v037 = True
        v34.v30.infer_mod = infer_mod
    os.makedirs(RESULTS, exist_ok=True)


def finish_text():
    for name in ('RESULT.md', 'summary.json'):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            s = open(p).read()                                    # read first: open(p, 'w') truncates the file
            open(p, 'w').write(retext(s))


def compare_rows(rows0, rows1, plan, skipped0, skipped1, scores_n=None):
    """Seed 0 (V-034) vs seed 1 (V-037) on the windows scored in both. REPLICATED iff both verdicts are PASS or both are
    FAIL; NOT_REPLICATED if one PASSes and the other FAILs; UNKNOWN otherwise (either verdict INVALID / UNKNOWN)."""
    e0, e1 = v34.evaluate_rule(rows0, plan, skipped0, scores_n), v34.evaluate_rule(rows1, plan, skipped1, scores_n)
    both = [n for n in rows0 if n in rows1 and rows0[n].get('finetuned') is not None and rows1[n].get('finetuned') is not None]
    g0 = np.array([rows0[n]['finetuned'] - rows0[n]['zero_shot'] for n in both])
    g1 = np.array([rows1[n]['finetuned'] - rows1[n]['zero_shot'] for n in both])
    corr = float(np.corrcoef(g0, g1)[0, 1]) if len(both) > 2 and g0.std() > 0 and g1.std() > 0 else None
    v0, v1 = e0['verdict'], e1['verdict']
    out = 'REPLICATED' if v0 == v1 and v0 in ('PASS', 'FAIL') else 'NOT_REPLICATED' if {v0, v1} == {'PASS', 'FAIL'} else 'UNKNOWN'
    return {'outcome': out, 'seed0': {'verdict': v0, 'mean_gain': e0['mean_gain'], 'above': e0['above'], 'n': e0['n']},
            'seed1': {'verdict': v1, 'mean_gain': e1['mean_gain'], 'above': e1['above'], 'n': e1['n']},
            'windows_in_both': len(both), 'gain_correlation': corr,
            'mean_abs_gain_difference': float(np.abs(g0 - g1).mean()) if len(both) else None}


def compare():
    configure()
    plan = v34.committed_plan()
    r0, r1 = v31.jload(os.path.join(V34_RESULTS, 'eval_rows.json'), {}), v31.jload(os.path.join(RESULTS, 'eval_rows.json'), {})
    sk = lambda d: [v31.jload(os.path.join(d, f'train_log_fold{k}.json'), {}).get('skipped_frac', 0.0) for k in range(3)]  # noqa: E731
    scores = v31.jload(v34.SCORES)                              # windows V-031 could not score are not comparable (rule_stage does the same)
    comp = {n: scores[n].get('canonical') is not None and scores[n].get('mrg20736_1um') is not None
            for f in plan['folds'].values() for n in f['test']}
    c = compare_rows(r0, r1, plan, sk(V34_RESULTS), sk(RESULTS), comp)
    lines = [f"COMPARE V-034 (seed 0) {c['seed0']['verdict']} (mean gain {v34.fmt(c['seed0']['mean_gain'])}, {c['seed0']['above']} of {c['seed0']['n']} up) vs "
             f"V-037 (seed 1) {c['seed1']['verdict']} (mean gain {v34.fmt(c['seed1']['mean_gain'])}, {c['seed1']['above']} of {c['seed1']['n']} up) -> {c['outcome']}",
             f"DESCR {c['windows_in_both']} windows in both; per-window gain correlation {v34.fmt(c['gain_correlation'])}; mean |gain seed 0 - gain seed 1| {v34.fmt(c['mean_abs_gain_difference'])}"]
    for ln in lines:
        v31.log(ln)
    v31.jsave({'compare': c, 'lines': lines}, os.path.join(RESULTS, 'compare.json'))
    return c


def selftest():
    global RESULTS
    configure()
    assert v34.RESULTS == RESULTS and v34.cdir(2).endswith('ckpt_seed1_fold2')
    cmd = v34.train_cmd('/v/f.py', '/c.ckpt', '/o', [('/a.npy', '/a.png')], 'cuda')
    assert cmd[-2:] == ['--seed', '1'] and cmd.count('--seed') == 1, cmd
    assert retext('RULE fine-tune on PHercParis4 (V-034): x').startswith('RULE fine-tune on PHercParis4, seed 1 (V-037)')
    import tempfile
    saved = v31.DATA
    try:
        with tempfile.TemporaryDirectory() as d:
            v31.DATA = d
            v31.timing('v034_eval0', 3.0)
            assert timing_total() == {'v037_eval0': 3.0}
    finally:
        v31.DATA = saved                                        # check() runs the selftest first and then uses DATA
    assert v31.DATA == saved
    # fine-tuned map cache: V-034 wrote <window>_ft_ink.npy in the shared work folder and infer() returns an existing file
    # without running the model, so V-037 must write its own names (the first V-037 scoring reused V-034's maps)
    calls = []
    fake = type('M', (), {'infer': staticmethod(lambda m, s, o: calls.append(o) or o)})
    wrap_infer(fake)
    wrap_infer(fake)                                            # idempotent
    fake.infer('ck', None, '/d/w1_ft_ink.npy'); fake.infer('ck', None, '/d/w1_zero.npy')
    assert calls == [f'/d/w1_s{SEED}_ft_ink.npy', '/d/w1_zero.npy'], calls
    p37, p34 = os.path.join(RESULTS, 'plan.json'), os.path.join(V34_RESULTS, 'plan.json')
    assert os.path.exists(p37), 'plan.json copy missing'
    if os.path.exists(p34):
        assert sha(p37) == sha(p34), 'V-037 plan.json differs from V-034 plan.json'
    plan = {'folds': {str(k): {'test': [f'{k}_{i}' for i in range(40)]} for k in range(3)}}

    def rows(gain_up, gain_dn, n_up):
        return {n: {'segment': 's', 'fold': int(n[0]), 'band': 0, 'zero_shot': 0.8, 'mrg20736_1um': 0.9,
                    'finetuned': 0.8 + (gain_up if int(n.split('_')[1]) < n_up else -gain_dn)} for f in plan['folds'].values() for n in f['test']}
    a, b = rows(0.05, 0.01, 40), rows(0.04, 0.01, 40)
    c = compare_rows(a, b, plan, [0, 0, 0], [0, 0, 0])
    assert c['outcome'] == 'REPLICATED' and c['seed0']['verdict'] == c['seed1']['verdict'] == 'PASS', c
    c = compare_rows(a, rows(0.01, 0.05, 10), plan, [0, 0, 0], [0, 0, 0])
    assert c['outcome'] == 'NOT_REPLICATED', c
    c = compare_rows(rows(0.01, 0.05, 10), rows(0.0, 0.05, 5), plan, [0, 0, 0], [0, 0, 0])
    assert c['outcome'] == 'REPLICATED' and c['seed0']['verdict'] == 'FAIL', c
    comp = {n: n[-1] != '9' for f in plan['folds'].values() for n in f['test']}               # windows ending in 9 have no V-031 score
    a2 = {n: r for n, r in a.items() if comp[n]}
    assert compare_rows(a2, a2, plan, [0, 0, 0], [0, 0, 0])['outcome'] == 'UNKNOWN'          # without the map the missing windows count
    assert compare_rows(a2, a2, plan, [0, 0, 0], [0, 0, 0], comp)['outcome'] == 'REPLICATED'
    c = compare_rows(a, {}, plan, [0, 0, 0], [])
    assert c['outcome'] == 'UNKNOWN', c
    saved_res = RESULTS
    try:
        with tempfile.TemporaryDirectory() as d:
            RESULTS = d
            open(os.path.join(d, 'RESULT.md'), 'w').write('RULE fine-tune on PHercParis4 (V-034): x')
            open(os.path.join(d, 'summary.json'), 'w').write('{"a": "V-034: V-016\'s fine-tune on PHercParis4\'s other segments"}')
            finish_text()
            assert open(os.path.join(d, 'RESULT.md')).read().startswith('RULE fine-tune on PHercParis4, seed 1 (V-037)')
            assert 'V-037' in open(os.path.join(d, 'summary.json')).read()                  # finish_text must not empty the files
    finally:
        RESULTS = saved_res
    v34.log('SELFTEST ok (V-037: seed flag, redirects, accounting, plan copy, replication outcomes, finish_text)')


def check():
    configure()
    selftest()
    ok = v31.check()
    v34.log('CHECK OK' if ok else 'CHECK FAIL')
    return ok


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    configure()
    if cmd in ('check', '--check'):
        sys.exit(0 if check() else 1)
    os.makedirs(v31.DATA, exist_ok=True)
    if cmd == 'selftest':
        selftest()
    elif cmd == 'prep':
        v34.prep(set(args))
    elif cmd == 'train':
        v34.train(int(args[0]))
    elif cmd == 'eval':
        v34.evaluate(int(args[0]))
    elif cmd == 'rule':
        v34.rule_stage()
        finish_text()
    elif cmd == 'compare':
        compare()
    elif cmd == 'status':
        v34.status()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
