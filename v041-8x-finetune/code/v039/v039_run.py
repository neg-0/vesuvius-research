"""V-039 driver: does V-034 fine-tuning improve with 3x the training patches and 3x the iterations?

Spec: experiments/v039/README.md (registered before any V-039 plan, render, training or score). This is V-034 with two
numbers changed: up to 9 densest training patches per training segment (V-034: 3) and 6000 iterations (V-034: 2000). The
folds, held-out windows, 3D guard, recipe, scoring and rule are V-034's: `v034_run.py` (unchanged, imported) implements every
stage; this file swaps the numbers, results folder, checkpoint folders, map names and time accounting.

usage: v039_run.py check | plan | prep [names] | train K | eval K | rule | compare | status | selftest
Env: as V-034 plus V039_RESULTS (default projects/vesuvius/results/v039-p4-scale) and V034_RESULTS / V037_RESULTS (read by
`compare` only).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v034'))
import v034_run as v34  # noqa: E402

v31, np = v34.v31, v34.np
PER_SEG, ITERS = 9, 6000
MIN_TRAIN = 24                                                       # plan stop condition: fewer than 24 training patches in any fold
PREFIX = 'v039_'
TAG = 'sc'                                                           # fine-tuned map names <window>_sc_ft_ink.npy (never V-034's _ft_ink.npy)
RES = os.path.join(HERE, '..', '..', 'results')
RESULTS = os.environ.get('V039_RESULTS', os.path.join(RES, 'v039-p4-scale'))
TEXT = [("fine-tune on PHercParis4 (V-034)", "fine-tune on PHercParis4, 9 patches per segment and 6000 iterations (V-039)"),
        ("V-034: V-016's fine-tune on PHercParis4's other segments", "V-039: V-034 with 3x patches and 3x iterations"),
        ('experiments/v034/README.md', 'experiments/v039/README.md'), ('v034_run.py', 'v039_run.py')]


def retext(s):
    for a, b in TEXT:
        s = s.replace(a, b)
    return s


def timing_total():
    """The budget counter is the held-out scoring time only (the cap in the README is 4 h for scoring); training has its own
    README cap and is resumable."""
    return {k: v for k, v in v31.jload(os.path.join(v31.DATA, 'timing.json'), {}).items() if k.startswith(PREFIX + 'eval')}


def wrap_infer(mod):
    if getattr(mod.infer, 'v039', False):
        return
    orig = mod.infer

    def infer(model_path, stack, out):
        if out.endswith('_ft_ink.npy'):
            out = out[:-len('_ft_ink.npy')] + f'_{TAG}_ft_ink.npy'
        return orig(model_path, stack, out)
    infer.v039 = True
    mod.infer = infer


def wrap_select(orig):
    def select(*a, **k):
        k.setdefault('max_n', PER_SEG)
        return orig(*a, **k)
    select.v039 = True
    return select


def configure():
    v34.RESULTS = RESULTS
    v34.ROWS = os.path.join(RESULTS, 'eval_rows.json')
    v34.MAX_TRAIN_PER_SEG = PER_SEG
    v34.ITERS = ITERS
    v34.cdir = lambda k: os.path.join(v31.DATA, f'ckpt_scale_fold{k}')
    v34.timing_total = timing_total
    if not getattr(v34.select_train_patches, 'v039', False):
        v34.select_train_patches = wrap_select(v34.select_train_patches)
    if not getattr(v31.timing, 'v039', False):
        orig = v31.timing

        def timing(key, s):
            return orig(key.replace('v034_', PREFIX, 1) if key.startswith('v034_') else key, s)
        timing.v039 = True
        v31.timing = timing
    if not getattr(v34.train_cmd, 'v039', False):
        orig_cmd = v34.train_cmd

        def train_cmd(*a, **k):
            return orig_cmd(*a, **k) + ['--iters', str(ITERS)]
        train_cmd.v039 = True
        v34.train_cmd = train_cmd
    if not getattr(v34.v30.infer_mod, 'v039', False):
        orig_im = v34.v30.infer_mod

        def infer_mod():
            m = orig_im()
            wrap_infer(m)
            return m
        infer_mod.v039 = True
        v34.v30.infer_mod = infer_mod
    if not getattr(v34.log, 'v039', False):
        o = v34.log

        def log(*a):
            o(*[retext(x) if isinstance(x, str) else x for x in a])
        log.v039 = True
        v34.log = log
    os.makedirs(RESULTS, exist_ok=True)


def finish_text():
    for name in ('RESULT.md', 'summary.json'):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            s = open(p).read()                                    # read first: open(p, 'w') truncates the file
            open(p, 'w').write(retext(s))


def plan_check():
    """Plan stop condition: report the FOLD lines and stop if any fold has fewer than MIN_TRAIN training patches."""
    pl = v31.jload(os.path.join(RESULTS, 'plan.json'))
    if pl is None:
        return None
    short = {k: len(f['train']) for k, f in pl['folds'].items() if len(f['train']) < MIN_TRAIN}
    v34.log('PLAN_CHECK', 'FAIL' if short else 'OK', f"training patches per fold { {k: len(f['train']) for k, f in pl['folds'].items()} } (need >= {MIN_TRAIN})")
    return not short


def compare_rows(rows39, rows34, rows37):
    """Descriptive only (no threshold): V-039's mean gain over zero-shot against V-034's and V-037's on the windows in all three."""
    both = [n for n in rows39 if n in rows34 and n in rows37 and None not in (rows39[n]['finetuned'], rows34[n]['finetuned'], rows37[n]['finetuned'])]
    g = lambda rows: np.array([rows[n]['finetuned'] - rows[n]['zero_shot'] for n in both])  # noqa: E731
    g39, g34, g37 = g(rows39), g(rows34), g(rows37)
    low = np.array([rows39[n]['zero_shot'] < v34.AUC_OK for n in both], bool)
    m = lambda a: float(a.mean()) if len(a) else None  # noqa: E731
    return {'n': len(both), 'mean_gain': {'v039': m(g39), 'v034_seed0': m(g34), 'v037_seed1': m(g37)},
            'mean_gain_stock_below_0.75': {'v039': m(g39[low]), 'v034_seed0': m(g34[low]), 'v037_seed1': m(g37[low]), 'n': int(low.sum())},
            'mean_gain_stock_at_least_0.75': {'v039': m(g39[~low]), 'v034_seed0': m(g34[~low]), 'v037_seed1': m(g37[~low])}}


def compare():
    configure()
    j = lambda d: v31.jload(os.path.join(d, 'eval_rows.json'), {})  # noqa: E731
    c = compare_rows(j(RESULTS), j(os.environ.get('V034_RESULTS', os.path.join(RES, 'v034-p4-finetune'))),
                     j(os.environ.get('V037_RESULTS', os.path.join(RES, 'v037-p4-seed1'))))
    f = v34.fmt
    lines = [f"COMPARE {c['n']} windows: mean gain over zero-shot V-039 {f(c['mean_gain']['v039'])}, V-034 seed 0 {f(c['mean_gain']['v034_seed0'])}, "
             f"V-037 seed 1 {f(c['mean_gain']['v037_seed1'])}",
             f"DESCR stock < 0.75 ({c['mean_gain_stock_below_0.75']['n']} windows): V-039 {f(c['mean_gain_stock_below_0.75']['v039'])}, "
             f"V-034 {f(c['mean_gain_stock_below_0.75']['v034_seed0'])}, V-037 {f(c['mean_gain_stock_below_0.75']['v037_seed1'])}; "
             f"stock >= 0.75: V-039 {f(c['mean_gain_stock_at_least_0.75']['v039'])}, V-034 {f(c['mean_gain_stock_at_least_0.75']['v034_seed0'])}, "
             f"V-037 {f(c['mean_gain_stock_at_least_0.75']['v037_seed1'])}"]
    for ln in lines:
        v31.log(ln)
    v31.jsave({'compare': c, 'lines': lines}, os.path.join(RESULTS, 'compare.json'))
    return c


def selftest():
    import tempfile
    configure()
    assert v34.MAX_TRAIN_PER_SEG == 9 and v34.ITERS == 6000 and v34.cdir(2).endswith('ckpt_scale_fold2')
    seen = {}
    wrap_select(lambda *a, **k: seen.update(k))('seg', None)
    assert seen == {'max_n': 9}, seen
    seen.clear()
    wrap_select(lambda *a, **k: seen.update(k))('seg', max_n=2)
    assert seen == {'max_n': 2}, seen                                 # an explicit value still wins
    cmd = v34.train_cmd('/v/f.py', '/c.ckpt', '/o', [('/a.npy', '/a.png')], 'cuda')
    assert cmd[-2:] == ['--iters', '6000'] and cmd.count('--iters') == 1, cmd
    calls = []
    fake = type('M', (), {'infer': staticmethod(lambda m, s, o: calls.append(o) or o)})
    wrap_infer(fake)
    wrap_infer(fake)
    fake.infer('ck', None, '/d/w1_ft_ink.npy'); fake.infer('ck', None, '/d/w1_zero.npy')
    assert calls == ['/d/w1_sc_ft_ink.npy', '/d/w1_zero.npy'], calls
    assert retext('RULE fine-tune on PHercParis4 (V-034): x').startswith('RULE fine-tune on PHercParis4, 9 patches')
    saved = v31.DATA
    try:
        with tempfile.TemporaryDirectory() as d:
            v31.DATA = d
            for k, s in (('v034_train0', 100.0), ('v034_eval0', 3.0), ('v034_prep', 7.0), ('v037_eval0', 9.0)):
                v31.timing(k, s)
            assert timing_total() == {'v039_eval0': 3.0}, timing_total()  # only V-039 scoring time counts against the 4 h
    finally:
        v31.DATA = saved
    assert v31.DATA == saved
    rows = lambda gain: {n: {'zero_shot': 0.7 if n == 'a' else 0.9, 'finetuned': (0.7 if n == 'a' else 0.9) + gain} for n in 'abcd'}  # noqa: E731
    c = compare_rows(rows(0.1), rows(0.05), rows(0.0))
    assert c['n'] == 4 and abs(c['mean_gain']['v039'] - 0.1) < 1e-9 and c['mean_gain_stock_below_0.75']['n'] == 1
    global RESULTS
    saved_res = RESULTS
    try:
        with tempfile.TemporaryDirectory() as d:
            RESULTS = d
            open(os.path.join(d, 'RESULT.md'), 'w').write('RULE fine-tune on PHercParis4 (V-034): x')
            open(os.path.join(d, 'summary.json'), 'w').write('{"a": "V-034: V-016\'s fine-tune on PHercParis4\'s other segments"}')
            finish_text()
            assert open(os.path.join(d, 'RESULT.md')).read().startswith('RULE fine-tune on PHercParis4, 9 patches')
            assert 'V-039' in open(os.path.join(d, 'summary.json')).read()                  # finish_text must not empty the files
    finally:
        RESULTS = saved_res
    v34.log('SELFTEST ok (V-039: patch count, iterations, map names, accounting, compare, finish_text)')


def check():
    configure()
    selftest()
    ok = v31.check()
    pc = plan_check()
    v34.log('CHECK OK' if ok and pc is not False else 'CHECK FAIL')
    return ok and pc is not False


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
    elif cmd == 'plan':
        v34.plan_stage()
        plan_check()
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
