"""V-049 driver: V-041's 8x recipe (up to 24 patches per training segment, 16000 iterations) on PHerc0139, for a rented pod.

Spec: experiments/v049/README.md (registered before any V-049 plan, render, training or score). `v039_run.py` (unchanged, imported)
implements every stage; this file only changes its numbers, results folder, checkpoint folders, map names and time accounting,
so the recipe, 3D guard, scoring and rule are V-034's by construction.

usage: v049_run.py check | plan | prep [names] | train K | eval K | rule | compare | status | selftest
Env: as V-039 plus V049_RESULTS (default projects/vesuvius/results/v049-p0139-scale2); compare also reads V039_RESULTS.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v039'))
sys.path.insert(0, os.path.join(HERE, '..', 'v036'))
import v039_run as v39  # noqa: E402
import v036_run as v36  # noqa: E402

v34, v31, np = v39.v34, v39.v31, v39.np
PER_SEG, ITERS = 24, 16000
MIN_TRAIN = int(os.environ.get('V049_MIN_TRAIN', '60'))              # 55 only for the registered rerun (README addendum)
PREFIX, TAG = 'v049_', 'sc2p0'
RESULTS = os.environ.get('V049_RESULTS', os.path.join(v39.RES, 'v049-p0139-scale2'))
TEXT = [("fine-tune on PHercParis4, 9 patches per segment and 6000 iterations (V-039)", "fine-tune on PHerc0139, 24 patches per segment and 16000 iterations (V-049)"),
        ("V-039: V-034 with 3x patches and 3x iterations", "V-049: V-034 on PHerc0139 with up to 8x patches and 8x iterations"),
        ("fine-tune on PHercParis4 (V-034)", "fine-tune on PHerc0139 (V-049)"),
        ('experiments/v039/README.md', 'experiments/v049/README.md'), ('v039_run.py', 'v049_run.py')]


def configure():
    v39.PER_SEG, v39.ITERS, v39.MIN_TRAIN, v39.PREFIX, v39.TAG, v39.RESULTS = PER_SEG, ITERS, MIN_TRAIN, PREFIX, TAG, RESULTS
    v39.TEXT = TEXT
    v39.configure()
    v34.SCROLL, v34.FOLDS = v36.SCROLL, v36.FOLDS                                  # PHerc0139's seven scored segments, V-036's folds
    v34.cdir = lambda k: os.path.join(v31.DATA, f'ckpt_p0139_scale2_fold{k}')      # after v39.configure, which sets its own folder
    v34.RESULTS, v34.ROWS = RESULTS, os.path.join(RESULTS, 'eval_rows.json')
    if not getattr(v34.p4_windows, 'v036', False):
        orig_w = v34.p4_windows

        def windows():
            return {s: w for s, w in orig_w().items() if w}                         # drop segments with no windows (0139_w043)
        windows.v036 = True
        v34.p4_windows = windows


def compare():
    """Descriptive only: V-049's mean gain over zero-shot against V-036's (V-034 recipe on the same scroll) on the windows in both."""
    configure()
    j = lambda d: v31.jload(os.path.join(d, 'eval_rows.json'), {})  # noqa: E731
    r36 = os.environ.get('V036_RESULTS', os.path.join(v39.RES, 'v036-p0139-finetune'))
    a, b = j(RESULTS), j(r36)
    both = [n for n in a if n in b and None not in (a[n]['finetuned'], b[n]['finetuned'])]
    g = lambda rows: np.array([rows[n]['finetuned'] - rows[n]['zero_shot'] for n in both])  # noqa: E731
    low = np.array([a[n]['zero_shot'] < v34.AUC_OK for n in both], bool)
    m = lambda x: float(x.mean()) if len(x) else None  # noqa: E731
    c = {'n': len(both), 'mean_gain': {'v049': m(g(a)), 'v036': m(g(b))},
         'stock_below_0.75': {'n': int(low.sum()), 'v049': m(g(a)[low]), 'v036': m(g(b)[low])},
         'stock_at_least_0.75': {'v049': m(g(a)[~low]), 'v036': m(g(b)[~low])}}
    f = v34.fmt
    lines = [f"COMPARE {c['n']} windows: mean gain over zero-shot V-049 {f(c['mean_gain']['v049'])}, V-036 {f(c['mean_gain']['v036'])}",
             f"DESCR stock < 0.75 ({c['stock_below_0.75']['n']} windows): V-049 {f(c['stock_below_0.75']['v049'])}, V-036 {f(c['stock_below_0.75']['v036'])}; "
             f"stock >= 0.75: V-049 {f(c['stock_at_least_0.75']['v049'])}, V-036 {f(c['stock_at_least_0.75']['v036'])}"]
    for ln in lines:
        v31.log(ln)
    v31.jsave({'compare': c, 'lines': lines}, os.path.join(RESULTS, 'compare.json'))
    return c


def selftest():
    configure()
    assert v34.MAX_TRAIN_PER_SEG == 24 and v34.ITERS == 16000 and v34.cdir(1).endswith('ckpt_p0139_scale2_fold1')
    assert v34.SCROLL == 'PHerc0139' and v34.FOLDS is v36.FOLDS
    assert v34.train_cmd('/v/f.py', '/c', '/o', [('/a.npy', '/a.png')], 'cuda')[-2:] == ['--iters', '16000']
    seen = {}
    v39.wrap_select(lambda *a, **k: seen.update(k))('seg', None)
    assert seen == {'max_n': 24}, seen
    assert RESULTS.endswith('v049-p0139-scale2') and v34.RESULTS == RESULTS
    import tempfile
    saved = v31.DATA
    try:
        with tempfile.TemporaryDirectory() as d:
            v31.DATA = d
            for k, s in (('v034_train0', 100.0), ('v034_eval0', 3.0), ('v039_eval0', 8.0)):
                v31.timing(k, s)
            assert v39.timing_total() == {'v049_eval0': 3.0}, v39.timing_total()
    finally:
        v31.DATA = saved
    assert v31.DATA == saved
    calls = []
    fake = type('M', (), {'infer': staticmethod(lambda m, s, o: calls.append(o) or o)})
    v39.wrap_infer(fake)
    fake.infer('ck', None, '/d/w1_ft_ink.npy')
    assert calls == ['/d/w1_sc2p0_ft_ink.npy'], calls
    # the plan is written fresh by `plan` on the pod: a different patch set from V-034's and V-039's, but V-036's folds and windows
    v34.log('SELFTEST ok (V-049: numbers, folders, map names, accounting)')


def check():
    configure()
    selftest()
    ok = v31.check()
    pc = v39.plan_check()
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
        v34.plan_stage(); v39.plan_check()
    elif cmd == 'prep':
        v34.prep(set(args))
    elif cmd == 'train':
        v34.train(int(args[0]))
    elif cmd == 'eval':
        v34.evaluate(int(args[0]))
    elif cmd == 'rule':
        v34.rule_stage(); v39.finish_text()
    elif cmd == 'compare':
        compare()
    elif cmd == 'status':
        v34.status()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
