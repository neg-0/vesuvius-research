"""V-041 driver: V-039's scale test pushed further (up to 24 patches per training segment, 16000 iterations), for a rented pod.

Spec: experiments/v041/README.md (registered before any V-041 plan, render, training or score). `v039_run.py` (unchanged, imported)
implements every stage; this file only changes its numbers, results folder, checkpoint folders, map names and time accounting,
so the recipe, 3D guard, scoring and rule are V-034's by construction.

usage: v041_run.py check | plan | prep [names] | train K | eval K | rule | compare | status | selftest
Env: as V-039 plus V041_RESULTS (default projects/vesuvius/results/v041-p4-scale2); compare also reads V039_RESULTS.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v039'))
import v039_run as v39  # noqa: E402

v34, v31, np = v39.v34, v39.v31, v39.np
PER_SEG, ITERS, MIN_TRAIN = 24, 16000, 60
PREFIX, TAG = 'v041_', 'sc2'
RESULTS = os.environ.get('V041_RESULTS', os.path.join(v39.RES, 'v041-p4-scale2'))
TEXT = [("fine-tune on PHercParis4, 9 patches per segment and 6000 iterations (V-039)", "fine-tune on PHercParis4, 24 patches per segment and 16000 iterations (V-041)"),
        ("V-039: V-034 with 3x patches and 3x iterations", "V-041: V-034 with up to 8x patches and 8x iterations"),
        ('experiments/v039/README.md', 'experiments/v041/README.md'), ('v039_run.py', 'v041_run.py')]


def configure():
    v39.PER_SEG, v39.ITERS, v39.MIN_TRAIN, v39.PREFIX, v39.TAG, v39.RESULTS = PER_SEG, ITERS, MIN_TRAIN, PREFIX, TAG, RESULTS
    v39.TEXT = TEXT
    v39.configure()
    v34.cdir = lambda k: os.path.join(v31.DATA, f'ckpt_scale2_fold{k}')           # after v39.configure, which sets its own folder
    v34.RESULTS, v34.ROWS = RESULTS, os.path.join(RESULTS, 'eval_rows.json')


def compare():
    configure()
    j = lambda d: v31.jload(os.path.join(d, 'eval_rows.json'), {})  # noqa: E731
    r39 = os.environ.get('V039_RESULTS', os.path.join(v39.RES, 'v039-p4-scale'))
    c = v39.compare_rows(j(RESULTS), j(os.environ.get('V034_RESULTS', os.path.join(v39.RES, 'v034-p4-finetune'))), j(r39))
    f = v34.fmt
    lines = [f"COMPARE {c['n']} windows: mean gain over zero-shot V-041 {f(c['mean_gain']['v039'])}, V-034 {f(c['mean_gain']['v034_seed0'])}, V-039 {f(c['mean_gain']['v037_seed1'])}",
             f"DESCR stock < 0.75 ({c['mean_gain_stock_below_0.75']['n']} windows): V-041 {f(c['mean_gain_stock_below_0.75']['v039'])}, V-034 {f(c['mean_gain_stock_below_0.75']['v034_seed0'])}, "
             f"V-039 {f(c['mean_gain_stock_below_0.75']['v037_seed1'])}; stock >= 0.75: V-041 {f(c['mean_gain_stock_at_least_0.75']['v039'])}, "
             f"V-034 {f(c['mean_gain_stock_at_least_0.75']['v034_seed0'])}, V-039 {f(c['mean_gain_stock_at_least_0.75']['v037_seed1'])}"]
    for ln in lines:
        v31.log(ln)
    v31.jsave({'compare': c, 'lines': lines}, os.path.join(RESULTS, 'compare.json'))
    return c


def gate(g41_mean_gain_v039, g41_mean_gain_v034):
    """Run condition, evaluated on V-039's and V-034's committed summaries before any pod is created: V-039 FAILed and its
    mean gain over zero-shot beats V-034's by at least +0.015. (V-039 PASS means scale is solved: V-041 is cancelled and
    a PHerc0139 replication is registered instead.)"""
    return g41_mean_gain_v039 - g41_mean_gain_v034 >= 0.015


def selftest():
    configure()
    assert v34.MAX_TRAIN_PER_SEG == 24 and v34.ITERS == 16000 and v34.cdir(1).endswith('ckpt_scale2_fold1')
    assert v34.train_cmd('/v/f.py', '/c', '/o', [('/a.npy', '/a.png')], 'cuda')[-2:] == ['--iters', '16000']
    seen = {}
    v39.wrap_select(lambda *a, **k: seen.update(k))('seg', None)
    assert seen == {'max_n': 24}, seen
    assert RESULTS.endswith('v041-p4-scale2') and v34.RESULTS == RESULTS
    import tempfile
    saved = v31.DATA
    try:
        with tempfile.TemporaryDirectory() as d:
            v31.DATA = d
            for k, s in (('v034_train0', 100.0), ('v034_eval0', 3.0), ('v039_eval0', 8.0)):
                v31.timing(k, s)
            assert v39.timing_total() == {'v041_eval0': 3.0}, v39.timing_total()
    finally:
        v31.DATA = saved
    assert v31.DATA == saved
    calls = []
    fake = type('M', (), {'infer': staticmethod(lambda m, s, o: calls.append(o) or o)})
    v39.wrap_infer(fake)
    fake.infer('ck', None, '/d/w1_ft_ink.npy')
    assert calls == ['/d/w1_sc2_ft_ink.npy'], calls
    # the plan is written fresh by `plan` on the pod: a different patch set from V-034's and V-039's, but the same folds and windows
    assert gate(-0.0131 + 0.016, -0.0131) and not gate(-0.0131 + 0.014, -0.0131)
    v34.log('SELFTEST ok (V-041: numbers, folders, map names, accounting, run gate)')


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
