"""V-047 driver: does V-041's result survive a second training seed?

Spec: experiments/v047/README.md (registered before any V-047 training or score). V-041 again (24 patches per segment, 16000
iterations, V-041's committed plan and rendered patches) with `finetune_on_scan.py --seed 1`. `v041_run.py` and `v037_run.py`
(`compare_rows`) are imported unchanged; this file swaps the seed, results folder, checkpoint folders, map names and accounting.

usage: v047_run.py check | train K | eval K | rule | compare | status | selftest
Env: as V-041 plus V047_RESULTS (default projects/vesuvius/results/v047-p4-scale2-seed1) and V041_RESULTS (read by compare / check).
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('v041', 'v037'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import v041_run as v41  # noqa: E402
import v039_run as v39  # noqa: E402
import v037_run as v37  # noqa: E402

v34, v31, np = v41.v34, v41.v31, v41.np
SEED = 1
PREFIX, TAG = 'v047_', 'sc2s1'
RESULTS = os.environ.get('V047_RESULTS', os.path.join(v39.RES, 'v047-p4-scale2-seed1'))
R41 = os.environ.get('V041_RESULTS', os.path.join(v39.RES, 'v041-p4-scale2'))
HOLD_FRAC = 0.5                                                      # seed 1 keeps at least half of V-041's mean gain
TEXT = [("fine-tune on PHercParis4, 24 patches per segment and 16000 iterations (V-041)", "fine-tune on PHercParis4, 24 patches per segment, 16000 iterations, seed 1 (V-047)"),
        ("V-041: V-034 with up to 8x patches and 8x iterations", "V-047: V-041 again with training seed 1"),
        ('experiments/v041/README.md', 'experiments/v047/README.md'), ('v041_run.py', 'v047_run.py')]


ORIG_CMD = v34.train_cmd                                           # V-034's, before any wrapper: configure() is called more than once


def configure():
    v34.train_cmd = ORIG_CMD
    v41.PREFIX, v41.TAG, v41.RESULTS, v41.TEXT = PREFIX, TAG, RESULTS, TEXT
    v41.configure()
    v34.cdir = lambda k: os.path.join(v31.DATA, f'ckpt_scale2_seed{SEED}_fold{k}')
    with_iters = v34.train_cmd                                      # V-041's wrapper (adds --iters), installed by v41.configure above

    def train_cmd(*a, **k):
        return with_iters(*a, **k) + ['--seed', str(SEED)]
    v34.train_cmd = train_cmd


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def compare():
    configure()
    plan = v34.committed_plan()
    j = lambda d: v31.jload(os.path.join(d, 'eval_rows.json'), {})  # noqa: E731
    sk = lambda d: [v31.jload(os.path.join(d, f'train_log_fold{k}.json'), {}).get('skipped_frac', 0.0) for k in range(3)]  # noqa: E731
    scores = v31.jload(v34.SCORES)
    comp = {n: scores[n].get('canonical') is not None and scores[n].get('mrg20736_1um') is not None for f in plan['folds'].values() for n in f['test']}
    c = v37.compare_rows(j(R41), j(RESULTS), plan, sk(R41), sk(RESULTS), comp)
    f = v34.fmt
    lines = [f"COMPARE V-041 (seed 0) {c['seed0']['verdict']} (mean gain {f(c['seed0']['mean_gain'])}, {c['seed0']['above']} of {c['seed0']['n']} up) vs "
             f"V-047 (seed 1) {c['seed1']['verdict']} (mean gain {f(c['seed1']['mean_gain'])}, {c['seed1']['above']} of {c['seed1']['n']} up) -> {c['outcome']}",
             f"DESCR {c['windows_in_both']} windows in both; per-window gain correlation {f(c['gain_correlation'])}; mean |gain seed 0 - gain seed 1| {f(c['mean_abs_gain_difference'])}"]
    g0, g1 = c['seed0']['mean_gain'], c['seed1']['mean_gain']
    c['mean_gain_holds'] = bool(g0 is not None and g1 is not None and g0 > 0 and g1 >= HOLD_FRAC * g0)
    lines.append(f"MEAN_GAIN_HOLDS {'yes' if c['mean_gain_holds'] else 'no'}: seed-1 mean gain {f(g1)} vs {HOLD_FRAC:.0%} of V-041's {f(g0)} (descriptive companion to the verdict match)")
    for ln in lines:
        v31.log(ln)
    v31.jsave({'compare': c, 'lines': lines}, os.path.join(RESULTS, 'compare.json'))
    return c


def selftest():
    configure()
    assert v34.cdir(2).endswith('ckpt_scale2_seed1_fold2') and v34.ITERS == 16000 and v34.MAX_TRAIN_PER_SEG == 24
    cmd = v34.train_cmd('/v/f.py', '/c', '/o', [('/a.npy', '/a.png')], 'cuda')
    assert cmd[-4:] == ['--iters', '16000', '--seed', '1'] and cmd.count('--seed') == 1, cmd
    calls = []
    fake = type('M', (), {'infer': staticmethod(lambda m, s, o: calls.append(o) or o)})
    v39.wrap_infer(fake)
    fake.infer('ck', None, '/d/w_ft_ink.npy')
    assert calls == ['/d/w_sc2s1_ft_ink.npy'], calls                      # not V-034's _ft_ink.npy, not V-041's _sc2_ft_ink.npy
    import tempfile
    saved = v31.DATA
    try:
        with tempfile.TemporaryDirectory() as d:
            v31.DATA = d
            for k, s in (('v034_eval0', 3.0), ('v041_eval0', 9.0), ('v034_train0', 50.0)):
                v31.timing(k, s)
            assert v39.timing_total() == {'v047_eval0': 3.0}, v39.timing_total()
    finally:
        v31.DATA = saved
    assert v31.DATA == saved
    p, q = os.path.join(RESULTS, 'plan.json'), os.path.join(R41, 'plan.json')
    assert os.path.exists(p), p
    if os.path.exists(q):
        assert sha(p) == sha(q), 'V-047 plan.json differs from V-041 plan.json'
    rows = lambda g: {f'{k}_{i}': {'segment': 's', 'fold': k, 'band': 0, 'zero_shot': 0.8, 'mrg20736_1um': 0.9, 'finetuned': 0.8 + g} for k in range(3) for i in range(40)}  # noqa: E731
    plan = {'folds': {str(k): {'test': [f'{k}_{i}' for i in range(40)]} for k in range(3)}}
    c = v37.compare_rows(rows(0.02), rows(0.012), plan, [0, 0, 0], [0, 0, 0])
    assert c['seed0']['mean_gain'] > 0 and c['seed1']['mean_gain'] >= HOLD_FRAC * c['seed0']['mean_gain']
    v34.log('SELFTEST ok (V-047: seed flag, folders, map names, accounting, plan copy)')


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
    elif cmd == 'train':
        v34.train(int(args[0]))
    elif cmd == 'prep':
        v34.prep(set(args))
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
