"""V-044 driver: does the three-member average (stock + two fine-tuned seeds) beat the stock model on a second scroll, PHerc0139?

Spec: experiments/v044/README.md (registered before any V-044 training or score). Replicates V-040 on PHerc0139: V-036's folds, windows
and committed plan, with two fine-tuning seeds trained here (seed 0 and seed 1, V-036's recipe) and scored in one pass. `v036_run.py`
(PHerc0139 configuration) and `v034_run.py` (plan, carry, rendering, training, rule) are imported unchanged.

usage: v044_run.py check | carry-and-prep (use v036's commands: R31 carry, prep) | train S K | eval K | rule | status | selftest
  train S K   fine-tune seed S (0 or 1) for fold K
Env: as V-036 plus V044_RESULTS (default projects/vesuvius/results/v044-p0139-ensemble) and V036_RESULTS (plan source, read by selftest).
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v036'))
import v036_run as v36  # noqa: E402

v34, v31, np = v36.v34, v36.v31, v36.np
v30 = v34.v30
PREFIX = 'v044_'
BUDGET_S = 4 * 3600                                                  # scoring only; training is resumable and has its own README cap
RES = os.path.join(HERE, '..', '..', 'results')
RESULTS = os.environ.get('V044_RESULTS', os.path.join(RES, 'v044-p0139-ensemble'))
R36 = os.environ.get('V036_RESULTS', os.path.join(RES, 'v036-p0139-finetune'))
ROWS = os.path.join(RESULTS, 'eval_rows.json')
SEEDS = (0, 1)
CDIR0 = None


def ckpt_dir(seed, k):
    return os.path.join(v31.DATA, f'ckpt_p0139_s{seed}_fold{k}')


def timing_total():
    return {k: v for k, v in v31.jload(os.path.join(v31.DATA, 'timing.json'), {}).items() if k.startswith(PREFIX + 'eval')}


def mean_maps(*maps):
    ms = [np.asarray(m, np.float32) for m in maps]
    assert all(m.shape == ms[0].shape and np.isfinite(m).all() for m in ms), [m.shape for m in ms]
    return sum(ms) / len(ms)


def configure():
    v36.PREFIX, v36.RESULTS = PREFIX, RESULTS
    v36.configure()
    v34.timing_total = timing_total
    v34.ROWS = ROWS
    if not getattr(v34.train_cmd, 'v044', False):
        orig = v34.train_cmd
        v34.train_cmd = lambda *a, **k: orig(*a, **k) + ['--seed', str(os.environ['V044_TRAIN_SEED'])]
        v34.train_cmd.v044 = True


def train(seed, k):
    """Seed-specific checkpoint folder and train log; V-034's `train` is otherwise unchanged (resumable)."""
    configure()
    assert seed in SEEDS, seed
    os.environ['V044_TRAIN_SEED'] = str(seed)
    v34.cdir = lambda kk: ckpt_dir(seed, kk)
    v34.train(k)
    src, dst = os.path.join(RESULTS, f'train_log_fold{k}.json'), os.path.join(RESULTS, f'train_log_s{seed}_fold{k}.json')
    os.replace(src, dst)                                              # keep the two seeds' logs apart
    v34.log(f'TRAINED seed {seed} fold {k}: log in {os.path.basename(dst)}')


def evaluate(k):
    configure()
    pl = v34.committed_plan()
    S = v31.segs_json()
    scores = v31.jload(v34.SCORES)
    W = {w['name']: w for ws in v34.p4_windows().values() for w in ws}
    cks = [os.path.join(ckpt_dir(s, k), 'final.pt') for s in SEEDS]
    for c in cks:
        assert os.path.exists(c), f'{c} missing (train seed/fold first)'
    d = v30.infer_mod()
    rows = v31.jload(ROWS, {})
    lab_cache = {}
    for name in pl['folds'][str(k)]['test']:
        src = scores[name]
        if name in rows or src.get('canonical') is None or src.get('mrg20736_1um') is None:
            continue
        t0 = time.time()
        w, seg = W[name], W[name]['segment']
        v34.ensure_carry(seg)
        out = os.path.join(v31.sdir(seg, 'win'), name + '_e44')
        tmp = [out + t + '.npy' for t in ('_zero', '_s0', '_s1')]
        for p in tmp:
            if os.path.exists(p):
                os.remove(p)                                          # infer() returns an existing file without running the model
        ct, lab, cm, _ = v34.render_window(w, S['scrolls'][S['segments'][seg]['scroll']]['vol_B'], lab_cache, out)
        stack = ct[:62]
        sets = v30.pixel_sets(lab, cm, stack)
        mz = d.infer(os.environ['CANON_CKPT'], stack, tmp[0])
        m0, m1 = d.infer(cks[0], stack, tmp[1]), d.infer(cks[1], stack, tmp[2])
        z, f0, f1 = (float(v30.auc_on(m, sets)) for m in (mz, m0, m1))
        if abs(z - src['canonical']) > v34.REPRO_TOL or (int(sets[0].sum()), int(sets[1].sum())) != (src['pos_ds8'], src['neg_ds8']):
            raise SystemExit(f"REPRO FAIL {name}: zero-shot {z:.4f} vs V-031 {src['canonical']:.4f}, pixels. Stop and report.")
        e3 = float(v30.auc_on(mean_maps(mz, m0, m1), sets))
        b0, b1 = float(v30.auc_on(mean_maps(mz, m0), sets)), float(v30.auc_on(mean_maps(mz, m1), sets))
        rows[name] = json.loads(json.dumps({'segment': seg, 'fold': k, 'band': w['band'], 'pos_ds8': src['pos_ds8'], 'neg_ds8': src['neg_ds8'],
                                            'zero_shot': src['canonical'], 'mrg20736_1um': src['mrg20736_1um'], 'ft_seed0': f0, 'ft_seed1': f1,
                                            'blend_seed0': b0, 'blend_seed1': b1, 'ens3': e3, 'repro_zero_diff': z - src['canonical'],
                                            'seconds': round(time.time() - t0, 1)}).replace('NaN', 'null'))
        v31.jsave(rows, ROWS)
        v31.timing(f'{PREFIX}eval{k}', time.time() - t0)
        for p in tmp:
            if os.path.exists(p):
                os.remove(p)
        v34.log('WINDOW', name, f"zero-shot {z:.4f} seed0 {f0:.4f} seed1 {f1:.4f} ensemble {e3:.4f} mrg20736 {src['mrg20736_1um']:.4f}")
        if sum(timing_total().values()) > BUDGET_S:
            v34.log('BUDGET reached; rerun the same line resumes'); break
    v34.log(f"EVAL fold {k}: {sum(1 for n in pl['folds'][str(k)]['test'] if n in rows)} windows scored")


def rule_rows(rows):
    return {n: {**r, 'finetuned': r['ens3']} for n, r in rows.items()}


def rule_stage():
    configure()
    pl = v34.committed_plan()
    rows = v31.jload(ROWS, {})
    skipped = [max(v31.jload(os.path.join(RESULTS, f'train_log_s{s}_fold{k}.json'), {}).get('skipped_frac', 0.0) for s in SEEDS) for k in range(3)]
    scores = v31.jload(v34.SCORES)
    comp = {n: scores[n].get('canonical') is not None and scores[n].get('mrg20736_1um') is not None for f in pl['folds'].values() for n in f['test']}
    ev = v34.evaluate_rule(rule_rows(rows), pl, skipped, comp)
    got = [rows[n] for f in pl['folds'].values() for n in f['test'] if n in rows]
    fmt = v34.fmt
    m = lambda x: float(np.mean(x)) if len(x) else None  # noqa: E731
    gain = lambda key: m([r[key] - r['zero_shot'] for r in got])  # noqa: E731
    low = [r['ens3'] - r['zero_shot'] for r in got if r['zero_shot'] < v34.AUC_OK]
    hi = [r['ens3'] - r['zero_shot'] for r in got if r['zero_shot'] >= v34.AUC_OK]
    lines = [f"RULE PHerc0139 three-member ensemble (stock + seed 0 + seed 1): {ev['reason'].replace('fine-tuned', 'ensemble')} -> {ev['verdict']}",
             f"DESCR mean AUC: zero-shot {fmt(ev['mean_zero_shot'])}, seed 0 alone {fmt(m([r['ft_seed0'] for r in got]))}, seed 1 alone {fmt(m([r['ft_seed1'] for r in got]))}, "
             f"ensemble {fmt(ev['mean_finetuned'])}, mrg20736-1um {fmt(ev['mean_mrg'])}",
             f"DESCR mean gain over zero-shot: seed 0 alone {fmt(gain('ft_seed0'))}, seed 1 alone {fmt(gain('ft_seed1'))}, two-member blends {fmt(gain('blend_seed0'))} / {fmt(gain('blend_seed1'))}, three-member ensemble {fmt(ev['mean_gain'])}",
             f"DESCR ensemble gain: stock < 0.75 on {len(low)} windows {fmt(m(low))}; stock >= 0.75 on {len(hi)} windows {fmt(m(hi))}"]
    for ln in lines:
        v34.log(ln)
    v31.jsave({'verdict': ev['verdict'], 'evaluation': ev, 'lines': lines, 'gpu_stage_h': round(sum(timing_total().values()) / 3600, 2), 'skipped_frac': skipped},
              os.path.join(RESULTS, 'summary.json'))
    body = [f"# V-044: PHerc0139 stock + seed 0 + seed 1 ensemble: {ev['verdict']}", '', 'Spec: `experiments/v044/README.md`. Lines printed by `rule`:', ''] + [f'- `{ln}`' for ln in lines]
    body += ['', '## Per window', '', '| window | fold | zero-shot | seed 0 | seed 1 | ensemble | mrg20736-1um |', '| --- | --- | --- | --- | --- | --- | --- |']
    for f in pl['folds'].values():
        for n in f['test']:
            r = rows.get(n)
            if r:
                body.append(f"| {n} | {r['fold']} | {fmt(r['zero_shot'])} | {fmt(r['ft_seed0'])} | {fmt(r['ft_seed1'])} | {fmt(r['ens3'])} | {fmt(r['mrg20736_1um'])} |")
    p = os.path.join(RESULTS, 'RESULT.md')
    old = open(p).read() if os.path.exists(p) else ''
    open(p, 'w').write('\n'.join(body) + '\n\n<!-- prose -->' + (old.split('<!-- prose -->', 1)[1] if '<!-- prose -->' in old else '\n'))
    return ev


def status():
    configure()
    tr = {s: [k for k in range(3) if os.path.exists(os.path.join(ckpt_dir(s, k), 'final.pt'))] for s in SEEDS}
    rows = v31.jload(ROWS, {})
    v34.log(f"trained folds by seed {tr}, scored {len(rows)} windows; scoring time {sum(timing_total().values()) / 3600:.2f} h of {BUDGET_S // 3600} h")


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def selftest():
    import tempfile
    configure()
    os.environ['V044_TRAIN_SEED'] = '1'
    assert ckpt_dir(1, 2).endswith('ckpt_p0139_s1_fold2') and ckpt_dir(0, 0) != ckpt_dir(1, 0)
    cmd = v34.train_cmd('/v/f.py', '/c', '/o', [('/a.npy', '/a.png')], 'cuda')
    assert cmd[-2:] == ['--seed', '1'] and cmd.count('--seed') == 1, cmd
    configure()
    assert v34.train_cmd('/v/f.py', '/c', '/o', [('/a.npy', '/a.png')], 'cuda').count('--seed') == 1      # configure is idempotent
    a, b, c = np.array([[0.0, 3.0]]), np.array([[3.0, 3.0]]), np.array([[0.0, 0.0]])
    assert np.allclose(mean_maps(a, b, c), [[1.0, 2.0]])
    saved = v31.DATA
    try:
        with tempfile.TemporaryDirectory() as dd:
            v31.DATA = dd
            for k, s in (('v034_eval0', 3.0), ('v034_train0', 50.0), ('v040_eval0', 9.0), ('v036_eval0', 9.0)):
                v31.timing(k, s)
            assert timing_total() == {'v044_eval0': 3.0}, timing_total()
    finally:
        v31.DATA = saved
    assert v31.DATA == saved
    assert v34.FOLDS is v36.FOLDS and v34.SCROLL == 'PHerc0139'
    p, q = os.path.join(RESULTS, 'plan.json'), os.path.join(R36, 'plan.json')
    assert os.path.exists(p), p
    if os.path.exists(q):
        assert sha(p) == sha(q), 'V-044 plan.json differs from V-036 plan.json'
    plan = {'folds': {str(k): {'test': [f'{k}_{i}' for i in range(40)]} for k in range(3)}}
    mk = lambda g: {n: {'segment': 's', 'fold': int(n[0]), 'band': 0, 'zero_shot': 0.8, 'mrg20736_1um': 0.9, 'ens3': 0.8 + g}  # noqa: E731
                    for f in plan['folds'].values() for n in f['test']}
    assert v34.evaluate_rule(rule_rows(mk(0.03)), plan, [0, 0, 0])['verdict'] == 'PASS'
    assert v34.evaluate_rule(rule_rows(mk(0.01)), plan, [0, 0, 0])['verdict'] == 'FAIL'
    v34.log('SELFTEST ok (V-044: seed folders, one --seed flag, accounting, PHerc0139 config, plan copy, rule)')


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
        train(int(args[0]), int(args[1]))
    elif cmd == 'eval':
        evaluate(int(args[0]))
    elif cmd == 'rule':
        rule_stage()
    elif cmd == 'status':
        status()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
