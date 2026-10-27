"""V-036 driver: V-034's cross-segment fine-tune test, run on PHerc0139's seven scored segments instead of PHercParis4's six.

Spec: experiments/v036/README.md (registered before any V-036 plan, render, training or score). This is V-034's driver with
its scroll, folds, results folder, checkpoint folders and time accounting swapped; nothing else is changed: `v034_run.py`
(unchanged, imported) implements every stage, so the recipe, patch rule, guard, scoring and rule are V-034's by
construction. A later refactor can make v034_run take these as arguments; it is not done now because V-034 is running.

usage: v036_run.py check | plan | prep [names] | train K | eval K | rule | status | selftest
Env: as V-034 (V031_DATA, V031_SRC, V031_VILLA, CANON_CKPT, XSCAN_CACHE, PUZZLE_RUN_DIR, INK_DEVICE) plus V036_RESULTS
(default projects/vesuvius/results/v036-p0139-finetune).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'v034'))
import v034_run as v34  # noqa: E402

v31, np = v34.v31, v34.np
SCROLL = 'PHerc0139'
FOLDS = {0: ['0139_w030', '0139_w045'],                                # 18 + 2 windows
         1: ['0139_w035', '0139_w044', '0139_w041'],                   # 18 + 5 + 11
         2: ['0139_w040', '0139_w039']}                                # 18 + 13  (0139_w043 was dropped at V-031's placement gate)
RESULTS = os.environ.get('V036_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v036-p0139-finetune'))
PREFIX = 'v036_'
TEXT = [("fine-tune on PHercParis4 (V-034)", "fine-tune on PHerc0139 (V-036)"),
        ("V-034: V-016's fine-tune on PHercParis4's other segments", "V-036: V-016's fine-tune on PHerc0139's other segments"),
        ('experiments/v034/README.md', 'experiments/v036/README.md'), ('v034_run.py', 'v036_run.py')]
N_EXPECT = 85                                                           # PHerc0139 windows with a defined canonical and mrg AUC in V-031b


def retext(s):
    for a, b in TEXT:
        s = s.replace(a, b)
    return s


def timing_total():
    return {k: v for k, v in v31.jload(os.path.join(v31.DATA, 'timing.json'), {}).items()
            if k.startswith(PREFIX) and k != PREFIX + 'plan'}


def configure():
    """Point v034_run at PHerc0139. Idempotent."""
    v34.SCROLL, v34.FOLDS = SCROLL, FOLDS
    v34.RESULTS = RESULTS
    v34.ROWS = os.path.join(RESULTS, 'eval_rows.json')
    v34.cdir = lambda k: os.path.join(v31.DATA, f'ckpt_p0139_fold{k}')
    v34.timing_total = timing_total
    if not getattr(v31.timing, 'v036', False):                           # V-036's GPU time is booked under v036_*
        orig = v31.timing

        def timing(key, s):
            return orig(key.replace('v034_', PREFIX, 1) if key.startswith('v034_') else key, s)
        timing.v036 = True
        v31.timing = timing
    if not getattr(v34.p4_windows, 'v036', False):
        orig_w = v34.p4_windows

        def windows():
            return {s: w for s, w in orig_w().items() if w}              # drop segments with no windows (0139_w043)
        windows.v036 = True
        v34.p4_windows = windows
    for fn in (v34.log,):
        if not getattr(fn, 'v036', False):
            def log(*a, _o=fn):
                _o(*[retext(x) if isinstance(x, str) else x for x in a])
            log.v036 = True
            v34.log = log
    os.makedirs(RESULTS, exist_ok=True)


def finish_text():
    for name in ('RESULT.md', 'summary.json'):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            s = open(p).read()
            open(p, 'w').write(retext(s))


def selftest():
    import tempfile
    configure()
    segs = sorted(s for f in FOLDS.values() for s in f)
    assert segs == ['0139_w030', '0139_w035', '0139_w039', '0139_w040', '0139_w041', '0139_w044', '0139_w045'], segs
    assert len(set(segs)) == 7 and all(len(f) >= 2 for f in FOLDS.values())
    assert v34.FOLDS is FOLDS and v34.SCROLL == SCROLL and v34.cdir(1).endswith('ckpt_p0139_fold1')
    assert v34.ROWS.endswith(os.path.join('v036-p0139-finetune', 'eval_rows.json')) or 'V036_RESULTS' in os.environ
    assert retext('RULE fine-tune on PHercParis4 (V-034): x').startswith('RULE fine-tune on PHerc0139 (V-036)')
    saved = v31.DATA
    try:
        with tempfile.TemporaryDirectory() as d:
            v31.DATA = d
            v31.timing('v034_train1', 5.0)
            t = timing_total()
            assert t == {'v036_train1': 5.0}, t
    finally:
        v31.DATA = saved                                        # the selftest runs inside check(): never leave DATA in the deleted temp folder
    # V-031's committed windows for the scroll: 7 segments with windows, 85 comparable
    root = os.path.join(HERE, '..', '..', 'results', 'v031-canonical-vs-mrg-1um')
    wj, sc = json.load(open(os.path.join(root, 'windows_v031b.json'))), json.load(open(os.path.join(root, 'scores.json')))
    have = {s: len(v['windows']) for s, v in wj['segments'].items() if v['windows']}
    assert sorted(have) == segs, have
    comp = [n for n, v in sc.items() if n.startswith('0139') and v.get('canonical') is not None and v.get('mrg20736_1um') is not None]
    assert len(comp) == N_EXPECT, len(comp)
    assert sum(have[s] for f in FOLDS.values() for s in f) == 85
    log = v34.log
    log('SELFTEST ok (V-036: folds, redirects, accounting, 85 comparable windows); V-034 logic is v034_run.py')


def check():
    configure()
    selftest()
    pj = v31.jload(os.path.join(RESULTS, 'plan.json'))
    v34.log('CHECK', 'ok  ' if pj else 'info', 'plan.json present' if pj else 'plan.json not written yet (run plan after carry)')
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
    elif cmd == 'plan':
        v34.plan_stage()
    elif cmd == 'prep':
        v34.prep(set(args))
    elif cmd == 'train':
        v34.train(int(args[0]))
    elif cmd == 'eval':
        v34.evaluate(int(args[0]))
    elif cmd == 'rule':
        v34.rule_stage()
        finish_text()
    elif cmd == 'status':
        v34.status()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
