"""V-034 driver: does V-016's cross-scan fine-tune, trained on PHercParis4's other segments, read held-out PHercParis4
segments better than the stock canonical 2.4 um model, zero-shot, on the 1.129 um scan?

Spec: experiments/v034/README.md (registered before any V-034 prep, training or evaluation). This script implements it
and does not change it. It reuses V-031's code by import (committed windows, step-2 mesh, carried labels, render, pixel
sets, AUC, bf16 inference; through it V-030/V-020) and runs villa's finetune_on_scan.py exactly as V-030 did.

Stages (each resumable; a stage whose output exists is skipped):
  plan     CPU (bucket reads). Needs V-031's `carry` for the six PHercParis4 segments. Folds are fixed (README):
           fold k holds out two whole segments; its training patches come from the other four segments, V-020's
           densest-patch rule, at least 1 mm + the renders' reach (1014 L0 voxels, 3D) from every in-volume vertex of
           the fold's held-out windows (the V-031 windows of those segments, unchanged). Writes
           results/v034-p4-finetune/plan.json, which must be committed before prep / train / eval.
  prep     GPU machine (network). Renders every training patch (63 layers, level 1, along mesh B, offset 0).
  train K  GPU. villa finetune_on_scan.py (defaults, --amp bf16, --voxel-um 2.258, 2000 iterations), one --pair per patch.
  eval K   GPU. The fold's held-out windows: renders as V-031 did, scores the fine-tuned model on V-031's ds8 pixels.
           Zero-shot and mrg20736-1um AUCs are taken from V-031's committed scores.json (offset 0 is exactly
           reproducible: V-032 REPRO 0.00000); the first 3 windows of each run are re-scored zero-shot as a check.
  rule     CPU. RULE / DESCR / VERDICT lines, summary.json, RESULT.md.
  status   what is done, GPU-stage hours against the 4 h per item cap.
  check    selftest, plan present, then V-031's check (env, villa commit, ckpt, imports, bucket URLs).

usage: v034_run.py check | plan | prep [names] | train K | eval K | rule | status | selftest
Env: as V-031 (V031_DATA = a NEW work folder, V031_SRC, V031_VILLA = neg-0/villa at 7a5ba5a, CANON_CKPT, XSCAN_CACHE,
PUZZLE_RUN_DIR, INK_DEVICE) plus V034_RESULTS (default projects/vesuvius/results/v034-p4-finetune).
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'v031'))
import v031_run as v31                                              # noqa: E402  (V-031 code, unchanged)

v30 = v31.v30
v20 = v30.v20
log, jload, jsave = v31.log, v31.jload, v31.jsave
RESULTS = os.environ.get('V034_RESULTS', os.path.join(HERE, '..', '..', 'results', 'v034-p4-finetune'))
SCORES = os.path.join(v31.RESULTS, 'scores.json')
ROWS = os.path.join(RESULTS, 'eval_rows.json')

SCROLL = 'PHercParis4'
FOLDS = {0: ['P4_20230702185753', 'P4_20231007101619'],           # held-out segments per fold (README)
         1: ['P4_20231012184424', 'P4_20231031143852'],
         2: ['P4_20231106155351', 'P4_20231210121321']}
MAX_TRAIN_PER_SEG = 3                                              # densest patches per training segment per fold
NEED = v30.GUARD_VOX + 2 * v30.HALF_LAYERS                         # 1014 L0 voxels = 1.14 mm, as V-030 Amendment 2
PATCH = v30.PATCH                                                  # 40 x 80 B cells
LAYERS = v31.LAYERS
UM_PX = v31.UM_PX
ITERS = 2000                                                       # finetune_on_scan.py default (V-030 used it)
WIN_FRAC, MEAN_GAIN_MIN, SKIP_MAX, N_MIN = 0.7, 0.02, 0.01, 30
AUC_OK = 0.75
REPRO_TOL, REPRO_N = 0.005, 3
BUDGET_S = 4 * 3600
EST = {'render_min': 1.5, 'train_min': 25, 'eval_min_per_window': 1.4}


# ------------------------------------------------------------------------------------------------------- patches
def cdir(k):
    return os.path.join(v31.DATA, f'ckpt_fold{k}')


def select_train_patches(seg, ink, usable, inside_c, Bm, roi, tree, need=NEED, max_n=MAX_TRAIN_PER_SEG,
                         px_mm2=(UM_PX * 1e-3) ** 2):
    """V-030's training-patch rule on one segment: the densest non-overlapping PATCH boxes of in-scope labelled cells
    whose every in-volume vertex (the render also reads the far row and column) is >= `need` L0 voxels (3D) from every
    vertex in `tree` (the fold's held-out windows). Arrays are on the segment's ROI grid (vertices = cells + 1)."""
    H, W = ink.shape
    far_v = np.zeros(usable.shape, bool)
    far_v[usable] = tree.query(Bm[usable], workers=-1)[0] >= need
    ok = v30.cell_ok(far_v)[:H, :W] & inside_c
    lab = np.where(ok, 0.0, 1.0)                                    # band 0 = clear of the fold's held-out windows
    keep = (v20.PATCH_H, v20.PATCH_W, v20.BAND_EDGES, v20.MAX_PER, v20.REL, v20.MIN_DENS, v20.MIN_VALID)
    v20.PATCH_H, v20.PATCH_W = PATCH
    v20.BAND_EDGES, v20.MAX_PER, v20.REL, v20.MIN_DENS, v20.MIN_VALID = (0.5,), max_n, 0.2, 0.02, 0.5
    try:
        picks = v20.choose(ink, inside_c, lab, 0)
    finally:
        (v20.PATCH_H, v20.PATCH_W, v20.BAND_EDGES, v20.MAX_PER, v20.REL, v20.MIN_DENS, v20.MIN_VALID) = keep
    out = []
    for rc, dn in picks:
        r0, r1, c0, c1 = rc
        q = Bm[r0:r1 + 1, c0:c1 + 1][usable[r0:r1 + 1, c0:c1 + 1]]
        d = float(tree.query(q, workers=-1)[0].min())
        if d < need:                                                # a vertex of a non-valid cell got too close
            continue
        R0, C0 = r0 + roi[0], c0 + roi[2]
        out.append({'name': f'T_{seg}_r{R0}_c{C0}', 'segment': seg, 'cells': [R0, R0 + PATCH[0], C0, C0 + PATCH[1]],
                    'cells_roi': [r0, r1, c0, c1], 'density': round(dn, 4),
                    'ink_mm2': round(float(ink[r0:r1, c0:c1][inside_c[r0:r1, c0:c1]].sum() * v31.CELL_B ** 2 * px_mm2), 3),
                    'canvas_origin': [R0 * v31.CELL_B, C0 * v31.CELL_B],
                    'shape_px': [PATCH[0] * v31.CELL_B, PATCH[1] * v31.CELL_B], 'min_dist_vox_to_test': round(d, 1)})
    return out


def window_vertices(w, Bm, usable):
    r0, r1, c0, c1 = w['cells_roi']
    return Bm[r0:r1 + 1, c0:c1 + 1][usable[r0:r1 + 1, c0:c1 + 1]]


def build_plan(win_by_seg, load_seg):
    """win_by_seg: {segment: [V-031 window dicts]} for the six segments. load_seg(seg) -> dict(ink, usable, full, roi, Bm)
    on the segment's ROI grid. Returns the plan (JSON types)."""
    from scipy.spatial import cKDTree
    segs = sorted(s for f in FOLDS.values() for s in f)
    assert segs == sorted(win_by_seg), (segs, sorted(win_by_seg))
    cache = {}

    def get(seg):
        if seg not in cache:
            cache.clear()                                           # one segment's arrays at a time
            cache[seg] = load_seg(seg)
        return cache[seg]
    qs = {k: [] for k in FOLDS}                                     # held-out vertices per fold, read segment by segment
    for seg in segs:
        k = next(k for k, f in FOLDS.items() if seg in f)
        d = get(seg)
        qs[k] += [window_vertices(w, d['Bm'], d['usable']) for w in win_by_seg[seg]]
    trees = {k: cKDTree(np.concatenate(qs[k])) for k in FOLDS}
    patches, folds = {}, {k: {'heldout_segments': FOLDS[k], 'test': [], 'train': [], 'min_vox': None} for k in FOLDS}
    for k in FOLDS:
        folds[k]['test'] = [w['name'] for s in FOLDS[k] for w in win_by_seg[s]]
    mins = {k: [] for k in FOLDS}
    for seg in segs:
        d = get(seg)
        H, W = d['ink'].shape
        inside_c = v30.cell_ok(d['usable'])[:H, :W] & d['full']
        for k in FOLDS:
            if seg in FOLDS[k]:
                continue
            for p in select_train_patches(seg, d['ink'], d['usable'], inside_c, d['Bm'], d['roi'], trees[k]):
                patches.setdefault(p['name'], p)
                patches[p['name']]['min_dist_vox_to_test'] = min(patches[p['name']]['min_dist_vox_to_test'],
                                                                 p['min_dist_vox_to_test'])
                folds[k]['train'].append(p['name']); mins[k].append(p['min_dist_vox_to_test'])
    for k in FOLDS:
        folds[k]['min_vox'] = round(min(mins[k]), 1) if mins[k] else None
    return json.loads(json.dumps({
        'rule': dict(PATCH=PATCH, MAX_TRAIN_PER_SEG=MAX_TRAIN_PER_SEG, NEED_VOX=NEED, LAYERS=LAYERS, LEVEL=1, UM_PX=UM_PX,
                     ITERS=ITERS, WIN_FRAC=WIN_FRAC, MEAN_GAIN_MIN=MEAN_GAIN_MIN, SKIP_MAX=SKIP_MAX, N_MIN=N_MIN),
        'folds': {str(k): v for k, v in folds.items()},
        'patches': sorted(patches.values(), key=lambda p: p['name'])}))


def load_segment(seg):
    """The segment's ROI-grid arrays from V-031's carry (this machine's V031_DATA) and its mesh B from the bucket."""
    S = v31.segs_json()
    s = S['segments'][seg]
    s0, stop = v31.gate(seg, 'plan')
    if stop or s0['mesh_for_step2'] != 'B':
        raise SystemExit(f'{seg}: {stop or "step 2 mesh " + str(s0["mesh_for_step2"])}; V-034 needs mesh B on every segment')
    cd = os.path.join(v31.DATA, seg, 'carry')
    if not os.path.exists(os.path.join(cd, 'roi.json')):
        raise SystemExit(f'{seg}: carry not found in {cd}; run `v031_run.py carry PHercParis4` first')
    roi = json.load(open(os.path.join(cd, 'roi.json')))
    Bm = v31.mesh(s['mesh_B'], np.float32)[0][roi[0]:roi[1], roi[2]:roi[3]]
    return {'ink': np.load(os.path.join(cd, 'ink.npy')), 'usable': v30.load_bits(os.path.join(cd, 'usable.npy')),
            'full': v30.load_bits(os.path.join(cd, 'full.npy')), 'roi': roi, 'Bm': Bm}


def p4_windows():
    pl = v31.committed_plan()
    S = v31.segs_json()['segments']
    out = {}
    for seg, p in pl['segments'].items():
        if S[seg]['scroll'] == SCROLL:
            out[seg] = p['windows']
    return out


def estimate(plan):
    n = len(plan['patches'])
    nt = sum(len(f['test']) for f in plan['folds'].values())
    r, t, e = n * EST['render_min'], 3 * EST['train_min'], nt * EST['eval_min_per_window']
    log(f'ESTIMATE {n} patches to render ~{r:.0f} min, 3 trainings ~{t:.0f} min (item A: ~{(r + t) / 60 + 0.5:.1f} h with the '
        f'carry), {nt} held-out windows ~{e:.0f} min (item B: ~{e / 60:.1f} h); cap {BUDGET_S // 3600} h per item')


def plan_stage():
    pj = os.path.join(RESULTS, 'plan.json')
    old = jload(pj)
    t0 = time.time()
    pl = build_plan(p4_windows(), load_segment)
    if old is None:
        jsave(pl, pj)
        log('PLAN written to', pj, '(commit it before prep)')
    else:
        log('PLAN', 'SAME' if old == pl else 'DIFFERENT', 'as', pj)
    for k, f in pl['folds'].items():
        log(f"FOLD {k} held-out {f['heldout_segments']}: {len(f['test'])} windows; train {len(f['train'])} patches, "
            f"min 3D distance to the held-out windows {f['min_vox']} voxels (need >= {NEED})")
    estimate(pl)
    v31.timing('v034_plan', time.time() - t0)
    return pl


def committed_plan():
    p = os.path.join(RESULTS, 'plan.json')
    pl = jload(p)
    if pl is None:
        raise SystemExit('plan.json is not committed; run plan and commit it before prep / train / eval')
    rel = os.path.relpath(os.path.abspath(p), ROOT)
    stt = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--', rel], capture_output=True, text=True).stdout
    tracked = subprocess.run(['git', '-C', ROOT, 'ls-files', '--error-unmatch', rel], capture_output=True).returncode == 0
    if not tracked or stt.strip():
        raise SystemExit(f'{rel} must be committed and unchanged (git status: {stt.strip() or "untracked"})')
    return pl


# ------------------------------------------------------------------------------------------------------- prep
def timing_total():
    return {k: v for k, v in jload(os.path.join(v31.DATA, 'timing.json'), {}).items() if k.startswith('v034_') and k != 'v034_plan'}


def render_window(w, vol, lab_cache, out):
    """The window / patch on the 1.129 um volume along mesh B (V-031's path): 63 layers, uncarried CT zeroed.
    Returns (stack, labels, carried)."""
    lab, cm = v31.window_labels(w, lab_cache)
    md, rr, cc = v31.window_mesh(w, 'B')
    ct = v31.render(md, vol, out, rr, cc, LAYERS, 1)
    shutil.rmtree(md, ignore_errors=True)
    assert list(ct.shape[1:]) == w['shape_px'] and ct.shape[0] == LAYERS, (ct.shape, w['shape_px'])
    unc = (ct != 0).any(0) & ~cm
    ct[:, unc] = 0
    return ct, lab, cm, int(unc.sum())


def ensure_carry(seg):
    if not os.path.exists(os.path.join(v31.DATA, seg, 'carry', 'roi.json')):
        v31.carry_one(seg)


def prep(names):
    pl = committed_plan()
    S = v31.segs_json()
    lab_cache = {}
    todo = [p for p in pl['patches'] if not names or p['name'] in names]
    for i, p in enumerate(todo):
        t0 = time.time()
        seg = p['segment']
        out = os.path.join(v31.sdir(seg, 'win'), p['name'] + '_train')
        info = jload(out + '.v034.json')
        if info is None:
            ensure_carry(seg)
            vol = S['scrolls'][S['segments'][seg]['scroll']]['vol_B']
            ct, lab, cm, unc = render_window(p, vol, lab_cache, out)
            np.save(out + '.npy', ct)
            info = {'uncarried_ct_px': unc, 'ct_valid_px': int((ct != 0).any(0).sum()), 'seconds': round(time.time() - t0, 1)}
            jsave(info, out + '.v034.json')
            v31.timing('v034_prep', time.time() - t0)
        log(f'[{i + 1}/{len(todo)}]', p['name'], 'ink', p['ink_mm2'], 'mm2', json.dumps(info))


# ------------------------------------------------------------------------------------------------------- train
def train_cmd(port, ckpt, out, pairs, device):
    cmd = [sys.executable, port, '--checkpoint', ckpt, '--out-dir', out, '--amp', 'bf16', '--voxel-um', str(UM_PX),
           '--device', device]
    for stack, labels in pairs:
        cmd += ['--pair', stack, labels]
    return cmd


def train(k):
    pl = committed_plan()
    P = {p['name']: p for p in pl['patches']}
    names = pl['folds'][str(k)]['train']
    assert names, f'fold {k}: no training patches'
    out = cdir(k)
    if not os.path.exists(os.path.join(out, 'final.pt')):
        t0 = time.time()
        pairs = []
        for n in names:
            d = v31.sdir(P[n]['segment'], 'win')
            stack = os.path.join(d, n + '_train.npy')
            assert os.path.exists(stack), f'{n} is not rendered (run prep)'
            pairs.append((stack, os.path.join(d, n + '_lab.png')))
        v = os.environ.get('V030_VILLA')
        if not v:
            raise SystemExit('set V031_VILLA to the neg-0/villa checkout at 7a5ba5a')
        cmd = train_cmd(os.path.join(v, 'ink-detection', 'scan_adapt', 'finetune_on_scan.py'), os.environ['CANON_CKPT'], out,
                        pairs, os.environ.get('INK_DEVICE', 'cuda'))
        log('train fold', k, 'patches', len(names), ' '.join(cmd[:10]), '...')
        subprocess.check_call(cmd)
        v31.timing(f'v034_train{k}', time.time() - t0)
    tl = json.load(open(os.path.join(out, 'train_log.json')))
    jsave({**tl, 'fold': k, 'patches': names}, os.path.join(RESULTS, f'train_log_fold{k}.json'))
    log('TRAINED fold', k, 'iters', tl.get('iters'), 'skipped_frac', tl.get('skipped_frac'))


# ------------------------------------------------------------------------------------------------------- eval
def evaluate(k):
    pl = committed_plan()
    S = v31.segs_json()
    scores = jload(SCORES)
    W = {w['name']: w for ws in p4_windows().values() for w in ws}
    ckpt = os.path.join(cdir(k), 'final.pt')
    assert os.path.exists(ckpt), f'fold {k} is not trained'
    d = v30.infer_mod()
    rows = jload(ROWS, {})
    lab_cache, done = {}, 0
    for name in pl['folds'][str(k)]['test']:
        src = scores[name]
        if name in rows or src.get('canonical') is None or src.get('mrg20736_1um') is None:
            continue                                                # V-031 had no defined zero-shot AUC: nothing to compare
        t0 = time.time()
        w, seg = W[name], W[name]['segment']
        ensure_carry(seg)
        out = os.path.join(v31.sdir(seg, 'win'), name + '_ft')
        ct, lab, cm, _ = render_window(w, S['scrolls'][S['segments'][seg]['scroll']]['vol_B'], lab_cache, out)
        stack = ct[:62]
        sets = v30.pixel_sets(lab, cm, stack)
        m = d.infer(ckpt, stack, out + '_ink.npy')
        r = {'segment': seg, 'fold': k, 'band': w['band'], 'pos_ds8': int(sets[0].sum()), 'neg_ds8': int(sets[1].sum()),
             'zero_shot': src['canonical'], 'mrg20736_1um': src['mrg20736_1um'], 'finetuned': float(v30.auc_on(m, sets))}
        if done < REPRO_N:                                          # offset-0 reproduction of V-031's stock score
            z = float(v30.auc_on(d.infer(os.environ['CANON_CKPT'], stack, out + '_zero.npy'), sets))
            r['repro_zero_shot_diff'] = z - src['canonical']
            if abs(z - src['canonical']) > REPRO_TOL or (r['pos_ds8'], r['neg_ds8']) != (src['pos_ds8'], src['neg_ds8']):
                raise SystemExit(f"REPRO FAIL {name}: zero-shot {z:.4f} vs V-031 {src['canonical']:.4f}, pixels "
                                 f"{r['pos_ds8']}/{r['neg_ds8']} vs {src['pos_ds8']}/{src['neg_ds8']}. Stop and report.")
        r['seconds'] = round(time.time() - t0, 1)
        rows[name] = json.loads(json.dumps(r).replace('NaN', 'null'))
        jsave(rows, ROWS)
        v31.timing(f'v034_eval{k}', time.time() - t0)
        for suffix in ('.npy', '.json'):
            if os.path.exists(out + suffix):
                os.remove(out + suffix)
        done += 1
        log('WINDOW', name, f"zero-shot {r['zero_shot']:.4f} finetuned {r['finetuned']:.4f} mrg20736 {r['mrg20736_1um']:.4f}")
        if sum(timing_total().values()) > BUDGET_S:
            log('BUDGET reached; rerun the same line resumes'); break
    n_done = sum(1 for n in pl['folds'][str(k)]['test'] if n in rows)
    log(f'EVAL fold {k}: {n_done} windows scored')


# ------------------------------------------------------------------------------------------------------- rule
def fmt(x):
    return 'n/a' if x is None or x != x else f'{x:.4f}'


def evaluate_rule(rows, plan, skipped, scores_n=None):
    """PASS: fine-tuned > zero-shot on >= ceil(0.7 n) of the n scored windows AND mean gain >= 0.02; FAIL otherwise;
    INVALID if any fold skipped > 1% of micro-batches or any fine-tuned AUC is undefined or 0; UNKNOWN if fewer than 3 folds
    are evaluated, a planned comparable window is missing, or n < 30."""
    planned = [n for f in plan['folds'].values() for n in f['test']]
    comparable = [n for n in planned if n in rows or (scores_n or {}).get(n, True)]
    got = [rows[n] for n in planned if n in rows]
    ev = {'n': len(got), 'n_planned_comparable': len(comparable), 'folds_done': sorted({r['fold'] for r in got})}
    bad = [n for n in planned if n in rows and (rows[n]['finetuned'] is None or rows[n]['finetuned'] != rows[n]['finetuned']
                                                or rows[n]['finetuned'] == 0.0)]
    gains = [r['finetuned'] - r['zero_shot'] for r in got if r['finetuned'] is not None]
    ev.update(mean_zero_shot=float(np.mean([r['zero_shot'] for r in got])) if got else None,
              mean_finetuned=float(np.mean([r['finetuned'] for r in got if r['finetuned'] is not None])) if got else None,
              mean_mrg=float(np.mean([r['mrg20736_1um'] for r in got])) if got else None,
              mean_gain=float(np.mean(gains)) if gains else None, above=int(sum(g > 0 for g in gains)),
              need=int(math.ceil(WIN_FRAC * len(got))) if got else 0, skipped=skipped)
    if len(ev['folds_done']) < 3 or len(got) < len(comparable):
        ev['verdict'], ev['reason'] = 'UNKNOWN', f"{len(got)} of {len(comparable)} windows scored; folds {ev['folds_done']}"
    elif len(got) < N_MIN:
        ev['verdict'], ev['reason'] = 'UNKNOWN', f'{len(got)} scored windows (need >= {N_MIN})'
    elif bad or (skipped and max(skipped) > SKIP_MAX):
        ev['verdict'], ev['reason'] = 'INVALID', f'skipped fractions {skipped}; undefined or zero fine-tuned AUC on {bad}'
    else:
        ok = ev['above'] >= ev['need'] and ev['mean_gain'] >= MEAN_GAIN_MIN
        ev['verdict'] = 'PASS' if ok else 'FAIL'
        ev['reason'] = (f"fine-tuned > zero-shot on {ev['above']} of {len(got)} windows (need {ev['need']}), mean gain "
                        f"{ev['mean_gain']:+.4f} (need >= {MEAN_GAIN_MIN:+.2f})")
    # descriptive
    low = [r for r in got if r['zero_shot'] < AUC_OK]
    high = [r for r in got if r['zero_shot'] >= AUC_OK]
    g = lambda rs: float(np.mean([r['finetuned'] - r['zero_shot'] for r in rs if r['finetuned'] is not None])) if rs else None  # noqa: E731
    ev['strata'] = {'stock_below_0.75': {'n': len(low), 'mean_gain': g(low)}, 'stock_at_least_0.75': {'n': len(high), 'mean_gain': g(high)}}
    by = {}
    for r in got:
        by.setdefault(r['segment'], []).append(r)
    ev['per_segment'] = {s: {'n': len(v), 'zero_shot': float(np.mean([x['zero_shot'] for x in v])),
                             'finetuned': float(np.mean([x['finetuned'] for x in v])), 'mrg': float(np.mean([x['mrg20736_1um'] for x in v])),
                             'gate_adapts': bool(np.mean([x['zero_shot'] for x in v]) < AUC_OK)} for s, v in sorted(by.items())}
    ev['wins_vs_mrg'] = {'zero_shot': int(sum(r['zero_shot'] >= r['mrg20736_1um'] for r in got)),
                         'finetuned': int(sum(r['finetuned'] is not None and r['finetuned'] >= r['mrg20736_1um'] for r in got))}
    return ev


def rule_stage():
    pl = committed_plan()
    rows = jload(ROWS, {})
    scores = jload(SCORES)
    skipped = []
    for k in range(3):
        tl = jload(os.path.join(RESULTS, f'train_log_fold{k}.json'))
        if tl:
            skipped.append(tl.get('skipped_frac', 0.0))
    comp = {n: scores[n].get('canonical') is not None and scores[n].get('mrg20736_1um') is not None
            for f in pl['folds'].values() for n in f['test']}
    ev = evaluate_rule(rows, pl, skipped, comp)
    s = ev['strata']
    lines = [f"RULE fine-tune on PHercParis4 (V-034): {ev['reason']} -> {ev['verdict']}",
             f"DESCR mean AUC: zero-shot {fmt(ev['mean_zero_shot'])}, fine-tuned {fmt(ev['mean_finetuned'])}, mrg20736-1um {fmt(ev['mean_mrg'])}; "
             f"windows with AUC >= mrg: zero-shot {ev['wins_vs_mrg']['zero_shot']}, fine-tuned {ev['wins_vs_mrg']['finetuned']} of {ev['n']}",
             f"DESCR mean gain: stock < 0.75 on {s['stock_below_0.75']['n']} windows {fmt(s['stock_below_0.75']['mean_gain'])}; "
             f"stock >= 0.75 on {s['stock_at_least_0.75']['n']} windows {fmt(s['stock_at_least_0.75']['mean_gain'])}"]
    for sg, v in ev['per_segment'].items():
        lines.append(f"DESCR {sg}: zero-shot {fmt(v['zero_shot'])} fine-tuned {fmt(v['finetuned'])} mrg {fmt(v['mrg'])} "
                     f"(toolkit gate, adapt only below {AUC_OK}: {'adapts' if v['gate_adapts'] else 'does not adapt'})")
    for ln in lines:
        log(ln)
    used = sum(timing_total().values())
    jsave({'verdict': ev['verdict'], 'evaluation': ev, 'lines': lines, 'gpu_stage_h': round(used / 3600, 2),
           'skipped_frac': skipped}, os.path.join(RESULTS, 'summary.json'))
    write_result(os.path.join(RESULTS, 'RESULT.md'), lines, rows, pl, ev)
    return ev


def write_result(path, lines, rows, pl, ev):
    keep = ''
    if os.path.exists(path):
        t = open(path).read()
        keep = t.split('<!-- prose -->', 1)[1] if '<!-- prose -->' in t else ''
    head = [f"# V-034: V-016's fine-tune on PHercParis4's other segments, read on held-out segments: {ev['verdict']}", '',
            'Spec: `experiments/v034/README.md`. Driver: `experiments/v034/v034_run.py`. Lines printed by `rule`:', '']
    head += [f'- `{ln}`' for ln in lines] + ['', '## Per window', '',
                                               '| window | fold | band | zero-shot | fine-tuned | gain | mrg20736-1um | pos / neg ds8 px |',
                                               '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for f in pl['folds'].values():
        for n in f['test']:
            r = rows.get(n)
            if r:
                gn = None if r['finetuned'] is None else r['finetuned'] - r['zero_shot']
                head.append(f"| {n} | {r['fold']} | {r['band']} | {fmt(r['zero_shot'])} | {fmt(r['finetuned'])} | "
                            f"{'' if gn is None else f'{gn:+.4f}'} | {fmt(r['mrg20736_1um'])} | {r['pos_ds8']} / {r['neg_ds8']} |")
    open(path, 'w').write('\n'.join(head) + '\n\n<!-- prose -->' + (keep or '\n'))


def status():
    pl = jload(os.path.join(RESULTS, 'plan.json'))
    if not pl:
        log('plan not run'); return
    rend = [p['name'] for p in pl['patches'] if os.path.exists(os.path.join(v31.DATA, p['segment'], 'win', p['name'] + '_train.v034.json'))]
    tr = [k for k in range(3) if os.path.exists(os.path.join(cdir(k), 'final.pt'))]
    rows = jload(ROWS, {})
    log(f"rendered {len(rend)}/{len(pl['patches'])} patches, trained folds {tr}, evaluated {len(rows)} windows; "
        f"GPU-stage time {sum(timing_total().values()) / 3600:.2f} h of {BUDGET_S // 3600} h per item; " + json.dumps(timing_total()))


# ------------------------------------------------------------------------------------------------------- selftest
def selftest():
    import tempfile
    from scipy.spatial import cKDTree
    assert sorted(s for f in FOLDS.values() for s in f) == [f'P4_{i}' for i in (
        '20230702185753', '20231007101619', '20231012184424', '20231031143852', '20231106155351', '20231210121321')]
    assert NEED == 886 + 128, NEED
    # synthetic segment: a flat B mesh (20 L0 vox per cell) with ink blobs; a held-out window far away on another segment
    rng = np.random.default_rng(5)
    H, W = 140, 340
    ii, jj = np.meshgrid(np.arange(H + 1, dtype=np.float32), np.arange(W + 1, dtype=np.float32), indexing='ij')
    Bm = np.stack([1000 + 20 * jj, 2000 + 20 * ii, np.full(ii.shape, 5000, np.float32)], -1)
    usable = np.ones((H + 1, W + 1), bool)
    ink = np.zeros((H, W), np.float32)
    for (r, c) in ((20, 30), (70, 60), (10, 220), (85, 255), (110, 20)):
        ink[r:r + 40, c:c + 80] = 0.5
    inside_c = v30.cell_ok(usable)[:H, :W] & np.ones((H, W), bool)
    # held-out vertices: a patch-sized block at cells (60..100, 55..135) of ANOTHER segment sharing this frame
    tvert = Bm[60:101, 55:136].reshape(-1, 3)
    tree = cKDTree(tvert)
    ps = select_train_patches('P4_x', ink, usable, inside_c, Bm, [0, H + 1, 0, W + 1], tree, need=NEED, max_n=3)
    assert ps and all(p['min_dist_vox_to_test'] >= NEED for p in ps), ps
    boxes = [tuple(p['cells_roi']) for p in ps]
    assert all(b[1] - b[0] == PATCH[0] and b[3] - b[2] == PATCH[1] for b in boxes), boxes
    for i, a in enumerate(boxes):                                   # non-overlapping
        for b in boxes[i + 1:]:
            assert a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2], (a, b)
    near = [b for b in boxes if b[0] < 100 + 51 and b[1] > 60 - 51 and b[2] < 135 + 51 and b[3] > 55 - 51]  # within ~51 cells = 1020 vox
    assert not near, near                                           # the blob at (70, 60) overlaps the held-out block
    assert ps[0]['name'].startswith('T_P4_x_r') and ps[0]['cells'][1] - ps[0]['cells'][0] == PATCH[0]
    # the six-segment plan on synthetic data: folds disjoint, held-out segments never train
    segs = sorted(s for f in FOLDS.values() for s in f)
    wins = {s: [{'name': f'{s}_r{10 * i}_c0', 'segment': s, 'band': i % 3, 'cells_roi': [10 * i, 10 * i + 40, 0, 80],
                 'cells': [10 * i, 10 * i + 40, 0, 80], 'shape_px': [400, 800]} for i in range(2)] for s in segs}

    def load(seg):
        sh = (segs.index(seg) * 4000, 0, 0)                         # distinct 3D locations per segment: far apart
        B2 = Bm + np.array([sh[0], 0, 0], np.float32)
        return {'ink': ink.copy(), 'usable': usable.copy(), 'full': np.ones((H, W), bool), 'roi': [0, H + 1, 0, W + 1], 'Bm': B2}
    pl = build_plan(wins, load)
    for k, f in pl['folds'].items():
        held = set(f['heldout_segments'])
        P = {p['name']: p for p in pl['patches']}
        assert f['train'] and all(P[n]['segment'] not in held for n in f['train']), (k, f['train'])
        assert all(P[n]['segment'] in segs for n in f['train'])
        assert len(f['test']) == 4 and f['min_vox'] >= NEED, f
        assert len(f['train']) <= MAX_TRAIN_PER_SEG * 4
    assert sorted(n for f in pl['folds'].values() for n in f['test']) == sorted(w['name'] for ws in wins.values() for w in ws)
    # train command
    cmd = train_cmd('/v/finetune_on_scan.py', '/c.ckpt', '/o', [('/a.npy', '/a.png'), ('/b.npy', '/b.png')], 'cuda')
    assert cmd.count('--pair') == 2 and '--voxel-um' in cmd and cmd[cmd.index('--voxel-um') + 1] == str(UM_PX) and '--amp' in cmd
    # rule
    plan = {'folds': {'0': {'test': [f'a{i}' for i in range(40)]}, '1': {'test': [f'b{i}' for i in range(40)]},
                      '2': {'test': [f'c{i}' for i in range(40)]}}}

    def mk(gain, n_up, zs=0.85):
        rows = {}
        for k, pre in enumerate('abc'):
            for i in range(40):
                up = i < n_up
                rows[f'{pre}{i}'] = {'segment': f'P4_{pre}', 'fold': k, 'band': 0, 'zero_shot': zs, 'mrg20736_1um': 0.92,
                                     'finetuned': zs + (gain if up else -gain)}
        return rows
    assert evaluate_rule(mk(0.04, 40), plan, [0, 0, 0])['verdict'] == 'PASS'                 # 120/120, +0.04
    assert evaluate_rule(mk(0.04, 30), plan, [0, 0, 0])['verdict'] == 'PASS'                 # 90/120 up (need 84), mean +0.02
    assert evaluate_rule(mk(0.04, 20), plan, [0, 0, 0])['verdict'] == 'FAIL'                 # 60/120 up (< 84)
    return finish_selftest(mk, plan)


def finish_selftest(mk, plan):
    e = evaluate_rule(mk(0.04, 30), plan, [0, 0, 0])
    assert abs(e['mean_gain'] - (90 * 0.04 - 30 * 0.04) / 120) < 1e-9, e
    assert e['above'] == 90 and e['n'] == 120, e
    assert evaluate_rule(mk(0.01, 40), plan, [0, 0, 0])['verdict'] == 'FAIL'                 # mean gain +0.01 < 0.02
    assert evaluate_rule(mk(0.04, 40), plan, [0, 0.02, 0])['verdict'] == 'INVALID'           # a fold skipped 2% of micro-batches
    r = mk(0.04, 40); r['a0']['finetuned'] = 0.0
    assert evaluate_rule(r, plan, [0, 0, 0])['verdict'] == 'INVALID'                         # an AUC of 0
    r = mk(0.04, 40); del r['c39']
    assert evaluate_rule(r, plan, [0, 0, 0])['verdict'] == 'UNKNOWN'                         # a planned window missing
    assert evaluate_rule({k: v for k, v in mk(0.04, 40).items() if k[0] != 'c'}, plan, [0, 0])['verdict'] == 'UNKNOWN'   # fold 2 not done
    log('SELFTEST ok (folds, patch guard, six-segment plan, train command, rule)')


def check():
    selftest()
    pj = jload(os.path.join(RESULTS, 'plan.json'))
    log('CHECK', 'ok  ' if pj else 'info', 'plan.json present' if pj else 'plan.json not written yet (run plan after carry)')
    log('CHECK delegating to V-031 check (env, villa commit, CKPT, imports, bucket URLs)')
    ok = v31.check()
    log('CHECK OK' if ok else 'CHECK FAIL')
    return ok


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd in ('check', '--check'):
        sys.exit(0 if check() else 1)
    os.makedirs(v31.DATA, exist_ok=True)
    if cmd == 'plan':
        plan_stage()
    elif cmd == 'prep':
        prep(set(args))
    elif cmd == 'train':
        train(int(args[0]))
    elif cmd == 'eval':
        evaluate(int(args[0]))
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
