"""V-016: adapt the canonical 2.4 um ink model to the PHerc1667 1.129 um (59 keV) scan.

Labels drawn on the 2.399 um volume supervise 1.129 um renders along the landmark-refit mesh (arm C, V-014), which
V-014's model-free placement test showed lands on the labelled papyrus. Leave-a-pair-out folds; the test windows are
exactly the V-014 windows, and every training patch keeps clear of them.

Runs inside the pinned cross-scan-ink-transfer src tree next to v014_transform_ink.py.
usage:
  v016_cross_scan_ink.py prep  <arm> <segment> [...]      render training patches + test windows (arm C or P; R test only)
  v016_cross_scan_ink.py train <arm> <fold>               fine-tune; resumable from ckpt (V016_ITERS iterations)
  v016_cross_scan_ink.py eval  <arm> <fold>               zero-shot vs fine-tuned AUC on the held-out pair (+ R)
  v016_cross_scan_ink.py baseline <arm> <segment>        zero-shot check against V-014
  v016_cross_scan_ink.py verdict <arm>                    combine the three folds
Env: V016_DATA (renders, labels, checkpoints), V014_DATA (V-014 meshes), INK_DEVICE, CANON_CKPT, VILLA_INFERENCE_DIR.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy import ndimage

os.environ['V014_DRYRUN'] = '1'
import v014_transform_ink as v  # noqa: E402

mb = v.mb
DATA = os.environ.get('V016_DATA', 'out/v016')
RUN = os.environ.get('PUZZLE_RUN_DIR', '.')
SEGS = ['20240304141531-w013_20240304141531_flatboi', '20240304144031-w018_20240304144031_flatboi',
        '20240304161941-w023_20240304161941_flatboi', '20251208130119-w028_20251208130119156_flatboi',
        '20251212185248-w029_20251212185248662_flatboi', '20251223230000-w031_2025122323_flatboi']
FOLDS = [SEGS[0:2], SEGS[2:4], SEGS[4:6]]                 # held-out pairs
PATCH_H, PATCH_W, MAX_PATCHES, GUARD = 40, 80, 4, 10      # grid cells; GUARD = cells kept clear around the test window
LAYERS = 63                                               # centre 62 = V-014 offset-0 window (c = 31 -> layers 0..61)
TILE, OUT_TILE = 256, 64
NEAR_PX = 2.0 / 2.399e-3                                  # negatives: within 2 mm of ink (same rule as eval_map)
ITERS = int(os.environ.get('V016_ITERS', '2000'))
BATCH, LR, WD, SEED = 2, 2e-5, 1e-4, 0
ARMS = {'C': (lambda d: os.path.join(d, 'mesh_C'), v.VOL_1129, ['--level', '1', '--oversample', '2']),
        'P': (lambda d: os.path.join(d, 'mesh_P'), v.VOL_1129, ['--level', '1', '--oversample', '2']),
        'R': (lambda d: os.path.join(d, 'mesh_2399'), v.VOL_2399, ['--level', '0', '--oversample', '1'])}


def render(arm, seg, rows, cols, out):
    if os.path.exists(out + '.npy'):
        return
    mdir, url, extra = ARMS[arm]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.check_call([mb.PY, os.path.join(mb.HERE, 'render_crop.py'), mdir(os.path.join(v.OUT, seg)), url, out,
                           '--rows', str(rows[0]), str(rows[1]), '--cols', str(cols[0]), str(cols[1]),
                           '--layers', str(LAYERS), '--smooth-normals', '1.0', '--cache-chunks', '1500'] + extra,
                          stdout=subprocess.DEVNULL)


def plan(seg):
    """V-014 test window and up to MAX_PATCHES training patches (densest labelled, in-volume, clear of the window)."""
    d = os.path.join(DATA, seg); os.makedirs(d, exist_ok=True)
    pj = os.path.join(d, 'plan.json')
    if os.path.exists(pj):
        return json.load(open(pj)), None
    Mpub, Mlsq, _ = v.transforms()
    r = v.run(seg, Mpub, Mlsq)                               # dryrun: window + mesh_P/mesh_C under V014_DATA
    r0, r1, c0, c1 = r['window_cells']
    lab, _ = v.labels(seg, os.path.join(v.OUT, seg))
    _, xyz, valid = v.mesh2399(seg, os.path.join(v.OUT, seg))
    zs, ys, xs = v.shape_l0(v.VOL_1129)
    inside = lambda Q: ((Q >= 0) & (Q <= np.array([xs - 1, ys - 1, zs - 1]))).all(-1)
    ok = ~valid | (inside(v.inv_apply(Mpub, xyz)) & inside(v.inv_apply(Mlsq, xyz)))
    Hc, Wc = min(lab.shape[0] // 20, xyz.shape[0]), min(lab.shape[1] // 20, xyz.shape[1])
    cells = lab[:Hc * 20, :Wc * 20].reshape(Hc, 20, Wc, 20).mean((1, 3), dtype=np.float32)
    block = ~ok[:Hc, :Wc]
    block[max(0, r0 - GUARD):r1 + GUARD, max(0, c0 - GUARD):c1 + GUARD] = True
    patches = []
    for _ in range(MAX_PATCHES):
        dens = ndimage.uniform_filter(cells, (PATCH_H, PATCH_W), mode='constant')
        bad = ndimage.maximum_filter(block.astype(np.uint8), (PATCH_H, PATCH_W), mode='constant', cval=1) > 0
        dm = np.where(bad, -1, dens)
        cy, cx = np.unravel_index(np.argmax(dm), dm.shape)
        if dm[cy, cx] <= 0 or (patches and dm[cy, cx] < 0.2 * patches[0]['density']):
            break
        pr0, pc0 = int(cy - PATCH_H // 2), int(cx - PATCH_W // 2)
        patches.append({'cells': [pr0, pr0 + PATCH_H, pc0, pc0 + PATCH_W], 'density': float(dm[cy, cx])})
        block[pr0:pr0 + PATCH_H, pc0:pc0 + PATCH_W] = True
    p = {'segment': seg, 'test_cells': [r0, r1, c0, c1], 'patches': patches}
    for k, q in enumerate(patches):
        a0, a1, b0, b1 = q['cells']
        np.save(os.path.join(d, f'train{k}_lab.npy'), np.packbits(lab[a0 * 20:a1 * 20, b0 * 20:b1 * 20], axis=None))
        q['lab_shape'] = [(a1 - a0) * 20, (b1 - b0) * 20]
    np.save(os.path.join(d, 'test_lab8.npy'), mb.down(lab).astype(np.float32))
    json.dump(p, open(pj, 'w'), indent=1)
    return p, lab


def prep(arm, segs):
    for seg in segs:
        p, _ = plan(seg)
        d = os.path.join(DATA, seg)
        r0, r1, c0, c1 = p['test_cells']
        render(arm, seg, (r0, r1), (c0, c1), os.path.join(d, f'test_{arm}'))
        if arm != 'R':
            for k, q in enumerate(p['patches']):
                a0, a1, b0, b1 = q['cells']
                render(arm, seg, (a0, a1), (b0, b1), os.path.join(d, f'train{k}_{arm}'))
        print(seg, arm, 'test', p['test_cells'], 'patches', [q['cells'] for q in p['patches']], flush=True)


# ---------------------------------------------------------------------------------------------------------- training
def torch_setup():
    import torch
    sys.path.insert(0, os.environ.get('VILLA_INFERENCE_DIR', 'villa/ink-detection/optimized_inference'))
    from model_resnet3d_3d_decoder import load_model
    dev = torch.device(os.environ.get('INK_DEVICE', 'cuda'))
    return torch, load_model, dev


def load_patches(arm, segs):
    out = []
    for seg in segs:
        d = os.path.join(DATA, seg); p = json.load(open(os.path.join(d, 'plan.json')))
        for k, q in enumerate(p['patches']):
            ct = np.load(os.path.join(d, f'train{k}_{arm}.npy'), mmap_mode='r')[:62]
            h, w = q['lab_shape']
            lab = np.unpackbits(np.load(os.path.join(d, f'train{k}_lab.npy')))[:h * w].reshape(h, w)
            h, w = min(h, ct.shape[1]), min(w, ct.shape[2])
            near = ndimage.distance_transform_edt(lab[:h, :w] == 0) <= NEAR_PX
            ok = (np.asarray(ct[:, :h, :w]) != 0).any(0)
            out.append((np.ascontiguousarray(ct[:, :h, :w]), lab[:h, :w].astype(np.float32), (near & ok).astype(np.float32)))
    return out


def sample(patches, rng):
    """One 256 tile with >= 2% ink and >= 50% supervised pixels; random flips/rot90 and mild contrast jitter."""
    while True:
        ct, lab, m = patches[rng.integers(len(patches))]
        y, x = rng.integers(0, ct.shape[1] - TILE + 1), rng.integers(0, ct.shape[2] - TILE + 1)
        L, M = lab[y:y + TILE, x:x + TILE], m[y:y + TILE, x:x + TILE]
        if M.mean() < 0.5 or L.mean() < 0.02:
            continue
        X = np.clip(ct[:, y:y + TILE, x:x + TILE].astype(np.float32), 0, 200) / 200.0
        k = rng.integers(4); X, L, M = np.rot90(X, k, (1, 2)), np.rot90(L, k), np.rot90(M, k)
        if rng.random() < 0.5:
            X, L, M = X[:, :, ::-1], L[:, ::-1], M[:, ::-1]
        g, b = rng.uniform(0.9, 1.1), rng.uniform(-0.05, 0.05)
        X = np.where(X > 0, np.clip(X * g + b, 0, 1), 0)
        pool = lambda a: a.reshape(OUT_TILE, 4, OUT_TILE, 4).mean((1, 3))
        return np.ascontiguousarray(X), pool(np.ascontiguousarray(L)), (pool(np.ascontiguousarray(M)) > 0.5).astype(np.float32)


def train(arm, fold):
    torch, load_model, dev = torch_setup()
    from torch.utils.checkpoint import checkpoint_sequential
    held = FOLDS[fold]; segs = [s for s in SEGS if s not in held]
    cdir = os.path.join(DATA, f'ckpt_{arm}_fold{fold}'); os.makedirs(cdir, exist_ok=True)
    final = os.path.join(cdir, 'final.pt')
    if os.path.exists(final):
        print('fold', fold, 'already trained'); return
    torch.manual_seed(SEED)
    net = load_model(os.environ['CANON_CKPT'], dev).model
    net.eval()                                            # BatchNorm statistics frozen; no deep-supervision heads
    bb = net.backbone
    for prm in net.parameters():
        prm.requires_grad_(True)

    def fwd(x):
        if net.normalization is not None:                 # amendment 1: same input normalisation as inference
            x = net.normalization(x)
        x = bb.relu(bb.bn1(bb.conv1(x)))
        if not bb.no_max_pool:
            x = bb.maxpool(x)
        feats = []
        for layer in (bb.layer1, bb.layer2, bb.layer3, bb.layer4):
            x = checkpoint_sequential(layer, max(1, len(layer) // 3), x, use_reentrant=False)
            feats.append(x)
        return net.decoder(feats)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 100) * 0.5 * (1 + np.cos(np.pi * min(i, ITERS) / ITERS)))
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
    it0, rng, log, skipped = 0, np.random.default_rng(SEED + fold), [], 0
    ck = sorted(glob.glob(os.path.join(cdir, 'iter*.pt')))
    if ck:
        s = torch.load(ck[-1], map_location='cpu', weights_only=False)
        net.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        scaler.load_state_dict(s['scaler']); it0 = s['iter']; rng.bit_generator.state = s['rng']; log = s['log']
        skipped = s.get('skipped', 0)
        print('resumed at', it0, flush=True)
    patches = load_patches(arm, segs)
    print('fold', fold, 'train segs', [s.split('-')[1][:4] for s in segs], 'patches', len(patches), flush=True)
    t0 = time.time()
    for it in range(it0, ITERS):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(BATCH):                            # micro-batches of 1 (BatchNorm frozen, so identical to batch BATCH)
            X, Y, M = (torch.from_numpy(a[None, None].copy()).to(dev) for a in sample(patches, rng))
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                logit = fwd(X)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), Y, reduction='none') * M).sum() / M.sum().clamp(min=1)
            if not torch.isfinite(loss):                  # amendment 1: drop a non-finite micro-batch
                skipped += 1
                continue
            scaler.scale(loss / BATCH).backward()
            tot += loss.item() / BATCH
        scaler.unscale_(opt)
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)):
            opt.zero_grad(set_to_none=True); skipped += 1  # amendment 1: never step on non-finite gradients
        scaler.step(opt); scaler.update(); sched.step()
        log.append(tot)
        if (it + 1) % 25 == 0:
            print(f'it {it + 1}/{ITERS} loss {np.mean(log[-25:]):.4f} {time.time() - t0:.0f}s', flush=True)
        if (it + 1) % 200 == 0 or it + 1 == ITERS:
            tmp = os.path.join(cdir, f'iter{it + 1:06d}.pt')
            torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                        'iter': it + 1, 'rng': rng.bit_generator.state, 'log': log, 'skipped': skipped}, tmp + '.part')
            os.replace(tmp + '.part', tmp)
            for old in sorted(glob.glob(os.path.join(cdir, 'iter*.pt')))[:-1]:
                os.remove(old)
    assert all(torch.isfinite(t).all() for t in net.state_dict().values() if t.is_floating_point()), 'non-finite weights'
    torch.save({'state_dict': net.state_dict()}, final + '.part'); os.replace(final + '.part', final)
    json.dump({'loss': log, 'iters': ITERS, 'batch': BATCH, 'lr': LR, 'train_segments': segs, 'skipped_nonfinite': skipped},
              open(os.path.join(cdir, 'train_log.json'), 'w'))


# ---------------------------------------------------------------------------------------------------------- eval
def infer(model_path, stack, out):
    if os.path.exists(out):
        return np.load(out)
    torch, load_model, dev = torch_setup()
    import infer_crop
    w = load_model(model_path, dev); w.eval()
    infer_crop.run(w, stack, argparse.Namespace(tile=TILE, stride=128, batch=4), dev, out)
    return np.load(out)


def evaluate(arm, fold):
    cdir = os.path.join(DATA, f'ckpt_{arm}_fold{fold}')
    res = {'fold': fold, 'arm': arm, 'segments': {}}
    for seg in FOLDS[fold]:
        d = os.path.join(DATA, seg); p = json.load(open(os.path.join(d, 'plan.json')))
        r0, _, c0, _ = p['test_cells']; lab8 = np.load(os.path.join(d, 'test_lab8.npy')); e0 = (r0 * 20 // mb.DS, c0 * 20 // mb.DS)
        r = {}
        for a in (arm, 'R'):
            st = os.path.join(d, f'test_{a}.npy')
            if not os.path.exists(st):
                continue
            stack = np.load(st, mmap_mode='r')[:62]
            r[f'{a}_zero_shot'] = mb.eval_map(infer(os.environ['CANON_CKPT'], stack, os.path.join(d, f'ink_{a}_zero.npy')), lab8, e0)
            r[f'{a}_finetuned'] = mb.eval_map(infer(os.path.join(cdir, 'final.pt'), stack, os.path.join(cdir, f'ink_{a}_{seg.split("-")[1][:4]}.npy')), lab8, e0)
        res['segments'][seg] = r
        print(seg, ' '.join(f"{k} {x['auc']:.4f} {x['offset']}" for k, x in r.items()), flush=True)
    os.makedirs(os.path.join(RUN, 'payload'), exist_ok=True)
    json.dump(res, open(os.path.join(cdir, 'eval.json'), 'w'), indent=1)
    json.dump(res, open(os.path.join(RUN, 'payload', f'v016-eval-{arm}-fold{fold}.json'), 'w'), indent=1)


def baseline(arm, seg):
    """Zero-shot AUC of one segment's test stack (should equal V-014's offset-0 number for that arm)."""
    d = os.path.join(DATA, seg); p = json.load(open(os.path.join(d, 'plan.json')))
    r0, _, c0, _ = p['test_cells']; lab8 = np.load(os.path.join(d, 'test_lab8.npy'))
    stack = np.load(os.path.join(d, f'test_{arm}.npy'), mmap_mode='r')[:62]
    r = mb.eval_map(infer(os.environ['CANON_CKPT'], stack, os.path.join(d, f'ink_{arm}_zero.npy')), lab8, (r0 * 20 // mb.DS, c0 * 20 // mb.DS))
    print(seg, arm, 'zero-shot', json.dumps(r), flush=True)


def verdict(arm):
    segs = {}
    for f in range(3):
        ej = os.path.join(DATA, f'ckpt_{arm}_fold{f}', 'eval.json')
        if os.path.exists(ej):
            segs.update(json.load(open(ej))['segments'])
    ft = [s[f'{arm}_finetuned']['auc'] for s in segs.values()]
    zs = [s[f'{arm}_zero_shot']['auc'] for s in segs.values()]
    above = sum(a > b for a, b in zip(ft, zs))
    if len(ft) < 6:
        vd = 'UNKNOWN'
    else:
        vd = 'PASS' if np.mean(ft) >= 0.75 and above >= 5 else 'FAIL'
    out = {'arm': arm, 'n': len(ft), 'mean_finetuned': float(np.mean(ft)) if ft else None,
           'mean_zero_shot': float(np.mean(zs)) if zs else None, 'above_zero_shot': int(above), 'verdict': vd, 'segments': segs}
    os.makedirs(os.path.join(RUN, 'payload'), exist_ok=True)
    json.dump(out, open(os.path.join(RUN, 'payload', f'v016-verdict-{arm}.json'), 'w'), indent=1)
    print('VERDICT', {k: out[k] for k in out if k != 'segments'}, flush=True)


def main():
    cmd, arm = sys.argv[1], sys.argv[2]
    assert arm in ARMS
    if cmd == 'prep':
        prep(arm, sys.argv[3:] or SEGS)
    elif cmd == 'train':
        train(arm, int(sys.argv[3]))
    elif cmd == 'eval':
        evaluate(arm, int(sys.argv[3]))
    elif cmd == 'baseline':
        baseline(arm, sys.argv[3])
    elif cmd == 'verdict':
        verdict(arm)


if __name__ == '__main__':
    main()
