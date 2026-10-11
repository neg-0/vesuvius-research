"""B-001 training: fine-tune the organisers' 9 um ink model (scrollprize/ink_9um hybrid_3d2d-seed42 step-075000) on real
9.36 / 8.64 um surface-volume tiles of the three training scrolls, with the canonical 2.4 um model's published map as the
teacher (b001_data.py tiles + fetch).

usage: b001_train.py train <out dir>      (GPU desktop; resumable; stops cleanly and saves after B001_BUDGET s, default
                                           6000; rerun the same command to resume)
       b001_train.py --check              (villa import, strict checkpoint load, 2 tiny steps, resume and export round
                                           trip, read-path load of the export; nothing kept)

Recipe (fixed in experiments/b001/README.md before any B-001 model output; no search, no checkpoint selection). It is
V-028 arm F's trainer (v028_armf_train.py, imported unchanged: villa loss, crop sampler, normalisation, optimiser,
resume, export) with these registered differences:
- all parameters trained (arm F froze the 3D stem and the 2D encoder);
- 6,000 steps of batch 16, peak LR 1e-3, 200 warm-up steps then cosine to 0; seed 4500;
- tiles from results/b001/tiles.json, CT from $B001_DATA/tiles/<tile>.npy (28 x 260 x 260, dz applied), labels from
  results/b001/tile_labels/<tile>.npz (teacher >= 160 ink, <= 50 no ink, else ignored); a tile is drawn with probability
  proportional to its supervised CT-valid px, each scroll's tiles scaled so the three scrolls weigh the same.
Output: <out>/b001_real.pth (the checkpoint's own format; the provenance block sits under 'armf', the key v028's export
uses), <out>/train_log.json.
Env: V026_VILLA_SRC (villa src at e0bbb8b), V026_CKPT (ink_9um step-075000.pth), B001_DATA, B001_RESULTS, INK_DEVICE,
B001_BUDGET; for B-001r arms (experiments/b001r/README.md) B001_SEED, B001_TILES, B001_CKPT_NAME; for B-001s B001_EXCLUDE_SCROLL, B001_STEPS (defaults = B-001).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', 'v026'), os.path.join(HERE, '..', 'v028'), HERE):
    sys.path.insert(0, p)
import v028_armf_train as f                                         # noqa: E402  (arm F trainer, unchanged)
import b001_data as bd                                              # noqa: E402

f.STEPS, f.BATCH, f.LR, f.WARMUP = 6000, 16, 1e-3, 200
f.FROZEN_PREFIXES, f.TRAINED_PREFIXES = (), ('depth_fusion.', 'network.')
f.set_mode = lambda model: model.train()                            # nothing frozen, so no module stays in eval mode
SEED = int(os.environ.get('B001_SEED', '4500'))                    # B-001: 4500; B-001r arms set their own
TILES = os.environ.get('B001_TILES', 'tiles')                      # B-001: real LR tiles; B-001r SIM: 'tiles_sim'
CKPT_NAME = os.environ.get('B001_CKPT_NAME', 'b001_real.pth')
EXCLUDE = os.environ.get('B001_EXCLUDE_SCROLL', '')                # B-001s folds: hold one scroll out of training
f.STEPS = int(os.environ.get('B001_STEPS', str(f.STEPS)))
BUDGET = float(os.environ.get('B001_BUDGET', '6000'))


class Tiles:
    """All training tiles in memory: (ct uint8 28 x 260 x 260, ink & supervised & CT-valid, supervised & CT-valid)."""

    def __init__(self, reg):
        self.items, w, scroll = [], [], []
        for n, t in reg['tiles'].items():
            if t['scroll'] == EXCLUDE:
                continue
            ct = np.load(os.path.join(bd.DATA, TILES, n + '.npy'))
            assert ct.shape == (f.NLAYERS, *t['px']) and ct.dtype == np.uint8, (n, ct.shape)
            ink, sup = bd.load_labels('tile', n)
            valid = (ct != 0).any(0)
            m = sup & valid
            if not m.any():
                continue
            self.items.append((ct, ink & m, m))
            w.append(float(m.sum()))
            scroll.append(t['scroll'])
        w, scroll = np.array(w), np.array(scroll)
        self.per_scroll = {}
        for sc in sorted(set(scroll)):
            k = scroll == sc
            self.per_scroll[sc] = int(k.sum())
            w[k] /= w[k].sum()
        self.w = w / w.sum()


def train(out):
    import torch
    V = f.villa()
    base = os.environ['V026_CKPT']
    sha = V['vr'].sha256(base)
    assert sha == V['vr'].CKPT_SHA256, ('unexpected base checkpoint', sha)
    reg = json.load(open(os.path.join(bd.RESULTS, 'tiles.json')))
    missing = [n for n, t in reg['tiles'].items() if t['scroll'] != EXCLUDE and not os.path.exists(os.path.join(bd.DATA, TILES, n + '.npy'))]
    if missing:
        raise SystemExit(f'{len(missing)} tiles not fetched (b001_data.py fetch), e.g. {missing[0]}')
    os.makedirs(out, exist_ok=True)
    final = os.path.join(out, CKPT_NAME)
    if os.path.exists(final):
        print('DONE already', final, flush=True)
        return
    dev = f.device_of()
    torch.manual_seed(SEED)
    payload, cfg, model = f.build(V, base, dev)
    loss_fn = V['create_loss'](cfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    assert n_tr == n_all, (n_tr, n_all)
    opt, scaler = f.make_opt(model, dev)
    rng = np.random.default_rng(SEED)
    sp = os.path.join(out, 'state.pt')
    step, losses = 0, []
    if os.path.exists(sp):
        step, losses = f.load_state(sp, model, opt, scaler, rng)
        print('resumed at step', step, flush=True)
    tiles = Tiles(reg)
    print('B-001 train', json.dumps({'device': str(dev), 'tiles': len(tiles.items), 'per_scroll': tiles.per_scroll,
                                     'trainable_params': n_tr, 'steps': f.STEPS, 'batch': f.BATCH, 'lr': f.LR}), flush=True)
    t0 = time.time()
    while step < f.STEPS:
        batch = [f.sample(tiles, rng, V) for _ in range(f.BATCH)]
        losses.append(f.step_once(model, loss_fn, V, batch, dev, scaler, opt, step))
        step += 1
        if step % f.SAVE_EVERY == 0 or step == f.STEPS:
            f.save_state(sp, model, opt, scaler, step, rng, losses)
            print(f'step {step}/{f.STEPS} loss(last 100) {np.mean(losses[-100:]):.4f} {time.time() - t0:.0f}s', flush=True)
        if time.time() - t0 > BUDGET and step < f.STEPS:
            f.save_state(sp, model, opt, scaler, step, rng, losses)
            print('BUDGET reached at step', step, '- rerun the same command to resume', flush=True)
            return
    info = {'study': 'B-001', 'tiles_dir': TILES, 'excluded_scroll': EXCLUDE or None, 'base_sha256': sha, 'tiles': len(tiles.items), 'per_scroll': tiles.per_scroll,
            'steps': f.STEPS, 'batch': f.BATCH, 'lr': f.LR, 'warmup': f.WARMUP, 'momentum': f.MOMENTUM,
            'weight_decay': f.WD, 'clip': f.CLIP, 'trained': 'all', 'trainable_params': n_tr, 'seed': SEED,
            'device': str(dev)}
    f.export(final, payload, model, info)
    log = dict(info, loss=losses, final_loss_last100=float(np.mean(losses[-100:])), sha256=V['vr'].sha256(final))
    json.dump(log, open(os.path.join(out, 'train_log.json'), 'w'), indent=1)
    run = os.environ.get('PUZZLE_RUN_DIR')
    if run:
        os.makedirs(os.path.join(run, 'payload'), exist_ok=True)
        json.dump(log, open(os.path.join(run, 'payload', 'b001-train.json'), 'w'), indent=1)
    print('DONE', final, 'sha256', log['sha256'], f"loss(last 100) {log['final_loss_last100']:.4f}", flush=True)


def main():
    a = sys.argv[1:]
    if a and a[0] == '--check':
        sys.exit(0 if f.check(None) else 1)
    if len(a) == 2 and a[0] == 'train':
        train(a[1])
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
