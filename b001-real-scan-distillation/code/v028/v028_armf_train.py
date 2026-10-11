"""Shared 9 um fine-tuning helpers, trimmed for this repository to what b001_train.py imports: villa loss path, model
build, crop sampler, optimiser, resume and export, and the --check round trip. b001_train.py overrides the recipe
constants it changes (steps, batch, warm-up, trained parameters) and supplies its own tile loader. Function bodies are
unchanged from the version that ran, except that check() here always uses a synthetic tile.
Env: V026_VILLA_SRC (villa src at e0bbb8b), V026_CKPT (ink_9um step-075000.pth), INK_DEVICE (optional).
"""
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', 'v026'), HERE):
    if os.path.isdir(p):
        sys.path.insert(0, p)

STEPS, BATCH, LR, WARMUP = 500, 8, 1e-3, 50
MOMENTUM, WD, CLIP = 0.99, 3e-5, 1.0
SAVE_EVERY = 100
CROP, DEPTH, JITTER, NLAYERS = 128, 17, 2, 28
MIN_SUP_FRAC = 0.05
SEED = 280
FROZEN_PREFIXES = ('depth_fusion.', 'network.shared_encoder.')
TRAINED_PREFIXES = ('network.shared_decoder.', 'network.task_heads.')
BUDGET = float(os.environ.get('V028F_BUDGET', '6000'))


def villa():
    import v026_read as vr
    vr.villa_infer()                                   # registers the light vesuvius package shim (no __init__)
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.models.model import make_model
    from vesuvius.ink_detection.training.losses import create_loss
    from vesuvius.ink_detection.training.deep_supervision import concatenate_deep_supervision_ignore
    from vesuvius.ink_detection.inference.infer import normalize_flat_patch, select_layer_indices
    return dict(InkConfig=InkConfig, make_model=make_model, create_loss=create_loss,
                concat_ignore=concatenate_deep_supervision_ignore, normalize=normalize_flat_patch,
                select_layers=select_layer_indices, prepare_loss_inputs=prepare_loss_inputs_flat, vr=vr)


def prepare_loss_inputs_flat(predictions, batch, *, mode):
    """villa training/train.py prepare_loss_inputs at e0bbb8b, flat branch, verbatim in effect (train.py itself is not
    imported: it pulls optional training-only dependencies such as cc3d)."""
    import torch
    import torch.nn.functional as F
    assert mode == 'flat'
    if isinstance(predictions, (list, tuple)):
        prepared, targets, ignore_mask = [], None, None
        for index, level in enumerate(predictions):
            p, t, i = prepare_loss_inputs_flat(level, batch, mode=mode)
            prepared.append(p)
            if index == 0:
                targets, ignore_mask = t, i
        return type(predictions)(prepared), targets, ignore_mask
    targets = (torch.amax(batch['inklabels'], dim=2) > 0).to(dtype=batch['inklabels'].dtype)
    supervision = torch.amax(batch['supervision_mask'], dim=2)
    ignore_mask = (supervision <= 0).to(dtype=targets.dtype)
    output_size = tuple(int(v) for v in predictions.shape[-2:])
    if tuple(int(v) for v in targets.shape[-2:]) != output_size:
        targets = F.interpolate(targets.float(), size=output_size, mode='nearest').to(dtype=batch['inklabels'].dtype)
        ignore_mask = F.interpolate(ignore_mask.float(), size=output_size, mode='nearest').to(dtype=targets.dtype)
    return predictions, targets, ignore_mask


def build(V, ckpt_path, device):
    import torch
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = V['InkConfig'].from_mapping(payload['config'])
    assert cfg.data.mode == 'flat' and tuple(cfg.model.crop_size) == (DEPTH, CROP, CROP), cfg.model.crop_size
    model = V['make_model'](cfg)
    model.load_state_dict(payload['model'], strict=True)
    names = [n for n, _ in model.named_parameters()]
    assert all(n.startswith(FROZEN_PREFIXES + TRAINED_PREFIXES) for n in names), [n for n in names if not n.startswith(FROZEN_PREFIXES + TRAINED_PREFIXES)]
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(TRAINED_PREFIXES)
    model.to(device)
    return payload, cfg, model


def set_mode(model):
    model.train()
    model.depth_fusion.eval()
    model.network.shared_encoder.eval()


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / max(1, STEPS - WARMUP)))


class Tiles:
    """Training tiles in memory: items = [(ct uint8 28 x H x W, ink bool, loss mask bool)], w = sampling weights.
    b001_train.py has its own loader; check() fills one synthetic tile."""


def sample(tiles, rng, V):
    """One training example: (image 17 x 128 x 128 float32 normalised, ink 128 x 128, mask 128 x 128)."""
    centre = V['select_layers'](NLAYERS, layer_start=None, layer_end=None, output_depth=DEPTH, direction='forward')
    assert centre.tolist() == list(range(6, 23)), centre
    while True:
        ct, ink, m = tiles.items[rng.choice(len(tiles.items), p=tiles.w)]
        H, W = m.shape
        y, x = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
        mm = m[y:y + CROP, x:x + CROP]
        if mm.mean() < MIN_SUP_FRAC:
            continue
        o = int(rng.integers(-JITTER, JITTER + 1))
        img = ct[centre + o, y:y + CROP, x:x + CROP]
        if not img.any():
            continue
        lab = ink[y:y + CROP, x:x + CROP]
        k, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        img, lab, mm = (np.rot90(a, k, axes=(-2, -1)) for a in (img, lab, mm))
        if flip:
            img, lab, mm = (a[..., ::-1] for a in (img, lab, mm))
        img = V['normalize'](np.ascontiguousarray(img).astype(np.float32), 'tifxyz_robust')
        return img, np.ascontiguousarray(lab), np.ascontiguousarray(mm)


def step_once(model, loss_fn, V, batch, device, scaler, opt, step):
    import torch
    img = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None].to(device)
    lab = torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.float32))[:, None, None].to(device)
    sup = torch.from_numpy(np.stack([b[2] for b in batch]).astype(np.float32))[:, None, None].to(device)
    for g in opt.param_groups:
        g['lr'] = lr_at(step)
    set_mode(model)
    use_amp = device.type == 'cuda'
    with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        out = model(img)['ink']
    pred, targ, ign = V['prepare_loss_inputs'](out, {'image': img, 'inklabels': lab, 'supervision_mask': sup}, mode='flat')
    twi = V['concat_ignore'](targ, ign, pred)
    if isinstance(pred, (list, tuple)):
        loss = loss_fn(type(pred)(p.float() for p in pred), type(twi)(t.float() for t in twi))
    else:
        loss = loss_fn(pred.float(), twi.float())
    if not torch.isfinite(loss):
        raise RuntimeError(f'non-finite loss at step {step}')
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP)
    scaler.step(opt)
    scaler.update()
    return float(loss.detach())


def make_opt(model, device):
    import torch
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=MOMENTUM, nesterov=True,
                          weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    return opt, scaler


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_state(path, model, opt, scaler, step, rng, losses):
    import torch
    torch.save({'trainable': trainable_state(model), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                'step': step, 'rng': rng.bit_generator.state, 'losses': losses,
                'torch_rng': torch.get_rng_state()}, path + '.part')
    os.replace(path + '.part', path)


def load_state(path, model, opt, scaler, rng):
    import torch
    s = torch.load(path, map_location='cpu', weights_only=False)
    missing = model.load_state_dict(s['trainable'], strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    opt.load_state_dict(s['opt'])
    scaler.load_state_dict(s['scaler'])
    rng.bit_generator.state = s['rng']
    torch.set_rng_state(s['torch_rng'])
    return s['step'], s['losses']


def export(path, payload, model, info):
    import torch
    full = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert set(full) == set(payload['model']), 'state keys differ from the base checkpoint'
    out = {'model': full, 'config': payload['config'], 'step': payload.get('step'), 'armf': info}
    torch.save(out, path + '.part')
    os.replace(path + '.part', path)


def device_of():
    import torch
    d = os.environ.get('INK_DEVICE')
    return torch.device(d if d else ('cuda' if torch.cuda.is_available() else 'cpu'))


def check(data=None):
    import torch
    V = villa()
    base = os.environ['V026_CKPT']
    sha = V['vr'].sha256(base)
    dev = device_of()
    payload, cfg, model = build(V, base, dev)
    loss_fn = V['create_loss'](cfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trained_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    rng = np.random.default_rng(0)
    src = 'synthetic'
    if src == 'synthetic':
        tiles = Tiles.__new__(Tiles)
        ct = rng.integers(1, 255, (NLAYERS, 160, 160), dtype=np.uint8)
        ink = np.zeros((160, 160), bool); ink[40:80, 20:140] = True
        tiles.items = [(ct, ink, np.ones((160, 160), bool))]; tiles.w = np.ones(1)
    opt, scaler = make_opt(model, dev)
    t0 = time.time()
    losses = [step_once(model, loss_fn, V, [sample(tiles, rng, V) for _ in range(2)], dev, scaler, opt, s) for s in range(2)]
    dt = time.time() - t0
    frozen_same = all(torch.equal(frozen_before[n], p.detach()) for n, p in model.named_parameters() if not p.requires_grad)
    trained_moved = any(not torch.equal(trained_before[n], p.detach()) for n, p in model.named_parameters() if p.requires_grad)
    with tempfile.TemporaryDirectory() as tmp:
        save_state(os.path.join(tmp, 's.pt'), model, opt, scaler, 2, rng, losses)
        p2, _, m2 = build(V, base, dev)
        o2, s2 = make_opt(m2, dev)
        st, _ = load_state(os.path.join(tmp, 's.pt'), m2, o2, s2, np.random.default_rng(1))
        resume_ok = st == 2 and all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), m2.state_dict().values()))
        export(os.path.join(tmp, 'f.pth'), payload, model, {'check': True})
        e = torch.load(os.path.join(tmp, 'f.pth'), map_location='cpu', weights_only=False)
        m3 = V['make_model'](V['InkConfig'].from_mapping(e['config']))
        m3.load_state_dict(e['model'], strict=True)
        export_ok = all(torch.equal(a.cpu(), b) for a, b in zip(model.state_dict().values(), m3.state_dict().values()))
        # the exported file loads through the read path's own loader too (as v026_read will)
        os.environ['V026_CKPT'] = os.path.join(tmp, 'f.pth')
        try:
            inf, args, cm, d2, info = V['vr'].load_model()
            readpath_ok = cm.input_depth == DEPTH and cm.patch_size == CROP
        finally:
            os.environ['V026_CKPT'] = base
    ok = (sha == V['vr'].CKPT_SHA256 and frozen_same and trained_moved and all(np.isfinite(losses)) and resume_ok
          and export_ok and readpath_ok)
    print('CHECK', 'OK' if ok else 'FAIL', json.dumps({
        'device': str(dev), 'sha256_ok': sha == V['vr'].CKPT_SHA256, 'trainable_params': n_tr, 'all_params': n_all,
        'frozen_unchanged': frozen_same, 'trained_moved': trained_moved, 'loss_finite': bool(np.all(np.isfinite(losses))),
        'resume_ok': resume_ok, 'export_strict_reload_ok': export_ok, 'read_path_loads_export': readpath_ok,
        'data': src, 'seconds_2_steps_batch2': round(dt, 1)}), flush=True)
    return ok
