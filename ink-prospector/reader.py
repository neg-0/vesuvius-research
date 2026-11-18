"""Minimal standalone runner for koine_machines 'flat' readers (Reader V3 etc.) on a (D,H,W) uint8 layer stack."""
import sys, numpy as np, torch
def load_reader(ckpt, villa_root):
    sys.path[:0] = [f"{villa_root}/ink-detection", f"{villa_root}/vesuvius/src"]
    from koine_machines.inference import infer as I
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cm = I.build_repo_training_model_bundle(payload, ckpt)
    return cm
def run_stack(cm, stack, layer_start=None, reverse=False, overlap=0.5, batch=2, device="cpu"):
    """stack: (D,H,W) uint8, layers along the normal. Returns float32 (H,W) sigmoid ink score."""
    from vesuvius.image_proc.intensity.normalization import normalize_robust
    D, H, W = stack.shape; d = cm.in_chans; roi = cm.roi_size
    s0 = (D // 2 - d // 2) if layer_start is None else layer_start
    sub = stack[s0:s0 + d]
    if reverse: sub = sub[::-1]
    step = max(1, int(round(roi * (1 - overlap))))
    ys = sorted(set(list(range(0, max(H - roi, 0) + 1, step)) + [max(H - roi, 0)]))
    xs = sorted(set(list(range(0, max(W - roi, 0) + 1, step)) + [max(W - roi, 0)]))
    wy = torch.hann_window(roi, periodic=False); w2 = torch.clamp(torch.outer(wy, wy) / (wy.max() ** 2), min=1e-3).numpy()
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    jobs = [(y, x) for y in ys for x in xs]; model = cm.model.to(device)
    with torch.no_grad():
        for i in range(0, len(jobs), batch):
            tiles = []
            for (y, x) in jobs[i:i + batch]:
                p = np.zeros((d, roi, roi), np.uint8); t = sub[:, y:y + roi, x:x + roi]; p[:, :t.shape[1], :t.shape[2]] = t
                tiles.append(torch.from_numpy(np.ascontiguousarray(normalize_robust(p))).unsqueeze(0))
            out = torch.sigmoid(model(torch.stack(tiles).to(device))).float().cpu().numpy()
            out = out.reshape(len(tiles), roi, roi)
            for (y, x), o in zip(jobs[i:i + batch], out):
                hh, ww = min(roi, H - y), min(roi, W - x)
                acc[y:y + hh, x:x + ww] += o[:hh, :ww] * w2[:hh, :ww]; wsum[y:y + hh, x:x + ww] += w2[:hh, :ww]
    return acc / np.maximum(wsum, 1e-6)
def run_fused(cm, stack, **kw):
    """Mean of four reads of the same stack (as is, turned 90 and 180 degrees, mirrored). The reader's output depends
    on patch orientation; averaging removes that dependence. It does not recover more ink: an early gain on the
    answer-key sheet of a scroll that has a second, finer scan (recall 0.47 -> 0.58) came from the key having been read in one of the averaged
    orientations. Against an independent judge (a different reader on the finer scan, six targets) 1, 4 and 8 views all give
    r = 0.45 (2026-11-15)."""
    outs = []
    for k, fl in ((0, False), (1, False), (2, False), (0, True)):
        x = np.rot90(stack, k, (1, 2)); x = x[:, :, ::-1] if fl else x
        o = run_stack(cm, np.ascontiguousarray(x), **kw); o = o[:, ::-1] if fl else o; outs.append(np.rot90(o, -k))
    return np.mean(outs, 0)
if __name__ == "__main__":
    import time
    cm = load_reader(sys.argv[1], sys.argv[2]); print(cm.roi_size, cm.in_chans, cm.preprocessing)
    x = (np.random.rand(17, 384, 384) * 255).astype(np.uint8); t = time.time(); o = run_stack(cm, x); print(o.shape, o.mean(), time.time() - t, "s/tile")
