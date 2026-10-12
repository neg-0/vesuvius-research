"""B-001g: first 9 um ink read of an eligible Grand Prize scroll (PHerc0800, 8.64 um 116 keV) with the B-001 reader,
side by side with the stock ink_9um model. Maps only: no labels exist, nothing is scored. Spec: experiments/b001g/README.md.

usage: b001g_read.py check               (CPU: segment list, shapes, both B-001 checkpoints' sha256; network)
       b001g_read.py fetch               (CPU + network: whole published LR surface volumes, centre 28 layers -> $B001G_DIR/ct)
       b001g_read.py read <ckpt root>    (GPU: S0, REAL, R1 x forward, reverse -> $B001G_DIR/maps; resumable)
       b001g_read.py compose             (CPU: ds2 panels CT | S0 | REAL | R1 | mean(REAL, R1) -> $B001G_DIR/panels)
       b001g_read.py clean               (delete $B001G_DIR/ct)
       b001g_read.py control <ckpt root> (GPU: REAL on the layer-shuffled stack, forward -> maps/REAL_shuf; needs ct)
       b001g_read.py compare             (CPU: corr(REAL fwd, REAL_shuf fwd) on PHerc0800 vs on B-001's 49 test windows)
Env: V026_VILLA_SRC, V026_CKPT, B001G_DIR (default workspaces/vesuvius/b001/read_gp), INK_DEVICE.
"""
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', 'v026'), os.path.join(HERE, '..', 'b001')):
    sys.path.insert(0, p)
import b001_data as bd                                              # noqa: E402
import b001_read as br                                              # noqa: E402

SCROLL, SCAN = 'PHerc0800', '20250521135224'
SEGMENTS = ('20251029010146-auto_grown_20251029010146642', '20251028220042-auto_grown_20251028220042762',
            '20251028220955-auto_grown_20251028220955262', '20251028213516-auto_grown_20251028213516907')
SHAPES = {SEGMENTS[0]: [31, 980, 980], SEGMENTS[1]: [31, 2040, 2000], SEGMENTS[2]: [31, 2100, 2100],
          SEGMENTS[3]: [31, 2160, 2240]}
CKPTS = {'REAL': ('b001_real.pth', '1674820d073b82b00a2e0b230c8c939779e9eda84f8fa746904645c8f9760dfb'),
         'R1': ('r1/b001_r1.pth', '92aed6dd2305f8d1d825e7c902c45caab47291a125c58f87f3d4292df3deb7fa')}
DIRS = {'fwd': 'forward', 'rev': 'reverse'}
OUT = os.environ.get('B001G_DIR', os.path.join('workspaces', 'vesuvius', 'b001', 'read_gp'))


def sv(seg):
    return f'{SCROLL}/segments/{seg}/surface-volumes/8.64um-1.2m-116keV-volume-{SCAN}.zarr'


def ct_path(seg):
    return os.path.join(OUT, 'ct', seg + '.npy')


def map_path(arm, seg, d):
    return os.path.join(OUT, 'maps', arm, f'{seg}_{d}.png')


def check(ckpt_root=None):
    for seg in SEGMENTS:
        m = bd.zmeta(sv(seg), 0)
        assert m['shape'] == SHAPES[seg] and m['compressor'] is None, (seg, m['shape'])
        bd.zcrop(sv(seg), 4, 0, 0, 8, 8)
        print(f'CHECK {seg} LR sv {m["shape"]} ok', flush=True)
    mpx = sum(s[1] * s[2] for s in SHAPES.values()) / 1e6
    print(f'CHECK {len(SEGMENTS)} segments, {mpx:.1f} Mpx, CT {mpx * bd.NL / 1e3:.2f} GB as uint8 (28 layers)', flush=True)
    if ckpt_root:
        import v026_read as vr
        for arm, (f, sha) in CKPTS.items():
            got = vr.sha256(os.path.join(ckpt_root, f))
            assert got == sha, (arm, got)
            print(f'CHECK {arm} checkpoint sha256 ok', flush=True)
    print('CHECK OK', flush=True)


def fetch():
    t0 = time.time()
    for seg in SEGMENTS:
        if os.path.exists(ct_path(seg)):
            continue
        _, h, w = SHAPES[seg]
        st = bd.centre28(bd.zcrop(sv(seg), 0, 0, 0, h, w))       # 31 layers -> centre 28 (B-001's convention, dz 0)
        os.makedirs(os.path.dirname(ct_path(seg)), exist_ok=True)
        np.save(ct_path(seg) + '.part.npy', st)
        os.replace(ct_path(seg) + '.part.npy', ct_path(seg))
        print(f'FETCH {seg} {st.shape} {time.time() - t0:.0f}s', flush=True)
    print('FETCH done', len(SEGMENTS), 'segments', flush=True)


def read(ckpt_root):
    import cv2
    import v026_read as vr
    base = os.environ['V026_CKPT']
    assert vr.sha256(base) == vr.CKPT_SHA256, 'unexpected S0 checkpoint'
    arms = [('S0', base)] + [(a, os.path.join(ckpt_root, f)) for a, (f, _) in CKPTS.items()]
    for a, (f, sha) in CKPTS.items():
        assert vr.sha256(os.path.join(ckpt_root, f)) == sha, a
    tmp = os.path.join(OUT, 'tmp')
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    for arm, ckpt in arms:
        todo = [(s, d) for s in SEGMENTS for d in DIRS if not os.path.exists(map_path(arm, s, d))]
        if not todo:
            continue
        os.environ['V026_CKPT'] = ckpt
        try:
            inf, args, cm, dev, info = vr.load_model()
        finally:
            os.environ['V026_CKPT'] = base
        print('model', arm, json.dumps({k: info[k] for k in ('checkpoint', 'input_depth', 'amp', 'device')}), flush=True)
        for seg, d in todo:
            ct = np.load(ct_path(seg))
            assert ct.shape == (bd.NL,) + tuple(SHAPES[seg][1:]), ct.shape
            im = br.infer_map((inf, args, cm, dev), ct, DIRS[d], tmp, f'{arm}_{d}')
            im[~(ct > 0).any(0)] = 0
            png = map_path(arm, seg, d)
            os.makedirs(os.path.dirname(png), exist_ok=True)
            cv2.imwrite(png + '.tmp.png', im)
            os.replace(png + '.tmp.png', png)
            print(f'{arm} {seg} {d} {time.time() - t0:.0f}s', flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print('READ done', f'{time.time() - t0:.0f}s', flush=True)


def compose():
    import cv2
    os.makedirs(os.path.join(OUT, 'panels'), exist_ok=True)
    for seg in SEGMENTS:
        ct = np.load(ct_path(seg))[bd.NL // 2]
        for d in DIRS:
            maps = {a: cv2.imread(map_path(a, seg, d), cv2.IMREAD_UNCHANGED) for a in ('S0', 'REAL', 'R1')}
            maps['MEAN'] = ((maps['REAL'].astype(np.uint16) + maps['R1']) // 2).astype(np.uint8)
            cols = [('CT centre layer', ct)] + [(n, maps[k]) for n, k in
                                                (('stock ink_9um', 'S0'), ('B-001 REAL', 'REAL'), ('B-001 R1', 'R1'),
                                                 ('mean REAL+R1', 'MEAN'))]
            tiles = []
            for name, im in cols:
                im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
                im = cv2.copyMakeBorder(im, 40, 0, 0, 8, cv2.BORDER_CONSTANT, value=255)
                cv2.putText(im, name, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
                tiles.append(im)
            cv2.imwrite(os.path.join(OUT, 'panels', f'{seg}_{d}.png'), np.hstack(tiles))
            cv2.imwrite(os.path.join(OUT, 'maps', f'MEAN_{seg}_{d}.png'), maps['MEAN'])
        print('PANEL', seg, flush=True)
    print('COMPOSE done', flush=True)


SHUF_TEXTURE, SHUF_DEPTH = 0.5, 0.2


def control(ckpt_root):
    """README "Shuffled-layer control": does the B-001 map survive destroying the depth order?"""
    import cv2
    import v026_read as vr
    base = os.environ['V026_CKPT']
    f, sha = CKPTS['REAL']
    ck = os.path.join(ckpt_root, f)
    assert vr.sha256(ck) == sha, 'REAL'
    todo = [s for s in SEGMENTS if not os.path.exists(map_path('REAL_shuf', s, 'fwd'))]
    if todo:
        os.environ['V026_CKPT'] = ck
        try:
            inf, args, cm, dev, info = vr.load_model()
        finally:
            os.environ['V026_CKPT'] = base
        tmp = os.path.join(OUT, 'tmp')
        os.makedirs(tmp, exist_ok=True)
        for seg in todo:
            ct = np.load(ct_path(seg))
            order, _ = vr.shuffled_layers(ct.shape[0], inf, cm.input_depth)
            im = br.infer_map((inf, args, cm, dev), ct[order], 'forward', tmp, 'REAL_shuf')
            im[~(ct > 0).any(0)] = 0
            png = map_path('REAL_shuf', seg, 'fwd')
            os.makedirs(os.path.dirname(png), exist_ok=True)
            cv2.imwrite(png, im)
            print('REAL_shuf', seg, flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    print('CONTROL done', flush=True)


def corr(a, b, m):
    a, b = a[m].astype(np.float64), b[m].astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def compare(b001_read_dir):
    """Pearson corr of REAL fwd with REAL_shuf fwd: on PHerc0800 (map > 0) and on B-001's 49 labelled test windows
    (whole window), where the depth-ordered read is known to carry ink (B-001 control AUC 0.56 vs REAL 0.80)."""
    import cv2
    gp = {s: corr(cv2.imread(map_path('REAL', s, 'fwd'), 0), cv2.imread(map_path('REAL_shuf', s, 'fwd'), 0),
                  cv2.imread(map_path('REAL', s, 'fwd'), 0) > 0) for s in SEGMENTS}
    W = json.load(open(os.path.join(HERE, '..', '..', 'results', 'b001', 'windows.json')))['windows']
    ref = []
    for wid in W:
        a = cv2.imread(os.path.join(b001_read_dir, 'REAL', f'{wid}_fwd.png'), 0)
        b = cv2.imread(os.path.join(b001_read_dir, 'REAL_shuf', f'{wid}_fwd.png'), 0)
        ref.append(corr(a, b, np.ones(a.shape, bool)))
    g = float(np.mean(list(gp.values())))
    lo, hi = float(np.percentile(ref, 10)), float(np.percentile(ref, 90))
    out = 'TEXTURE_DRIVEN' if g >= SHUF_TEXTURE else 'DEPTH_DEPENDENT' if g <= SHUF_DEPTH else 'MIXED'
    for s, v in gp.items():
        print(f'COMPARE {s} corr(REAL, REAL_shuf) {v:.3f}', flush=True)
    print(f'COMPARE PHerc0800 mean {g:.3f}; B-001 test windows mean {np.mean(ref):.3f} (p10 {lo:.3f}, p90 {hi:.3f}, n {len(ref)})',
          flush=True)
    print(f'COMPARE_OUTCOME {out} (texture if >= {SHUF_TEXTURE}, depth-dependent if <= {SHUF_DEPTH}); '
          f'{"inside" if lo <= g <= hi else "outside"} the test windows p10-p90 range', flush=True)


def main():
    a = sys.argv[1:]
    if a and a[0] == 'check' and len(a) <= 2:
        check(a[1] if len(a) == 2 else None)
    elif a == ['fetch']:
        fetch()
    elif len(a) == 2 and a[0] == 'read':
        read(a[1])
    elif a == ['compose']:
        compose()
    elif len(a) == 2 and a[0] == 'control':
        control(a[1])
    elif len(a) == 2 and a[0] == 'compare':
        compare(a[1])
    elif a == ['clean']:
        shutil.rmtree(os.path.join(OUT, 'ct'), ignore_errors=True)
        print('CLEAN done', flush=True)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
