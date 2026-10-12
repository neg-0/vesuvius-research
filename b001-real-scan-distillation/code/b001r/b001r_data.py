"""B-001r data: simulated 9 um inputs for the B-001 training tiles (arm SIM), i.e. the high-res (2.2-2.4 um) surface
volume of each tile pooled onto its low-res grid through the B-001 CT-only correction.

usage: b001r_data.py fetch_sim        (pod: CPU + network; $B001_DATA/tiles_sim/<tile>.npy, uint8 28 x 260 x 260;
                                       resumable; B001_WORKERS parallel tiles, default 8)
       b001r_data.py --check          (selftest of the coordinate maths, then SIM vs real LR NCC on 12 tiles)

Pooling (fixed in experiments/b001r/README.md): HR level 2 (4x averaged in-plane, ~9.6 / 8.9 um) is read over the
tile's corrected footprint; its layers are averaged into the 28 LR layer bins by physical depth (b001_data.pooled_hr's
binning, centred on the HR surface, so no dz is applied); each LR px p is resampled bilinearly at the corrected HR
position k (p + 0.5 - s(p)) - 0.5 (level-0 px; s = the PLACE_OK affine shift). That is the same position b001_data.sample
uses for the teacher labels, so SIM tiles carry the same labels as the real tiles. Rounded to uint8.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'b001'))
import b001_data as bd                                              # noqa: E402

WORKERS = int(os.environ.get('B001_WORKERS', '8'))
CHECK_TILES, CHECK_NCC = 12, 0.6


def hr_pos(pl, yy, xx):
    """Float HR level-0 px of LR px (yy, xx) after removing the fitted shift (b001_data.hr_index without rounding)."""
    k = pl['k']
    c = np.array(pl['fit']['coef'])
    dy = c[0, 0] + c[1, 0] * yy + c[2, 0] * xx
    dx = c[0, 1] + c[1, 1] * yy + c[2, 1] * xx
    return (yy - dy + 0.5) * k - 0.5, (xx - dx + 0.5) * k - 0.5


def pool_layers(s, hr):
    """(nh, ...) HR stack -> (nl, ...) by physical depth bins of the LR layers (b001_data.pooled_hr's binning)."""
    um = bd.SCROLLS[s['scroll']]
    nh, nl = s['sv_hr_shape'][0], s['sv_lr_shape'][0]
    assert hr.shape[0] == nh, (hr.shape, nh)
    zh = (np.arange(nh) - (nh - 1) / 2) * um['um_hr']
    out = np.zeros((nl,) + hr.shape[1:], np.float32)
    for j in range(nl):
        c = (j - (nl - 1) / 2) * um['um_lr']
        sel = (zh >= c - um['um_lr'] / 2) & (zh < c + um['um_lr'] / 2)
        if sel.any():
            out[j] = hr[sel].mean(0)
    return out


def sim_crop(s, pl, y0, x0, h, w):
    import cv2
    yy, xx = np.meshgrid(np.arange(y0, y0 + h, dtype=np.float64), np.arange(x0, x0 + w, dtype=np.float64), indexing='ij')
    hy, hx = hr_pos(pl, yy, xx)
    ly, lx = (hy - 1.5) / 4.0, (hx - 1.5) / 4.0                    # level-2 px (2x2 means twice: centre offset 1.5)
    by0, bx0 = int(np.floor(ly.min())) - 2, int(np.floor(lx.min())) - 2
    bh, bw = int(np.ceil(ly.max())) + 3 - by0, int(np.ceil(lx.max())) + 3 - bx0
    hr = bd.zcrop(s['sv_hr'], 2, by0, bx0, bh, bw, workers=4).astype(np.float32)
    pooled = pool_layers(s, hr)
    my, mx = (ly - by0).astype(np.float32), (lx - bx0).astype(np.float32)
    st = np.stack([cv2.remap(p, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for p in pooled])
    st = np.clip(np.rint(st), 0, 255).astype(np.uint8)
    off = (st.shape[0] - bd.NL) // 2
    return st[off:off + bd.NL]


def fetch_one(name, t, segs, places):
    out = os.path.join(bd.DATA, 'tiles_sim', name + '.npy')
    if os.path.exists(out):
        return False
    y0, x0 = t['origin']
    h, w = t['px']
    st = sim_crop(segs[t['segment']], places[t['segment']], y0, x0, h, w)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + '.part.npy', st)
    os.replace(out + '.part.npy', out)
    return True


def load_all():
    reg = bd.jload(os.path.join(bd.RESULTS, 'tiles.json'))
    segs = bd.segs()['segments']
    places = {n: bd.jload(os.path.join(bd.RESULTS, 'place', n + '.json')) for n in {t['segment'] for t in reg['tiles'].values()}}
    assert all(p and p['verdict'] == 'PLACE_OK' for p in places.values())
    return reg['tiles'], segs, places


def fetch_sim():
    tiles, segs, places = load_all()
    t0, done = time.time(), [0]

    def job(item):
        n, t = item
        new = fetch_one(n, t, segs, places)
        done[0] += 1
        if done[0] % 50 == 0:
            print(f'SIM {done[0]}/{len(tiles)} {time.time() - t0:.0f}s', flush=True)
        return new

    with ThreadPoolExecutor(WORKERS) as ex:
        n_new = sum(ex.map(job, sorted(tiles.items())))
    have = sum(os.path.exists(os.path.join(bd.DATA, 'tiles_sim', n + '.npy')) for n in tiles)
    print(f'FETCH_SIM done: {have} of {len(tiles)} tiles present ({n_new} new) {time.time() - t0:.0f}s', flush=True)
    if have != len(tiles):
        raise SystemExit('FETCH_SIM incomplete: rerun')


def check():
    # coordinate maths: hr_pos rounds to b001_data.hr_index
    pl = {'k': 3.9, 'fit': {'coef': [[-6.0, 7.5], [0.002, 0.0003], [-0.0003, 0.0024]]}}
    yy, xx = np.meshgrid(np.arange(100, 140.0), np.arange(900, 950.0), indexing='ij')
    hy, hx = hr_pos(pl, yy, xx)
    iy, ix = bd.hr_index(pl, yy, xx, True)
    assert (np.rint(hy).astype(np.int64) == iy).all() and (np.rint(hx).astype(np.int64) == ix).all()
    # SIM vs real LR on the first tiles of each training scroll (CT only, no labels, no model)
    tiles, segs, places = load_all()
    by = {}
    for n, t in sorted(tiles.items()):
        by.setdefault(t['scroll'], []).append((n, t))
    pick = [x for sc in sorted(by) for x in by[sc][:CHECK_TILES // len(by)]]
    vals = []
    for n, t in pick:
        s, pl = segs[t['segment']], places[t['segment']]
        y0, x0 = t['origin']
        h, w = t['px']
        sim = sim_crop(s, pl, y0, x0, h, w).astype(np.float32)
        lr = bd.centre28(bd.zcrop(s['sv_lr'], 0, y0, x0, h, w))
        b = sim[bd.NCC_LAYERS[0]:bd.NCC_LAYERS[1]]
        res = []
        for e in range(-bd.DZ, bd.DZ + 1):                         # real LR depth offset varies inside a segment
            a = bd.apply_dz(lr, t['dz'] + e)[bd.NCC_LAYERS[0]:bd.NCC_LAYERS[1]].astype(np.float32)
            res.append((bd.ncc(a, b, (a > 0) & (b > 0)), e))
        v, e = max(res)
        vals.append(v)
        print(f'CHECK sim {n} {t["scroll"]} NCC(sim, real LR) {v:.3f} at extra dz {e:+d} (segment dz {t["dz"]})', flush=True)
    med = float(np.median(vals))
    ok = med >= CHECK_NCC
    print(f'CHECK {"OK" if ok else "FAIL"}: median NCC {med:.3f} over {len(vals)} tiles (need >= {CHECK_NCC})', flush=True)
    return ok


def main():
    a = sys.argv[1:]
    if a == ['--check']:
        sys.exit(0 if check() else 1)
    elif a == ['fetch_sim']:
        fetch_sim()
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
