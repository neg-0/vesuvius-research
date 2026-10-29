#!/usr/bin/env python3
"""Per-layer histogram matching of one surface render to another render's intensity distribution.

  match_intensity.py SRC.zarr REF.zarr OUT.zarr     write SRC remapped, layer k to REF layer k's histogram
  match_intensity.py stats A.zarr [B.zarr ...]      print per-layer p5/p50/p95 over each render's papyrus px
  match_intensity.py selftest

Papyrus px are the pixels > 0 on every layer. Each layer of SRC gets a uint8 lookup table that
maps its CDF onto REF's CDF for the same layer index (same depth offset from the mesh). Px outside SRC's mask stay 0;
px inside are clamped to >= 1, so the mask, and therefore every scorer's canvas, is unchanged. Geometry is untouched.
"""
import os
import sys

import numpy as np


def arr(path, mode='r'):
    import zarr
    a = zarr.open(path, mode=mode)
    return a['0'] if hasattr(a, 'keys') else a


def mask(a):
    m = None
    for k in range(a.shape[0]):
        lay = np.asarray(a[k]) > 0
        m = lay if m is None else (m & lay)
    return m


def hists(a, m):
    return np.stack([np.bincount(np.asarray(a[k])[m], minlength=256)[:256] for k in range(a.shape[0])])


def lut(hs, hr):
    cs, cr = np.cumsum(hs) / max(hs.sum(), 1), np.cumsum(hr) / max(hr.sum(), 1)
    t = np.searchsorted(cr, cs, side='left').clip(0, 255).astype(np.uint8)
    return np.maximum(t, 1)


def create_like(src, out):
    import zarr
    a = arr(src)
    root = zarr.open(src, mode='r')
    kw = dict(shape=a.shape, chunks=a.chunks, dtype='uint8', fill_value=0)
    if hasattr(root, 'keys'):
        g = zarr.open_group(out, mode='w')
        f = getattr(g, 'create_array', None) or g.create_dataset
        return f('0', **kw)
    return zarr.open_array(out, mode='w', **kw)


def match(src, ref, out):
    a, r = arr(src), arr(ref)
    if a.shape[0] != r.shape[0]:
        sys.exit(f'layer counts differ: {a.shape[0]} vs {r.shape[0]}')
    ms, mr = mask(a), mask(r)
    hs, hr = hists(a, ms), hists(r, mr)
    o = create_like(src, out)
    for k in range(a.shape[0]):
        x = np.asarray(a[k])
        o[k] = np.where(ms, lut(hs[k], hr[k])[x], 0).astype(np.uint8)
    return ms, hs, hr


def pct(h, q):
    c = np.cumsum(h) / max(h.sum(), 1)
    return int(np.searchsorted(c, q))


def stats(paths):
    for p in paths:
        a = arr(p)
        h = hists(a, mask(a))
        print(os.path.basename(p.rstrip('/')), 'layer p5 p50 p95:',
              ' '.join(f'{k}:{pct(h[k], .05)}/{pct(h[k], .5)}/{pct(h[k], .95)}' for k in range(a.shape[0])))


def selftest():
    import tempfile
    import zarr
    rng = np.random.default_rng(0)
    d = tempfile.mkdtemp()
    L, H, W = 4, 64, 80
    s = rng.normal(90, 10, (L, H, W)).clip(1, 255).astype(np.uint8)
    r = rng.normal(150, 25, (L, H, W)).clip(1, 255).astype(np.uint8)
    s[:, :8] = 0
    s[2, 10:12, :5] = 0
    for name, x in (('s', s), ('r', r)):
        g = zarr.open_group(os.path.join(d, name + '.zarr'), mode='w')
        f = getattr(g, 'create_array', None) or g.create_dataset
        f('0', shape=x.shape, chunks=(1, 32, 32), dtype='uint8', fill_value=0)[:] = x
    match(os.path.join(d, 's.zarr'), os.path.join(d, 'r.zarr'), os.path.join(d, 'o.zarr'))
    o = np.asarray(arr(os.path.join(d, 'o.zarr'))[:])
    m = (s > 0).all(0)
    assert ((o > 0).all(0) == m).all(), 'mask changed'
    assert (o[:, ~m] == 0).all()
    for k in range(L):
        assert abs(float(np.median(o[k][m])) - float(np.median(r[k]))) <= 2, 'median not matched'
        assert abs(float(o[k][m].std()) - float(r[k].std())) <= 2, 'spread not matched'
        x = s[k][m].astype(int)
        y = o[k][m].astype(int)
        assert (np.diff(y[np.argsort(x, kind='stable')]) >= 0).all(), 'not monotone'
    print('selftest OK')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['selftest']:
        selftest()
    elif a[:1] == ['stats']:
        stats(a[1:])
    elif len(a) == 3:
        match(*a)
    else:
        sys.exit(__doc__)
