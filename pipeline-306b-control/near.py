#!/usr/bin/env python3
"""Winding choice against a known-text mesh: near.py SNAP_DIR WINNERS_TIFXYZ OUT_TSV [rows r0:r1] [cols c0:c1].

Claim points = the winners' tifxyz grid nodes in rows/cols (default the claim box, rows 82:138, cols 122:342), at most
20,000. Every snapped winding with frac_snapped >= 0.8 (SNAP_DIR/snap_report.tsv) is densified 4x by bilinear
interpolation of its 20-vx grid (cells with 4 valid corners), then a cKDTree on the winding gives each claim point's
distance to it. near.tsv: winding, median, p90, fraction within 10 vx (sorted by median).
Prints CHOSEN (among the 15 nearest by median: median <= 30 vx or >= 10% of claim points within 10 vx; plus two either side
of the nearest) and the gate:
MESH_REACHES if the nearest median < 10 vx, else MESH_MISS."""
import csv, os, sys
import numpy as np, tifffile
from scipy.spatial import cKDTree

def grid(d):
    return np.stack([tifffile.imread(os.path.join(d, c + '.tif')).astype(np.float64) for c in 'xyz'], -1)

def dense(g, k=4):
    ok = (g > 0).all(-1); c = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    i, j = np.nonzero(c); out = []
    for a in np.arange(k) / k:
        for b in np.arange(k) / k:
            out.append((1-a)*(1-b)*g[i, j] + a*(1-b)*g[i+1, j] + (1-a)*b*g[i, j+1] + a*b*g[i+1, j+1])
    return np.concatenate(out + [g[ok]])

def main(a):
    snap, win, out = a[:3]
    r = slice(*map(int, (a[a.index('rows')+1] if 'rows' in a else '82:138').split(':')))
    c = slice(*map(int, (a[a.index('cols')+1] if 'cols' in a else '122:342').split(':')))
    g = grid(win)[r, c]; P = g[(g > 0).all(-1)]
    if len(P) > 20000: P = P[np.linspace(0, len(P) - 1, 20000).astype(int)]
    lo, hi = P.min(0) - 60, P.max(0) + 60
    rows = sorted(csv.DictReader(open(os.path.join(snap, 'snap_report.tsv')), delimiter='\t'), key=lambda x: int(x['winding'][1:]))
    ok = [x['winding'] for x in rows if float(x['frac_snapped']) >= 0.8]
    res = []; U = np.full(len(P), 200.0)
    for w in ok:
        Q = dense(grid(os.path.join(snap, w))); Q = Q[((Q >= lo) & (Q <= hi)).all(1)]
        if len(Q) == 0: continue
        d = cKDTree(Q).query(P, distance_upper_bound=200)[0]; d = np.minimum(d, 200.0); U = np.minimum(U, d)  # beyond 200 vx counts as 200; U = union of windings
        res.append((w, float(np.median(d)), float(np.percentile(d, 90)), float((d < 10).mean()), float(d.min())))
    res.sort(key=lambda t: t[1])
    with open(out, 'w') as f:
        f.write('winding\tmedian_vx\tp90_vx\tfrac_within_10\tmin_vx\n')
        for t in res: f.write('%s\t%.2f\t%.2f\t%.4f\t%.2f\n' % t)
    if not res: print('CHOSEN'); print('MESH_MISS: no snapped winding enters the claim box'); return
    best = res[0][0]; i = ok.index(best)
    ch = {t[0] for t in res[:15] if t[1] <= 30 or t[3] >= 0.10} | set(ok[max(0, i-2):i+3])  # median <= 30 vx or >= 10% of claim points within 10 vx; at most the 15 nearest, plus +-2
    print('CHOSEN ' + ' '.join(sorted(ch, key=lambda w: int(w[1:]))))
    print('UNION all scored windings: median %.2f vx p90 %.2f, frac within 10 vx %.4f' % (np.median(U), np.percentile(U, 90), (U < 10).mean()))
    print(('MESH_REACHES' if res[0][1] < 10 else 'MESH_MISS') + ': nearest %s median %.2f vx p90 %.2f, %d points, %d windings scored' % (best, res[0][1], res[0][2], len(P), len(res)))

def selftest():
    import tempfile
    t = tempfile.mkdtemp(); yy, xx = np.mgrid[0:30, 0:40] * 20.0
    def put(d, z):
        os.makedirs(d, exist_ok=True)
        for n, v in zip('xyz', (xx + 100, yy + 100, np.full_like(xx, z))): tifffile.imwrite(os.path.join(d, n + '.tif'), v.astype(np.float32))
    put(t + '/win', 500.0); s = t + '/snap'
    with open(t + '/snap_report.tsv', 'w') as f: f.write('winding\tfrac_snapped\n')
    os.makedirs(s); open(s + '/snap_report.tsv', 'w').write('winding\tfrac_snapped\n' + ''.join('w%03d\t%s\n' % (k, '0.9' if k != 3 else '0.5') for k in range(1, 9)))
    for k in range(1, 9): put(s + '/w%03d' % k, 500.0 + 25 * (k - 5) + 3)
    import io, contextlib; b = io.StringIO()
    with contextlib.redirect_stdout(b): main([s, t + '/win', t + '/near.tsv', 'rows', '0:30', 'cols', '0:40'])
    o = b.getvalue(); print(o)
    assert 'MESH_REACHES: nearest w005 median 3.00' in o, o
    assert o.split('\n')[0] == 'CHOSEN w002 w004 w005 w006 w007', o   # w003 excluded (frac 0.5); w004/w006 median 22 vx; +-2
    put(s + '/w005', 540.0)
    with contextlib.redirect_stdout(io.StringIO()) as b2: main([s, t + '/win', t + '/near.tsv', 'rows', '0:30', 'cols', '0:40'])
    assert 'MESH_MISS' in b2.getvalue(), b2.getvalue(); print('SELFTEST_OK')

if __name__ == '__main__':
    selftest() if sys.argv[1:] == ['selftest'] else main(sys.argv[1:])
