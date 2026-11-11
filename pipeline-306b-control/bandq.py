"""Own band-quality score from the public m7 prediction: per z bin, structure-tensor coherence and crest regularity."""
import sys, json, numpy as np
from scipy import ndimage as nd
from scipy.signal import find_peaks
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fl024")); import make_normals as mn

def axis_at(cps, z):
    zs = np.array([c['z'] for c in cps], float); o = np.argsort(zs); zs = zs[o]
    ys = np.array([cps[i]['y'] for i in o], float); xs = np.array([cps[i]['x'] for i in o], float)
    if len(zs) == 1: return ys[0], xs[0]
    f = lambda v: np.interp(z, zs, v) if zs[0] <= z <= zs[-1] else (v[0] + (v[1]-v[0])*(z-zs[0])/(zs[1]-zs[0]) if z < zs[0] else v[-1] + (v[-1]-v[-2])*(z-zs[-1])/(zs[-1]-zs[-2]))
    return f(ys), f(xs)

def score_box(m):
    m = m.astype(np.float32) / 255.0
    s = nd.gaussian_filter(m, 1.0)
    g = np.gradient(s)
    J = np.empty(m.shape + (3, 3), np.float32)
    for a in range(3):
        for b in range(a, 3):
            J[..., a, b] = J[..., b, a] = nd.gaussian_filter(g[a] * g[b], 3.0)
    w, v = np.linalg.eigh(J)
    coh = (w[..., 2] - w[..., 1]) / (w[..., 2] + w[..., 1] + 1e-9)
    sheet = s > 0.3
    if sheet.mean() < 0.01: return None
    # crest regularity along normals at random sheet points
    n = v[..., :, 2]; rng = np.random.default_rng(0); t = np.arange(-24, 24.25, 0.5); cvs = []
    pts = np.argwhere(sheet[8:-8, 16:-16, 16:-16]) + [8, 16, 16]
    for p in pts[rng.choice(len(pts), min(400, len(pts)), replace=False)]:
        q = p[:, None] + n[tuple(p)][:, None] * t[None]
        pk, _ = find_peaks(nd.map_coordinates(s, q, order=1), height=0.2, distance=4)
        if len(pk) >= 4:
            d = np.diff(t[pk]); cvs.append(d.std() / d.mean())
    return dict(coh=float(np.median(coh[sheet])), frac=float(sheet.mean()), reg=float(np.median(cvs)) if cvs else None, n=len(cvs))

def main(m7, umb, z0, z1, step=768, rad=(400, 900), boxes=6):
    src = mn.Src(m7 + '/2'); cps = json.load(open(umb))['control_points']
    for zc in range(z0, z1, step):
        ay, ax = axis_at(cps, zc); out = []
        for i in range(boxes):
            r = rad[i % 2]; th = 2 * np.pi * i / boxes
            y, x = int((ay + r * np.sin(th)) / 4), int((ax + r * np.cos(th)) / 4); z = zc // 4
            b = src.box(z - 24, z + 24, y - 48, y + 48, x - 48, x + 48)
            sc = score_box(b)
            if sc: out.append(sc)
        if not out: print(zc, 'empty', flush=True); continue
        coh = np.median([o['coh'] for o in out]); fr = np.median([o['frac'] for o in out])
        reg = [o['reg'] for o in out if o['reg'] is not None]
        print(f"{zc}\tcoh {coh:.3f}\tfrac {fr:.3f}\tcrestCV {np.median(reg) if reg else float('nan'):.3f}\tboxes {len(out)}", flush=True)

if __name__ == '__main__':
    a = sys.argv[1:]; main(a[0], a[1], int(a[2]), int(a[3]))
