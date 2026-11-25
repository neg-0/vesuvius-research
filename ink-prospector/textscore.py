"""Letter-scale text score from the two saved side maps (ds4 PNGs) of a patch.

Counts only confident ink (score above 0.62, PNG value 190) in structures 0.3 to 6 mm across that sit away from the
patch edge, and takes the difference between the two sides (ink is one-sided). Faint mottling, large glows, edge flare
and two-sided brightness score near zero. Checked 2026-11-15: ranks the 306B prize patch 1st of 177 (the earlier
letter-size blob count ranked it 10th, because touching letters merge into one large blob) and tracks a finer-scan answer key better on one
scroll (Spearman 0.87 against 0.58) and the same on another (0.73).

usage: textscore.py <run dir with shard-*> <um> [N] [contact.png]   prints the top N patches (default 20) of a sharded run
(prospect.py rank does the same for any scan folder).
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
def side(img, um_px, edge_mm=0.35, lo_mm=0.3, hi_mm=6.0, thr=190):
    valid = ndi.binary_erosion(ndi.binary_closing(img > 0, iterations=2), iterations=max(1, int(edge_mm * 1000 / um_px)))
    lab, n = ndi.label((img > thr) & valid)
    if n == 0 or valid.sum() == 0: return 0.0, valid
    keep = 0
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        h = (sl[0].stop - sl[0].start) * um_px / 1000; w = (sl[1].stop - sl[1].start) * um_px / 1000
        if lo_mm <= max(h, w) <= hi_mm: keep += (lab[sl] == i).sum()
    return keep / valid.sum(), valid
def score(png_a, png_b, um=8.64, ds=4):
    a = np.asarray(Image.open(png_a)); b = np.asarray(Image.open(png_b))
    ta, _ = side(a, um * ds); tb, _ = side(b, um * ds)
    return ta - tb, ta, tb
def lineness(img, um_px):
    """Experimental "is this writing" measure for maps of 18 mm or more (a spread or a stitched sheet), img 0 = no data.
    Confident ink is summed along straight bands at every angle; the result is how much stronger the banding is at
    the best angle than at the median angle, with that angle and the band spacing. Lines of text give banding at one
    angle. First numbers (2026-11-15, few samples, see the README Limits): 306B maps with known or suspected text
    above 2, back faces and stroke-only sheets of other scrolls 1.1 to 1.9. Second check failed: 1.3 to 1.9 on both faces at three places on the awarded
    PHerc0343 line (ragged sheet, one sparse line). Not reliable on
    stitched mosaics (seams and edge glow make bands of their own), so only spread reports it."""
    m = ndi.binary_erosion(img > 0, iterations=max(1, int(280 / um_px)))
    if m.sum() * (um_px / 1e4) ** 2 < 0.25: return None                      # under 0.25 cm2
    x = ndi.gaussian_filter(np.where(m, np.clip(img.astype(np.float32) - 127, 0, None), 0), 100 / um_px); s = []
    for ang in np.arange(0, 180, 4):
        r = ndi.rotate(x, ang, reshape=True, order=1); n = ndi.rotate(m.astype(np.float32), ang, reshape=True, order=1).sum(1)
        ok = n > 0.5 * n.max(); p = r.sum(1)[ok] / n[ok]; p = p - ndi.uniform_filter1d(p, max(3, int(9000 / um_px))); s.append((p.std(), ang, p))
    v = np.array([q[0] for q in s]); k = int(v.argmax()); p = s[k][2]; ac = np.correlate(p, p, "full")[len(p) - 1:]; ac = ac / max(ac[0], 1e-9)
    lo, hi = int(2500 / um_px), int(9000 / um_px); lag = lo + int(ac[lo:hi].argmax()) if len(ac) > hi else 0
    return dict(ratio=round(float(v[k] / max(np.median(v), 1e-9)), 2), angle=int(s[k][1]), spacing_mm=round(lag * um_px / 1000, 1))
if __name__ == "__main__":
    import sys, glob, csv, os
    run, um = sys.argv[1], float(sys.argv[2]); rows = []
    for sh in sorted(glob.glob(run + "/shard-*")):
        seen = set()
        for r in csv.DictReader(open(sh + "/scores.tsv"), delimiter="\t"):
            p = f"{sh}/p{int(r['id']):04d}"
            if r["status"] != "ok" or r["id"] in seen or not os.path.exists(p + "_a.png") or r["frac_a"] == r["frac_b"]: continue
            seen.add(r["id"]); d, ta, tb = score(p + "_a.png", p + "_b.png", um)
            rows.append((abs(d), os.path.basename(sh), int(r["id"]), "a" if d > 0 else "b", ta, tb, r["seed_z"], r["seed_y"], r["seed_x"]))
    rows.sort(reverse=True)
    for i, r in enumerate(rows[:int(sys.argv[3]) if len(sys.argv) > 3 else 20]): print(i + 1, "%.4f %s p%d side %s  a %.4f b %.4f  seed %s %s %s" % r)
    if len(sys.argv) > 4:
        n = int(sys.argv[3]); tiles = [np.asarray(Image.open(f"{run}/{r[1]}/p{r[2]:04d}_{r[3]}.png").resize((256, 256))) for r in rows[:n]]
        c = 6; g = np.zeros(((n + c - 1) // c * 256, c * 256), np.uint8)
        for i, t in enumerate(tiles): g[i // c * 256:i // c * 256 + 256, i % c * 256:i % c * 256 + 256] = t
        Image.fromarray(g).save(sys.argv[4])
