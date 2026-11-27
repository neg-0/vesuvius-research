"""Letter-likeness of an ink map: are there separate letter-sized forms standing in a row on one face?

This turns the by-eye rule used to judge follow-up maps into numbers. A LEAD is at least 4 letter-sized forms
(2 to 5 mm) on one baseline, on one face, with the other face under a third of it, and not haze, blobs, bars or
bands. Steps:
  1. confident ink (> thr) on the ink face; strokes closer than `join_mm` are grouped into one form
  2. a form is letter-like if its height or width is 2-5 mm (`lo_mm`, `hi_mm`), it is not a bar (long side / short side
     <= `max_elong`), not a filled blob (ink fills <= `max_fill` of its bounding box), and drawn in strokes no wider
     than `max_stroke_mm` (haze patches and glows are wide)
  3. baseline: the largest run of letter-like forms whose centres lie on one straight line (within `line_tol` of the
     median form height), neighbours no more than `max_gap` form sizes apart (centre to centre)
  4. haze: the share of all confident ink that belongs to letter-like forms (`form_share`)
  5. one-sidedness: confident share on the back face / ink face
`letters` is True when the row holds >= `min_row` forms, form_share >= `min_form_share` and back/ink < `max_back`.
`score` = forms in the row x form_share x (1 - back/ink), clipped at 0; it ranks maps, `letters` is the pass rule.
Strict by design: letters blurred together into one word-sized form are not counted, so a fail does not rule text out.

usage: letterscore.py <prob_ink.npy> <prob_back.npy> [valid.npy] --um U       (maps from followup.py or reader.run_stack)
       letterscore.py --dirs <dir>... [--um U]   each dir holds prob_v3_fwd.npy, prob_v3_rev.npy, valid.npy, followup.json;
                                                 the ink face is taken from followup.json ("face"), else the brighter one;
                                                 without --um the pixel is voxel_um x --block (default 4)
"""
import argparse, json, os, sys
import numpy as np
from scipy import ndimage as ndi

DEFAULTS = dict(thr=0.62, join_mm=0.35, lo_mm=2.0, hi_mm=5.0, max_elong=4.0, max_fill=0.6, max_stroke_mm=0.8, line_tol=0.35, max_gap=2.0,
                min_row=4, min_form_share=0.35, max_back=1 / 3)


def forms(ink, valid, um, thr=0.62, join_mm=0.35, lo_mm=2.0, hi_mm=5.0, max_elong=4.0, max_fill=0.6, max_stroke_mm=0.8, **_):
    """Connected forms of confident ink after joining strokes closer than join_mm. Returns a list of dicts
    (centre in mm, height/width in mm, fill, letter-like flag, confident pixel count)."""
    A = (ink > thr) & valid
    r = max(1, int(round(join_mm * 1000 / um / 2)))
    J = ndi.binary_dilation(A, structure=np.ones((3, 3), bool), iterations=r)
    lab, n = ndi.label(J, structure=np.ones((3, 3), bool))
    E = ndi.distance_transform_edt(A)                                     # half the local stroke width, in px
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = (lab[sl] == i) & A[sl]; k = int(m.sum())
        if k == 0: continue
        ys, xs = np.nonzero(m); h = (ys.max() - ys.min() + 1) * um / 1000; w = (xs.max() - xs.min() + 1) * um / 1000
        fill = k / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        size = max(h, w); elong = max(h, w) / max(min(h, w), um / 1000)
        stroke = 2 * float(np.percentile(E[sl][m], 90)) * um / 1000            # stroke width (90th pct), mm
        out.append(dict(y=(sl[0].start + ys.mean()) * um / 1000, x=(sl[1].start + xs.mean()) * um / 1000, h=h, w=w,
                        fill=fill, stroke=stroke, n=k,
                        letter=bool(lo_mm <= size <= hi_mm and elong <= max_elong and fill <= max_fill and stroke <= max_stroke_mm)))
    return out


def best_row(F, line_tol=0.35, max_gap=2.0, **_):
    """Largest set of letter-like forms on one straight line with no gap above max_gap form heights. Returns (count,
    indices)."""
    L = sorted([f for f in F if f["letter"]], key=lambda f: -f["n"])[:300]      # cap the pair search on busy maps
    if len(L) < 2: return len(L), list(range(len(L)))
    P = np.array([[f["x"], f["y"]] for f in L]); H = float(np.median([max(f["h"], f["w"]) for f in L]))
    best = (1, [0])
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            d = P[j] - P[i]; nd = np.linalg.norm(d)
            if nd == 0: continue
            u = d / nd; nrm = np.array([-u[1], u[0]])
            on = np.nonzero(np.abs((P - P[i]) @ nrm) <= line_tol * H)[0]
            t = np.sort((P[on] - P[i]) @ u); idx = on[np.argsort((P[on] - P[i]) @ u)]
            # longest run with gaps <= max_gap * H
            s = 0
            for k in range(1, len(t) + 1):
                if k == len(t) or t[k] - t[k - 1] > max_gap * H:
                    if k - s > best[0]: best = (k - s, [int(q) for q in idx[s:k]])
                    s = k
    return best


def score_maps(ink, back, valid=None, um=7.91, **kw):
    """Letter-likeness of one face (ink) against the other (back). Returns a dict; 'letters' is the pass rule."""
    p = dict(DEFAULTS); p.update(kw)
    ink = np.asarray(ink, np.float32); back = np.asarray(back, np.float32)
    valid = np.ones(ink.shape, bool) if valid is None else np.asarray(valid, bool)
    nv = max(1, int(valid.sum()))
    s_ink = float(((ink > p["thr"]) & valid).sum() / nv); s_back = float(((back > p["thr"]) & valid).sum() / nv)
    F = forms(ink, valid, um, **p); row, idx = best_row(F, **p)
    conf = sum(f["n"] for f in F); share = sum(f["n"] for f in F if f["letter"]) / conf if conf else 0.0
    ratio = s_back / s_ink if s_ink > 0 else 1.0
    letters = bool(row >= p["min_row"] and share >= p["min_form_share"] and ratio < p["max_back"])
    return dict(letters=letters, score=round(max(0.0, row * share * (1 - ratio)), 4), row=int(row),
                letter_forms=int(sum(f["letter"] for f in F)), forms=len(F), form_share=round(share, 4),
                ink_share=round(s_ink, 5), back_share=round(s_back, 5), back_ratio=round(ratio, 4))


def score_dir(d, um=None, block=4, **kw):
    """A followup.py output folder: picks the ink face from followup.json (or the brighter V3 face). With um=None the
    pixel size is followup.json's voxel_um x block (follow-up maps are saved as block x block means)."""
    fw = np.load(os.path.join(d, "prob_v3_fwd.npy")); rv = np.load(os.path.join(d, "prob_v3_rev.npy"))
    v = np.load(os.path.join(d, "valid.npy")) if os.path.exists(os.path.join(d, "valid.npy")) else None
    j = json.load(open(os.path.join(d, "followup.json"))) if os.path.exists(os.path.join(d, "followup.json")) else {}
    face = j.get("face")
    if um is None:
        if "voxel_um" not in j: raise ValueError(f"{d}: no voxel_um in followup.json, give --um")
        um = j["voxel_um"] * block
    if face not in ("fwd", "rev"): face = "fwd" if (fw > 0.62).mean() >= (rv > 0.62).mean() else "rev"
    ink, back = (fw, rv) if face == "fwd" else (rv, fw)
    r = score_maps(ink, back, v, um, **kw); r["face"] = face; return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="*"); ap.add_argument("--dirs", nargs="+"); ap.add_argument("--um", type=float)
    ap.add_argument("--block", type=int, default=4, help="with --dirs and no --um: map pixel = voxel_um x block")
    for k, v in DEFAULTS.items(): ap.add_argument("--" + k.replace("_", "-"), dest=k, type=float, default=v)
    a = ap.parse_args(argv); kw = {k: getattr(a, k) for k in DEFAULTS}; kw["min_row"] = int(kw["min_row"])
    if a.dirs:
        rows = [(d, score_dir(d, a.um, a.block, **kw)) for d in a.dirs]
        rows.sort(key=lambda t: -t[1]["score"])
        cols = ["letters", "score", "row", "letter_forms", "forms", "form_share", "ink_share", "back_ratio", "face"]
        print("dir\t" + "\t".join(cols))
        for d, r in rows: print(os.path.basename(os.path.normpath(d)) + "\t" + "\t".join(str(r[c]) for c in cols))
        return
    if len(a.maps) < 2 or a.um is None: sys.exit("give <prob_ink.npy> <prob_back.npy> [valid.npy] --um U, or --dirs")
    v = np.load(a.maps[2]) if len(a.maps) > 2 else None
    print(json.dumps(score_maps(np.load(a.maps[0]), np.load(a.maps[1]), v, a.um, **kw)))


if __name__ == "__main__":
    main()
