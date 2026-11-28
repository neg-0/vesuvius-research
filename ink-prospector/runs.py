"""Read the patches of one or more scan folders (a folder with scores.tsv, or a run folder holding shard-*/)."""
import csv, glob, json, os
import textscore


def shard(s):
    """'i/n' -> (i, n) with 0 <= i < n, or SystemExit naming the problem. Every command that takes --shard checks it
    with this before writing anything, so a typo cannot overwrite a run's files."""
    try: i, n = (int(v) for v in s.split("/"))
    except ValueError: raise SystemExit(f"--shard {s}: give i/n, e.g. 0/4")
    if not 0 <= i < n: raise SystemExit(f"--shard {s}: need 0 <= i < n")
    return i, n
def scan_dirs(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "shard-*"))) if glob.glob(os.path.join(p, "shard-*")) else [p]
    return [d for d in out if os.path.exists(os.path.join(d, "scores.tsv"))]
def not_runs(paths):
    """The paths with no scores.tsv in them or in a shard-*/ below, such as a mistyped run folder."""
    return [p for p in paths if not scan_dirs([p])]
def meta(d):
    for f in (os.path.join(d, "run.json"), os.path.join(os.path.dirname(d.rstrip("/")), "run.json")):
        if os.path.exists(f): return json.load(open(f))
    return {}
_TS = {}
def _score(base, u):
    """textscore of a patch's two faces, remembered while their files are unchanged: a long-lived caller (prospect.py
    serve on a run that is still being written) then rescores only the new or rewritten patches, not all of them."""
    fa, fb = base + "_a.png", base + "_b.png"; st_a, st_b = os.stat(fa), os.stat(fb)
    k = (fa, st_a.st_mtime_ns, st_a.st_size, st_b.st_mtime_ns, st_b.st_size, u)
    if k not in _TS:
        if len(_TS) > 200000: _TS.clear()
        _TS[k] = textscore.score(fa, fb, u)
    return _TS[k]
def patches(paths, um=None, min_area=0.0):
    """Usable patches of all folders, best text score first. Each: dict with id, dir, base path, scores.
    min_area (cm2) drops patches with less read surface, such as slivers at a sheet's edge, whose scores are noisy."""
    rows = []; dirs = scan_dirs(paths)
    multi = len({os.path.dirname(d.rstrip("/")) for d in dirs if os.path.basename(d.rstrip("/")).startswith("shard-")}) > 1
    for d in dirs:
        u = um or meta(d).get("um") or 8.64; seen = set(); tag = os.path.basename(d.rstrip("/")).replace("shard-", "s")
        if multi and tag != os.path.basename(d.rstrip("/")): tag = os.path.basename(os.path.dirname(d.rstrip("/"))) + "-" + tag   # several runs: keep ids apart
        src = {}                                                         # score_tifxyz folders: which tifxyz and grid window
        if os.path.exists(os.path.join(d, "surfaces.tsv")):
            for r in csv.DictReader(open(os.path.join(d, "surfaces.tsv")), delimiter="\t"):
                src[r["id"]] = f"{os.path.basename(r['tifxyz'].rstrip('/'))} rows {r['r0']}-{r['r1']} cols {r['c0']}-{r['c1']}"
        for r in csv.DictReader(open(os.path.join(d, "scores.tsv")), delimiter="\t"):
            base = os.path.join(d, f"p{int(r['id']):04d}")
            if r["status"] != "ok" or r["id"] in seen or not os.path.exists(base + "_a.png") or r["frac_a"] == r["frac_b"]: continue
            seen.add(r["id"]); s, ta, tb = _score(base, u)
            rows.append(dict(id=f"{tag}-{int(r['id'])}", dir=d, base=base, um=u, seed=[int(float(r[k])) for k in ("seed_z", "seed_y", "seed_x")],
                             tracked=float(r["tracked"]), area=float(r["area_cm2"]), fa=float(r["frac_a"]), fb=float(r["frac_b"]), ts=abs(s), side="a" if s > 0 else "b",
                             src=src.get(r["id"], "")))
    return sorted((r for r in rows if r["area"] >= min_area), key=lambda r: -r["ts"])
def letters(p, faces=None):
    """letterscore.py on a patch's saved maps: the ink face against the other. The PNGs are ds4 reads stored as
    (prob - 0.25) / 0.5 x 255, with 0 where the patch is not tracked, so a pixel is 4 voxels. faces names the two
    files (ink, back) when they are not a/b, as for stitched sheets (sheetK_ink.png, sheetK_back.png)."""
    import numpy as np, letterscore
    from PIL import Image
    fs = faces or (p["side"], "b" if p["side"] == "a" else "a")
    a, b = (np.asarray(Image.open(f"{p['base']}_{s}.png").convert("L"), np.float32) for s in fs)
    if a.shape != b.shape: return None
    return letterscore.score_maps(0.25 + a / 510, 0.25 + b / 510, (a > 0) | (b > 0), p["um"] * 4)
def status(paths):
    """Progress of score_tifxyz folders: per folder done/expected, missing ids and how many rows of each status
    (ok read, noct off the scan, small, ...), then per set of shards
    made from the same surfaces, the ids no folder has (a shard never run, or cut off). Folders without a
    job count in run.json (older runs, scan.py runs) report done only. `prospect.py scan` runs count their lattice seeds
    (from box and spacing in run.json, as scan.py lays them) and take the shard from the shard-i folder name."""
    rows, sets = [], {}
    for d in scan_dirs(paths):
        m = meta(d); n = m.get("jobs"); si, sn = (int(v) for v in str(m.get("shard") or "0/1").split("/"))
        if n is None and m.get("box") and m.get("spacing"):                  # a prospect.py scan run
            z0, z1, y0, y1, x0, x1 = m["box"]; S = int(m["spacing"])
            n = len(range(z0 + S // 2, z1, S)) * len(range(y0 + S // 2, y1, S)) * len(range(x0 + S // 2, x1, S))
            b = os.path.basename(d.rstrip("/")); sh = b.startswith("shard-") and b[6:].isdigit()
            sn = int(m.get("shards") or (0 if sh else 1)); si = int(b[6:]) if sh else 0
            if not 0 <= si < sn: n, si, sn = None, 0, 1                    # older run.json without a shard count
        T = list(csv.DictReader(open(os.path.join(d, "scores.tsv")), delimiter="\t")); ids = {int(r["id"]) for r in T}
        kinds = {}
        for r in T: k = (r.get("status") or "?").strip(); kinds[k] = kinds.get(k, 0) + 1     # ok / noct / small / no_sheet ...
        want = {i for i in range(n) if i % sn == si} if n is not None else None
        rows.append(dict(dir=d, done=len(ids & want) if want is not None else len(ids), expected=None if want is None else len(want),
                         missing=None if want is None else sorted(want - ids), shard=f"{si}/{sn}", kinds=dict(sorted(kinds.items(), key=lambda t: -t[1]))))
        if n is not None:
            k = json.dumps([m.get("surfaces"), m.get("crop"), m.get("tile"), m.get("box"), m.get("spacing"), n, sn]); s = sets.setdefault(k, dict(jobs=n, n=sn, ids=set(), shards=set(), dirs=[]))
            s["ids"] |= ids; s["shards"].add(si); s["dirs"].append(d)
    groups = [dict(jobs=s["jobs"], shards=s["n"], have=sorted(s["shards"]), missing=sorted(set(range(s["jobs"])) - s["ids"]), dirs=s["dirs"]) for s in sets.values()]
    return rows, groups
LCOLS = ("letter_score", "letters", "fails")
COLS = ("rank", "id", "text_score", "side", "bright_a", "bright_b", "area_cm2", "tracked", "seed_z", "seed_y", "seed_x", "um", "src", "png")
def write_tsv(P, path, with_letters=False):
    """All patches, best first, one row each: scores, seed, source tifxyz window and the ink-side PNG, for sorting and
    merging runs in a spreadsheet or script. with_letters adds letterscore.py on each patch's ink face (letter_score,
    letters, fails; blank where the two faces differ in size)."""
    with open(path, "w") as f:
        f.write("\t".join(COLS + (LCOLS if with_letters else ())) + "\n")
        for i, p in enumerate(P):
            row = [i + 1, p["id"], f"{p['ts']:.4f}", p["side"], f"{p['fa']:.4f}", f"{p['fb']:.4f}", f"{p['area']:.3f}", f"{p['tracked']:.3f}",
                   *p["seed"], p["um"], p.get("src", ""), f"{p['base']}_{p['side']}.png"]
            if with_letters:
                L = letters(p) or {}; row += [L.get("score", ""), L.get("letters", ""), L.get("fails", "")]
            f.write("\t".join(map(str, row)) + "\n")
