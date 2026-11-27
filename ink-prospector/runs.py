"""Read the patches of one or more scan folders (a folder with scores.tsv, or a run folder holding shard-*/)."""
import csv, glob, json, os
import textscore
def scan_dirs(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "shard-*"))) if glob.glob(os.path.join(p, "shard-*")) else [p]
    return [d for d in out if os.path.exists(os.path.join(d, "scores.tsv"))]
def meta(d):
    for f in (os.path.join(d, "run.json"), os.path.join(os.path.dirname(d.rstrip("/")), "run.json")):
        if os.path.exists(f): return json.load(open(f))
    return {}
def patches(paths, um=None):
    """Usable patches of all folders, best text score first. Each: dict with id, dir, base path, scores."""
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
            seen.add(r["id"]); s, ta, tb = textscore.score(base + "_a.png", base + "_b.png", u)
            rows.append(dict(id=f"{tag}-{int(r['id'])}", dir=d, base=base, um=u, seed=[int(float(r[k])) for k in ("seed_z", "seed_y", "seed_x")],
                             tracked=float(r["tracked"]), area=float(r["area_cm2"]), fa=float(r["frac_a"]), fb=float(r["frac_b"]), ts=abs(s), side="a" if s > 0 else "b",
                             src=src.get(r["id"], "")))
    return sorted(rows, key=lambda r: -r["ts"])
def status(paths):
    """Progress of score_tifxyz folders: per folder done/expected and missing ids, then per set of shards
    made from the same surfaces, the ids no folder has (a shard never run, or cut off). Folders without a
    job count in run.json (older runs, scan.py runs) report done only."""
    rows, sets = [], {}
    for d in scan_dirs(paths):
        m = meta(d); n = m.get("jobs"); si, sn = (int(v) for v in str(m.get("shard") or "0/1").split("/"))
        ids = {int(r["id"]) for r in csv.DictReader(open(os.path.join(d, "scores.tsv")), delimiter="\t")}
        want = {i for i in range(n) if i % sn == si} if n is not None else None
        rows.append(dict(dir=d, done=len(ids & want) if want is not None else len(ids), expected=None if want is None else len(want),
                         missing=None if want is None else sorted(want - ids), shard=f"{si}/{sn}"))
        if n is not None:
            k = json.dumps([m.get("surfaces"), m.get("crop"), m.get("tile"), n, sn]); s = sets.setdefault(k, dict(jobs=n, n=sn, ids=set(), shards=set(), dirs=[]))
            s["ids"] |= ids; s["shards"].add(si); s["dirs"].append(d)
    groups = [dict(jobs=s["jobs"], shards=s["n"], have=sorted(s["shards"]), missing=sorted(set(range(s["jobs"])) - s["ids"]), dirs=s["dirs"]) for s in sets.values()]
    return rows, groups
