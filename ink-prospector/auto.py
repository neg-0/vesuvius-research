"""Auto mode: one command from a scroll to ranked, read, stitched leads ready for VC3D.

usage: auto.py --scroll S [S ...] --out NAME [--runs runs] [--box z0 z1 y0 y1 x0 x1] [--spacing 800] [--workers 2]
               [--leads 10] [--spread 3] [--device cpu] [--sweep DIR]
(prospect.py auto calls this; run from the work folder, PROSPECTOR_HOME.)

For each scroll, into <runs>/<NAME>-<scroll>/ (a normal run folder, so rank, status, viewer and the app all read it):
  1. sweep    seeds a lattice through the whole scroll volume (or --box) and scans it, split over --workers local
              processes (shard-i/). Resume-safe. --sweep DIR uses an existing scan or run folder instead.
  2. leads    ranks every patch by text score, adds letterscore on its saved maps, keeps the best --leads
              (auto/leads.tsv; a lead must be one-sided: the other face under a third of the inked one).
  3. spread   grows the best --spread leads into 22 mm patches and reads the inked face with fused reads
              (auto/spread/<id>/), then scores letters on each.
  4. stitch   joins patches on the same sheet (_stitch/), as `prospect.py stitch`.
  5. viewer   builds the 3D viewer page (_viewer/).
  6. vc       writes the leads and the spread patches as tifxyz surfaces, a VC3D point collection and grow.sh (auto/vc/),
              as vc_export.py: open auto/vc/paths/* in VC3D, or grow them with VC's tracer.
auto/auto.json records each stage (done, seconds, what it made); a stage that is done is skipped on the next call, so
running the same command again resumes. auto/REPORT.md sums it up.
"""
import argparse, csv, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import runs as runlib
STAGES = ("sweep", "leads", "spread", "stitch", "viewer", "vc")
PY = sys.executable


def log(s): print(f"[auto {time.strftime('%H:%M:%S')}] {s}", flush=True)


def whole_box(ct, level=0, margin=200):
    """The scroll volume minus a margin, z0 z1 y0 y1 x0 x1 (as run_whole.sh)."""
    from lazyvol import LazyVol
    shp = LazyVol(ct, level=level).shape
    return [int(v) for n in shp for v in (margin, n - margin)]


def lattice_size(box, S):
    z0, z1, y0, y1, x0, x1 = box
    return len(range(z0 + S // 2, z1, S)) * len(range(y0 + S // 2, y1, S)) * len(range(x0 + S // 2, x1, S))


class State:
    def __init__(self, run):
        self.dir = os.path.join(run, "auto"); os.makedirs(self.dir, exist_ok=True); self.f = os.path.join(self.dir, "auto.json")
        self.s = json.load(open(self.f)) if os.path.exists(self.f) else dict(stages={})

    def save(self):
        tmp = self.f + ".tmp"; json.dump(self.s, open(tmp, "w"), indent=1); os.replace(tmp, self.f)

    def done(self, k): return self.s["stages"].get(k, {}).get("state") == "done"

    def mark(self, k, state, **kw):
        st = self.s["stages"].setdefault(k, {}); st.update(state=state, at=time.time(), **kw); self.save()


def sweep(a, sc, run, st):
    """Scan the box with --workers shard processes; returns the folder to rank."""
    if a.sweep: return os.path.abspath(a.sweep)
    box = a.box or whole_box(sc["ct"], sc.get("ct_level", 0)); n = max(1, a.workers)
    json.dump(dict(scroll=sc["name"], box=box, spacing=a.spacing, half_nodes=a.half_nodes, shards=n, um=sc.get("um"), ct=sc["ct"], m7=sc["m7"], source="auto"),
              open(os.path.join(run, "run.json"), "w"), indent=1)
    seeds = lattice_size(box, a.spacing); log(f"{sc['name']}: sweep of box {box} at spacing {a.spacing}: {seeds} seeds in {n} worker(s)")
    st.mark("sweep", "running", box=box, seeds=seeds, workers=n)
    env = dict(os.environ, OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 4) // n)))
    procs = [subprocess.Popen([PY, os.path.join(HERE, "scan.py"), "--ct", sc["ct"], "--m7", sc["m7"], "--out", os.path.join(run, f"shard-{i}"),
                              "--box", *map(str, box), "--spacing", str(a.spacing), "--half-nodes", str(a.half_nodes), "--shard", f"{i}/{n}",
                              "--device", a.device, "--ct-level", str(sc.get("ct_level", 0)), "--um", str(sc.get("um") or 0),
                              "--max-cache-gb", str(a.max_cache_gb)], env=env) for i in range(n)]
    codes = [p.wait() for p in procs]
    if any(codes): raise RuntimeError(f"sweep: scan worker(s) exited {codes}; run the same command again to resume")
    return run


def one_sided(p): return min(p["fa"], p["fb"]) < max(p["fa"], p["fb"]) / 3


def leads(a, src, st):
    P = runlib.patches([src]); L = []
    for p in P:
        if not one_sided(p): continue
        r = runlib.letters(p) or {}; p.update(ls=r.get("score"), lrow=r.get("row"), letters=r.get("letters")); L.append(p)
        if len(L) >= a.leads: break
    with open(os.path.join(st.dir, "leads.tsv"), "w") as f:
        f.write("lead\tid\ttext_score\tletters\trow\tpasses\tside\tarea_cm2\tseed_z\tseed_y\tseed_x\tpng\n")
        for i, p in enumerate(L):
            f.write(f"{i + 1}\t{p['id']}\t{p['ts']:.4f}\t{p.get('ls')}\t{p.get('lrow')}\t{p.get('letters')}\t{p['side']}\t{p['area']:.3f}\t"
                    f"{p['seed'][0]}\t{p['seed'][1]}\t{p['seed'][2]}\t{p['base']}_{p['side']}.png\n")
    st.mark("leads", "done", patches=len(P), leads=len(L), best=[dict(id=p["id"], ts=round(p["ts"], 4), ls=p.get("ls"), side=p["side"], seed=p["seed"],
                                                                       png=f"{p['base']}_{p['side']}.png") for p in L])
    return L


def read_leads(st):
    return [dict(id=r["id"], side=r["side"], seed=[int(r["seed_z"]), int(r["seed_y"]), int(r["seed_x"])])
            for r in csv.DictReader(open(os.path.join(st.dir, "leads.tsv")), delimiter="\t")]


def spread_letters(d, um):
    """letterscore on a spread's fused read of the inked face, block-averaged 4x to the ~35 um the defaults were set at.
    Only one face is read in a spread, so one-sidedness comes from the scan, not from this score."""
    import letterscore
    o = np.load(os.path.join(d, "ink.npy")).astype(np.float32); v = np.load(os.path.join(d, "valid.npy"))
    h, w = (o.shape[0] // 4) * 4, (o.shape[1] // 4) * 4
    ob = o[:h, :w].reshape(h // 4, 4, w // 4, 4).mean((1, 3)); vb = v[:h, :w].reshape(h // 4, 4, w // 4, 4).mean((1, 3)) > 0.5
    return letterscore.score_maps(ob, np.zeros_like(ob), vb, um * 4)


def spread(a, sc, L, st):
    out = []
    for p in L[:a.spread]:
        d = os.path.join(st.dir, "spread", p["id"])
        if not os.path.exists(os.path.join(d, "ink.npy")):
            log(f"spread {p['id']} from seed {p['seed']} side {p['side']}")
            c = subprocess.call([PY, os.path.join(HERE, "spread.py"), sc["ct"], sc["m7"], d, *map(str, p["seed"]), str(a.spread_nodes),
                                 "fwd" if p["side"] == "a" else "rev", str(sc.get("ct_level", 0))])
            if c: log(f"spread {p['id']} failed (exit {c}); skipped"); out.append(dict(id=p["id"], ok=False)); continue
        r = spread_letters(d, sc.get("um") or 8.64)
        out.append(dict(id=p["id"], ok=True, png=os.path.join(d, "ink_ds2.png"), ls=r["score"], lrow=r["row"], letters=r["letters"]))
    st.mark("spread", "done", spreads=out)
    return out


def vc(a, src, st, spreads, sc):
    import vc_export
    from export_tifxyz import write_tifxyz
    d = os.path.join(st.dir, "vc"); vc_export.main([d, src, "-n", str(a.leads)])
    n = 0
    for s in spreads:
        if not s.get("ok"): continue
        g = np.load(os.path.join(st.dir, "spread", s["id"], "grid.npz"))
        write_tifxyz(g["P"], g["ok"], os.path.join(d, "paths", f"prospector_spread_{s['id']}"), every=5, scroll=sc["name"], scan=os.path.basename(sc["ct"]),
                     extra={"prospector": {"spread_of": s["id"], "letter_score": s.get("ls")}}); n += 1
    st.mark("vc", "done", folder=d, spreads=n)


def report(run, st, sc):
    S = st.s["stages"]; L = S.get("leads", {}).get("best", []); sp = {s["id"]: s for s in S.get("spread", {}).get("spreads", [])}
    sw = S.get("sweep", {})
    lines = [f"# Auto run: {sc['name']}", "", f"Folder `{run}`. Sweep: {sw.get('seeds', '?')} seeds, box {sw.get('box', 'given')}, "
             f"{S.get('leads', {}).get('patches', '?')} usable patches.", "", "| lead | patch | text score | letters (scan) | spread letters | side | seed z y x |", "|---|---|---|---|---|---|---|"]
    for i, p in enumerate(L):
        s = sp.get(p["id"], {})
        lines.append(f"| {i + 1} | {p['id']} | {p['ts']} | {p.get('ls')} | {s.get('ls', '')}{' (row)' if s.get('letters') else ''} | {p['side']} | {' '.join(map(str, p['seed']))} |")
    lines += ["", "Open in VC3D: copy `auto/vc/paths/*` into a volpkg's `paths/` (or open them directly) and load "
              "`auto/vc/prospector_hits.json` as a point collection; `auto/vc/grow.sh` grows each lead with VC's tracer.",
              "3D map: `_viewer/index.html` (serve the folder). Stitched sheets: `_stitch/`. A lead is a place to look, not a reading."]
    open(os.path.join(st.dir, "REPORT.md"), "w").write("\n".join(lines) + "\n")


def run_scroll(a, sc):
    run = os.path.join(os.path.abspath(a.runs), f"{a.out}-{sc['name']}"); os.makedirs(run, exist_ok=True); st = State(run)
    st.s.update(scroll=sc["name"], args={k: v for k, v in vars(a).items() if k != "scroll"}); st.save()
    src = os.path.abspath(a.sweep) if a.sweep else run
    for k in STAGES:
        if st.done(k): log(f"{sc['name']}: {k} already done"); continue
        t = time.time(); log(f"{sc['name']}: {k}")
        try:
            if k == "sweep": sweep(a, sc, run, st); st.mark("sweep", "done", **({"from": src} if a.sweep else {}))
            elif k == "leads": leads(a, src, st)
            elif k == "spread": spread(a, sc, read_leads(st), st)
            elif k == "stitch":
                c = subprocess.call([PY, os.path.join(HERE, "stitch.py"), os.path.join(run, "_stitch"), src, "--top", str(max(a.leads, 30))])
                st.mark("stitch", "done" if c == 0 else "failed", exit=c)
            elif k == "viewer":
                c = subprocess.call([PY, os.path.join(HERE, "build_viewer.py"), os.path.join(run, "_viewer"), f"{sc['name']} auto", src, "--full", "50"])
                st.mark("viewer", "done" if c == 0 else "failed", exit=c)
            elif k == "vc": vc(a, src, st, st.s["stages"].get("spread", {}).get("spreads", []), sc)
            st.s["stages"][k]["secs"] = round(time.time() - t); st.save()
        except Exception as e:
            st.mark(k, "failed", error=f"{type(e).__name__}: {e}"); report(run, st, sc); raise
    report(run, st, sc); log(f"{sc['name']}: done -> {run}/auto/REPORT.md")
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scroll", nargs="+"); ap.add_argument("--ct"); ap.add_argument("--m7"); ap.add_argument("--um", type=float)
    ap.add_argument("--out", required=True); ap.add_argument("--runs", default="runs"); ap.add_argument("--box", type=int, nargs=6)
    ap.add_argument("--spacing", type=int, default=800); ap.add_argument("--half-nodes", type=int, default=128); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--leads", type=int, default=10); ap.add_argument("--spread", type=int, default=3); ap.add_argument("--spread-nodes", type=int, default=320)
    ap.add_argument("--max-cache-gb", type=float, default=2.0, help="CT/m7 chunk cache per sweep worker (default 2 GB; uncapped a scan can pass 6 GB)")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--sweep", help="use this existing scan or run folder instead of sweeping")
    a = ap.parse_args(argv)
    reg = json.load(open(os.path.join(HERE, "scrolls.json"))); todo = []
    for s in a.scroll or [None]:
        sc = dict(reg.get(s, {})) if s else {}
        for k in ("ct", "m7", "um"):
            if getattr(a, k): sc[k] = getattr(a, k)
        if "ct" not in sc or "m7" not in sc: sys.exit(f"unknown scroll {s!r}: name one from scrolls.json or pass --ct and --m7")
        sc["name"] = s or "scroll"; todo.append(sc)
    if a.sweep and len(todo) > 1: sys.exit("--sweep names one existing run, so give one scroll")
    failed = []
    for sc in todo:
        try: run_scroll(a, sc)
        except Exception as e: log(f"{sc['name']}: stopped: {e}"); failed.append(sc["name"])
    if failed: sys.exit(f"auto: {', '.join(failed)} did not finish; run the same command again to resume")


if __name__ == "__main__":
    main()
