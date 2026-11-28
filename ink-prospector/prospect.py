#!/usr/bin/env python3
"""Ink Prospector command line.

  prospect.py scan   --scroll PHerc0306B --box z0 z1 y0 y1 x0 x1 --out runs/demo [--spacing 250] [--device cuda] [--shard i/n]
  prospect.py rank   runs/demo [-n 20] [--sheet top.png] [--tsv ranked.tsv]
  prospect.py status runs/sheet-s0 runs/sheet-s1 ...   (score_tifxyz progress: what each shard has left, ids no shard has)
  prospect.py spread --scroll PHerc0306B --seed z y x --side a --out runs/demo-spread [--half-nodes 320]
  prospect.py viewer runs/demo --out site [--title "PHerc0306B"]
  prospect.py auto   --scroll PHerc0306B [S ...] --out auto [--box ...] [--workers 2]   (whole pipeline unattended; auto.py)
  prospect.py serve  --runs runs [--port 8765]   (the app: start, watch, resume and browse runs in a browser; docs/APP.md)

The work folder (PROSPECTOR_HOME, default: current folder) holds code/, data/weights/reader-v3.pth and ext/villa-mip,
as made by setup.sh or the Docker image. Scrolls are named in scrolls.json; --ct/--m7/--um override it.
"""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); HOME = os.path.abspath(os.environ.get("PROSPECTOR_HOME", "."))
def scroll(a):
    reg = json.load(open(os.path.join(HERE, "scrolls.json"))); s = dict(reg.get(a.scroll, {})) if a.scroll else {}
    for k in ("ct", "m7", "um"):
        if getattr(a, k, None): s[k] = getattr(a, k)
    if "ct" not in s or "m7" not in s: sys.exit(f"unknown scroll {a.scroll!r}: name one from scrolls.json or pass --ct and --m7")
    return s
def run(script, *args): return subprocess.call([sys.executable, os.path.join(HERE, script), *map(str, args)], cwd=HOME)
def check_runs(runs, paths):
    """A run folder that holds no scan is named on stderr; if none of them does, stop (a typo, not an empty run)."""
    bad = runs.not_runs(paths)
    for b in bad: print(f"no scores.tsv in {b} or its shard-*/ folders: not a scan run", file=sys.stderr)
    if bad and len(bad) == len(paths): sys.exit(f"nothing to read in {len(bad)} folder(s)")
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sys.path.insert(0, HERE); import release_notes
    cl = os.path.join(HERE, "CHANGELOG.md")                               # setup.sh / run_*.sh copy only *.py into code/
    ap.add_argument("--version", action="version", version="ink-prospector " + (release_notes.version(open(cl).read()) if os.path.exists(cl) else "(version unknown: no CHANGELOG.md next to prospect.py)"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p): p.add_argument("--scroll"); p.add_argument("--ct"); p.add_argument("--m7"); p.add_argument("--um", type=float); p.add_argument("--out", required=True)
    p = sub.add_parser("scan"); common(p); p.add_argument("--box", type=int, nargs=6, required=True); p.add_argument("--spacing", type=int, default=250)
    p.add_argument("--half-nodes", type=int, default=128); p.add_argument("--shard", default="0/1"); p.add_argument("--device", default="cpu"); p.add_argument("--max-patches", type=int); p.add_argument("--fuse", action="store_true")
    p.add_argument("--max-cache-gb", type=float, help="cap on CT/m7 chunks held in memory per scan (uncapped, a scan can pass 6 GB)")
    p = sub.add_parser("rank"); p.add_argument("run", nargs="+"); p.add_argument("-n", type=int, default=20); p.add_argument("--sheet"); p.add_argument("--tsv", help="write every usable patch, best first, to this table"); p.add_argument("--letters", action="store_true", help="with --tsv: add letterscore.py columns (letter_score, letters, fails)"); p.add_argument("--min-area", type=float, default=0.0, help="leave out patches with less than this many cm2 read")
    p = sub.add_parser("spread"); common(p); p.add_argument("--seed", type=float, nargs=3, required=True); p.add_argument("--side", choices="ab", default="a"); p.add_argument("--half-nodes", type=int, default=320)
    p = sub.add_parser("status"); p.add_argument("run", nargs="+")
    p = sub.add_parser("stitch"); p.add_argument("run", nargs="+"); p.add_argument("--out", required=True); p.add_argument("--top", type=int, default=30)
    p = sub.add_parser("viewer"); p.add_argument("run", nargs="+"); p.add_argument("--out", required=True); p.add_argument("--title", default=""); p.add_argument("--full", type=int, default=50, help="copy full-size ink maps of the N best patches (0: none)"); p.add_argument("--min-area", type=float, default=0.0, help="leave out patches with less than this many cm2 read")
    p.add_argument("--known", nargs="+", help="x.tif y.tif z.tif r0 r1 c0 c1 [label]: surface with known text")
    p = sub.add_parser("auto", add_help=False); p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("serve"); p.add_argument("--runs", default="runs", help="folder holding the run folders (made if missing)")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765)
    if sys.argv[1:2] == ["auto"]: sys.exit(run("auto.py", *sys.argv[2:]))   # argparse REMAINDER rejects a leading --option
    a = ap.parse_args(); ab = os.path.abspath
    if a.cmd == "serve":
        sys.path.insert(0, HERE); import server
        return server.main(a.runs, a.host, a.port)
    if a.cmd == "scan":
        sys.path.insert(0, HERE); import runs; i, n = runs.shard(a.shard); runs.lattice(a.box, a.spacing)      # before run.json is rewritten
        s = scroll(a); out = ab(a.out) if n == 1 else os.path.join(ab(a.out), f"shard-{i}"); os.makedirs(out, exist_ok=True)
        json.dump(dict(scroll=a.scroll, box=a.box, spacing=a.spacing, half_nodes=a.half_nodes, shards=n, **s), open(os.path.join(ab(a.out), "run.json"), "w"), indent=1)
        sys.exit(run("scan.py", "--ct", s["ct"], "--m7", s["m7"], "--out", out, "--box", *a.box, "--spacing", a.spacing, "--half-nodes", a.half_nodes, "--shard", a.shard,
                     "--device", a.device, "--ct-level", s.get("ct_level", 0), "--um", s.get("um", 0), *(["--max-patches", a.max_patches] if a.max_patches else []), *(["--fuse"] if a.fuse else []),
                     *(["--max-cache-gb", a.max_cache_gb] if a.max_cache_gb else [])))
    if a.cmd == "rank":
        sys.path.insert(0, HERE); import runs
        check_runs(runs, [ab(r) for r in a.run])
        P = runs.patches([ab(r) for r in a.run], min_area=a.min_area); print(f"{len(P)} usable patches, {sum(p['area'] for p in P):.1f} cm2 read")
        for i, p in enumerate(P[:a.n]): print(f"{i + 1:3d}  {p['id']:>10}  text {p['ts']:.4f} side {p['side']}  bright a/b {p['fa']:.3f}/{p['fb']:.3f}  seed z y x {p['seed'][0]} {p['seed'][1]} {p['seed'][2]}" + (f"  {p['src']}" if p.get("src") else ""))
        if a.tsv:
            runs.write_tsv(P, a.tsv, a.letters); print(f"table of {len(P)} patches: {a.tsv}")
        if a.sheet and not P: print("no usable patches, so no contact sheet")
        elif a.sheet:
            from PIL import Image
            c = 6; T = P[:a.n]; g = Image.new("L", (c * 256, ((len(T) + c - 1) // c) * 256))
            for i, p in enumerate(T): g.paste(Image.open(f"{p['base']}_{p['side']}.png").convert("L").resize((256, 256)), ((i % c) * 256, (i // c) * 256))
            g.save(a.sheet); print("contact sheet:", a.sheet)
        return
    if a.cmd == "status":
        sys.path.insert(0, HERE); import runs
        check_runs(runs, [ab(r) for r in a.run])
        rows, groups = runs.status([ab(r) for r in a.run]); short = lambda m: ", ".join(map(str, m[:10])) + (" ..." if len(m) > 10 else "")
        for r in rows:
            print(f"{r['dir']}: {r['done']}" + (f" of {r['expected']} done" + (f", missing {short(r['missing'])}" if r["missing"] else ", complete") if r["expected"] is not None else " scored (no job count in run.json)")
                  + (" [" + ", ".join(f"{k} {v}" for k, v in r["kinds"].items()) + "]" if r["kinds"] else ""))
        for g in groups:
            gone = sorted(set(range(g["shards"])) - set(g["have"]))
            print(f"set of {g['jobs']} patches in {g['shards']} shard(s): " + ("complete" if not g["missing"] else f"{len(g['missing'])} missing ({short(g['missing'])})")
                  + (f"; no folder for shard(s) {', '.join(map(str, gone))}" if gone else ""))
        sys.exit(1 if any(g["missing"] for g in groups) or any(r["missing"] for r in rows) else 0)
    if a.cmd == "spread":
        s = scroll(a); sys.exit(run("spread.py", s["ct"], s["m7"], ab(a.out), *a.seed, a.half_nodes, "fwd" if a.side == "a" else "rev", s.get("ct_level", 0)))
    if a.cmd == "stitch":
        sys.path.insert(0, HERE); import runs; check_runs(runs, [ab(r) for r in a.run])
        sys.exit(run("stitch.py", ab(a.out), *[ab(r) for r in a.run], "--top", a.top))
    if a.cmd == "viewer":
        sys.path.insert(0, HERE); import runs; check_runs(runs, [ab(r) for r in a.run])
        sys.exit(run("build_viewer.py", ab(a.out), a.title or "scan", *[ab(r) for r in a.run], "--full", a.full, "--min-area", a.min_area, *(["--known"] + [ab(x) if x.endswith(".tif") else x for x in a.known] if a.known else [])))
if __name__ == "__main__": main()
