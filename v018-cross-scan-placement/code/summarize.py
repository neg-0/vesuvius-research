"""Summarise audit JSONL by scroll and scan pair.  usage: summarize.py audit.jsonl [out.md]"""
import collections, itertools, json, re, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
st = collections.Counter(r.get("status", "?").split(" ")[0] for r in rows)
recs = []
for r in rows:
    for v, p in r.get("pairs", {}).items():
        w = [x for x in p["windows"] if "shift_um" in x]
        if not w:
            continue
        sh = np.median([x["shift_um"] for x in w], 0)
        recs.append(dict(scroll=r["segment"].split("/")[0], ref=r["reference"].split("-")[0], other=v.split("-")[0],
                         seg=r["segment"], ncc0=p["median_ncc0"], best=p["median_best"],
                         inplane=float(np.hypot(sh[1], sh[2])), dz=float(sh[0]), grid=p["grid_um"]))
lines = [f"Segments: {len(rows)} ({dict(st)}); scan pairs with data: {len(recs)}", "",
         "| Scroll | Reference | Other scan | Segments | Median NCC, no shift | Median NCC, best shift | Median in-plane shift (µm) | Grid (µm) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
k = lambda t: (t["scroll"], t["ref"], t["other"])
for key, g in itertools.groupby(sorted(recs, key=k), key=k):
    g = list(g)
    lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(g)} | {np.median([x['ncc0'] for x in g]):.2f} | "
                 f"{np.median([x['best'] for x in g]):.2f} | {np.median([x['inplane'] for x in g]):.0f} | {g[0]['grid']:.1f} |")
bad = sorted([x for x in recs if x["best"] < 0.4], key=lambda x: x["best"])
lines += ["", f"Pairs whose best-shift NCC is under 0.4 (no placement explains the other render): {len(bad)}"]
lines += [f"- {x['seg']} {x['other']}: best {x['best']:.2f}" for x in bad[:40]]
txt = "\n".join(lines); print(txt)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(txt + "\n")
