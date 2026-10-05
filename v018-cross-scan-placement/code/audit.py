"""V-018 audit: run sv_placement.check on every segment in the bucket with 2+ surface volumes.
Resumable: skips segments already in the output JSONL.  usage: audit.py out.jsonl [windows]"""
import json, os, sys, concurrent.futures as cf
import sv_placement as sp

out = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 3
done = set()
if os.path.exists(out):
    done = {json.loads(l)["segment"] for l in open(out)}
scrolls = [p for p in sp.ls("") if p.startswith("PHerc")]
segs = []
for sc in scrolls:
    for s in sp.ls(sc + "segments/"):
        seg = sc + "segments/" + s
        if seg in done:
            continue
        if sum(p.rstrip("/").endswith(".zarr") for p in sp.ls(seg + "surface-volumes/")) >= 2:
            segs.append(seg)
print(len(segs), "segments to check", flush=True)


def one(seg):
    try:
        return sp.check(seg, n_windows=nw, verbose=False)
    except Exception as e:
        return {"segment": seg, "status": "error " + str(e)[:200]}


with cf.ThreadPoolExecutor(3) as ex, open(out, "a") as f:
    for r in ex.map(one, segs):
        f.write(json.dumps(r) + "\n"); f.flush()
        print(r["segment"], r.get("status"), {k[:12]: (p["median_ncc0"], p["median_best"]) for k, p in r.get("pairs", {}).items()}, flush=True)
