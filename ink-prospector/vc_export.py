"""Hand the best Prospector hits to Volume Cartographer (VC3D and vc_grow_seg_from_seed).

usage: vc_export.py <out dir> <scan or run dir>... [-n 10] [--every 5] [--m7 ZARR] [--generations 40]
Writes, for the top N patches by text score:
  <out>/paths/<id>/            each patch as a tifxyz surface (every 5th node = VC's 20-voxel tracer step)
  <out>/prospector_hits.json   a VC3D point collection, one point per hit (x y z, on the sheet, at the patch centre)
  <out>/hits.tsv               rank, id, text score, inked face, centre x y z
  <out>/params.json, grow.sh   vc_grow_seg_from_seed --resume commands that grow each hit into a larger segment on
                               the surface prediction (needs VC's tools, e.g. the ghcr.io/scrollprize/villa/
                               volume-cartographer image)
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runs
from export_tifxyz import write_tifxyz

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"


def centre(P, ok):
    """Tracked node nearest the middle of the tracked area (on the sheet), as x y z."""
    v = P[ok]; c = v[np.argmin(np.linalg.norm(v - v.mean(0), axis=1))]
    return [float(c[2]), float(c[1]), float(c[0])]


def point_collection(hits, name="prospector hits"):
    pts = {str(i + 1): {"p": h["xyz"], "creation_time": 0, "wind_a": None} for i, h in enumerate(hits)}
    return {"vc_pointcollections_json_version": "1",
            "collections": {"1": {"name": name, "points": pts, "metadata": {"winding_is_absolute": True}, "color": [1.0, 0.35, 0.1]}}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("run", nargs="+"); ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--every", type=int, default=5); ap.add_argument("--m7", help="surface prediction zarr for grow.sh (default: from run.json)")
    ap.add_argument("--generations", type=int, default=40)
    a = ap.parse_args(argv)
    P = runs.patches([os.path.abspath(r) for r in a.run])[:a.n]
    if not P: sys.exit("no usable patches")
    m7 = a.m7 or runs.meta(P[0]["dir"]).get("m7", "")
    if m7 and not (m7.startswith(("http://", "https://", "s3://")) or os.path.isdir(m7)): m7 = BUCKET + m7
    meta0 = runs.meta(P[0]["dir"])
    os.makedirs(os.path.join(a.out, "paths"), exist_ok=True); hits = []; lines = []
    for k, p in enumerate(P):
        g = np.load(p["base"] + ".npz"); ok = g["ok"].astype(bool) & np.isfinite(g["P"]).all(-1)
        sid = f"prospector_{p['id']}"; d = os.path.join(a.out, "paths", sid)
        n = write_tifxyz(g["P"], ok, d, every=a.every, scroll=meta0.get("scroll") or "", scan=os.path.basename(meta0.get("ct", "")),
                         extra={"prospector": {"rank": k + 1, "text_score": round(p["ts"], 4), "ink_face": p["side"], "seed_zyx": p["seed"]}})
        hits.append(dict(rank=k + 1, id=p["id"], ts=p["ts"], side=p["side"], xyz=centre(g["P"], ok), nodes=n, path=sid))
        lines.append(f'vc_grow_seg_from_seed -v "$M7" -t grown/{sid}_grown -p params.json --resume paths/{sid} --segment-name {sid}_grown || true')
    json.dump(point_collection(hits), open(os.path.join(a.out, "prospector_hits.json"), "w"), indent=1)
    with open(os.path.join(a.out, "hits.tsv"), "w") as f:
        f.write("rank\tid\ttext_score\tink_face\tx\ty\tz\tnodes\ttifxyz\n")
        for h in hits: f.write(f"{h['rank']}\t{h['id']}\t{h['ts']:.4f}\t{h['side']}\t{h['xyz'][0]:.1f}\t{h['xyz'][1]:.1f}\t{h['xyz'][2]:.1f}\t{h['nodes']}\tpaths/{h['path']}\n")
    # voxelsize: remote volumes are not probed for it, and without it VC computes 0 cm2 and discards the surface
    json.dump({"mode": "seed", "generations": a.generations, "voxelsize": float(meta0.get("um") or P[0]["um"]), "min_area_cm": 0,
               "cache_size": 2000000000, "thread_limit": 4}, open(os.path.join(a.out, "params.json"), "w"), indent=1)
    with open(os.path.join(a.out, "grow.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\n# Grow each Prospector hit into a larger segment with VC's tracer, on the surface prediction.\n"
                "# Run from this folder. Output: grown/<id>_grown/ (tifxyz, opens in VC3D).\nset -u\ncd \"$(dirname \"$0\")\"\n"
                f'M7="${{M7:-{m7}}}"\nmkdir -p grown\n' + "\n".join(lines) + "\n")
    os.chmod(os.path.join(a.out, "grow.sh"), 0o755)
    print(f"{len(hits)} hits -> {a.out} (paths/, prospector_hits.json, hits.tsv, grow.sh)")


if __name__ == "__main__":
    main()
