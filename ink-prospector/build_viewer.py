"""Build the viewer page for scan folders: <out>/index.html, data.json, atlas.jpg.

usage: build_viewer.py <out> <title> <scan_or_run_dir> [...] [--known x.tif y.tif z.tif r0 r1 c0 c1 label] [--full N] [--min-area CM2]
--known marks a surface with known text (tifxyz grid, optional row/col window) and tags patches lying on it.
--full N copies the full-size ink maps of the N best patches (default 50, 0 for none) to <out>/full/, and the detail
panel links them; the atlas only holds 256 px thumbnails.
--min-area CM2 leaves out patches with less read surface (as prospect.py rank --min-area).
"""
import sys, os, json, shutil, numpy as np
from PIL import Image
import runs
def main(argv):
    out, title = argv[0], argv[1]; rest = argv[2:]; known = None
    full = 50
    if "--full" in rest: k = rest.index("--full"); full = int(rest[k + 1]); rest = rest[:k] + rest[k + 2:]
    min_area = 0.0
    if "--min-area" in rest: k = rest.index("--min-area"); min_area = float(rest[k + 1]); rest = rest[:k] + rest[k + 2:]
    if "--known" in rest: k = rest.index("--known"); known = rest[k + 1:]; rest = rest[:k]
    P = runs.patches(rest, min_area=min_area); assert P, "no usable patches"
    os.makedirs(out, exist_ok=True); tree = None; kp = []; klabel = ""
    if known:
        import tifffile
        from scipy import ndimage as ndi
        from scipy.spatial import cKDTree
        W = np.stack([tifffile.imread(known[i]) for i in (2, 1, 0)], -1)                     # -> z, y, x
        r0, r1, c0, c1 = [int(v) for v in known[3:7]]; sub = W[r0:r1, c0:c1]; klabel = " ".join(known[7:]) or "known text"
        up = np.stack([ndi.zoom(sub[..., i], 4, order=1) for i in range(3)], -1).reshape(-1, 3); up = up[(up > 0).all(1)]
        tree = cKDTree(up); kp = up[::max(1, len(up) // 900)].astype(int).reshape(-1).tolist()
    C = 6; atlas = Image.new("L", (C * 512, ((len(P) + C - 1) // C) * 256)); pats = []
    for i, p in enumerate(P):
        g = np.load(p["base"] + ".npz"); G, ok = g["P"], g["ok"]
        for k, s in enumerate("ab"): atlas.paste(Image.open(f"{p['base']}_{s}.png").convert("L").resize((256, 256)), ((i % C) * 512 + k * 256, (i // C) * 256))
        on = 0.0
        if tree is not None: d, _ = tree.query(G[ok][::5], distance_upper_bound=6); on = float(np.isfinite(d).mean())
        fl = []
        if i < full:                                                     # P is best first
            os.makedirs(os.path.join(out, "full"), exist_ok=True)
            for s in "ab": shutil.copy(f"{p['base']}_{s}.png", os.path.join(out, "full", f"{p['id']}_{s}.png")); fl.append(f"full/{p['id']}_{s}.png")
        pats.append(dict(full=fl, id=p["id"], seed=p["seed"], c=[int(v) for v in G[ok].mean(0)], n=[round(float(v), 2) for v in g["n0"]], tr=round(p["tracked"], 2),
                         area=round(p["area"], 2), fa=round(p["fa"], 3), fb=round(p["fb"], 3), ts=round(p["ts"], 4), side=p["side"], known=round(on, 2), src=p.get("src", ""),
                         pts=[int(v) for v in G[::16, ::16][ok[::16, ::16]].reshape(-1)]))
    atlas.save(os.path.join(out, "atlas.jpg"), quality=75)
    json.dump(dict(title=title, um=P[0]["um"], patches=pats, known=kp, known_label=klabel), open(os.path.join(out, "data.json"), "w"), separators=(",", ":"))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer", "index.html"), os.path.join(out, "index.html"))
    print(f"viewer: {len(pats)} patches, {sum(p['area'] for p in pats):.1f} cm2 -> {out}/index.html (serve the folder: python -m http.server -d {out})")
if __name__ == "__main__": main(sys.argv[1:])
