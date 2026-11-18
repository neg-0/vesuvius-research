"""Contact sheet of the top one-sided text-score patches of a screen. Usage: sheet_whole.py <run dir with shard-*> <n> <out.png>
Same candidate rule as judge_whole.py (stroke score >= 2% on one face, other face under a third); prints the table it draws."""
import sys, csv, glob, os, numpy as np
from PIL import Image
run, n, out = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3]; R = []
for f in sorted(glob.glob(run + "/shard-*/scores.tsv")):
    for r in csv.DictReader(open(f), delimiter="\t"):
        if r["status"] != "ok": continue
        a, b = float(r["stroke_a"]), float(r["stroke_b"]); hi, lo = max(a, b), min(a, b)
        if hi >= 0.02 and lo < hi / 3 and float(r["area_cm2"]) >= 0.25: R.append((hi, lo, "a" if a >= b else "b", os.path.dirname(f), int(r["id"]), r["seed_z"], r["seed_y"], r["seed_x"], float(r["area_cm2"])))
R.sort(reverse=True); ims = []
for h in R[:n]:
    p = f"{h[3]}/p{h[4]:04d}_{h[2]}.png"
    if not os.path.isfile(p): continue
    print("%.3f vs %.3f side %s %s p%04d z %s y %s x %s area %.2f" % (h[0], h[1], h[2], os.path.basename(h[3]), *h[4:])); ims.append(np.asarray(Image.open(p).convert("L").resize((384, 384))))
print(len(R), "candidates")
if ims:
    while len(ims) % 6: ims.append(np.zeros((384, 384), np.uint8))
    Image.fromarray(np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)])).save(out)
