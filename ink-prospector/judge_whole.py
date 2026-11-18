"""Judge a whole-scroll screen by the rule below. Usage: judge_whole.py <runs/whole-SCROLL>
HIT candidate = patch with >= 2% confident one-sided ink (the stroke_a / stroke_b text score columns: confident ink in letter-sized structures; frac_a / frac_b is a looser haze-prone measure and is not used) and the other face under one third of that; the top 3 by that fraction count."""
import sys, glob, csv, collections
st = collections.Counter(); C = []
for f in sorted(glob.glob(sys.argv[1].rstrip("/") + "/shard-*/scores.tsv")):
    for r in csv.DictReader(open(f), delimiter="\t"):
        st[r["status"]] += 1
        if r["status"] != "ok": continue
        a, b = float(r["stroke_a"]), float(r["stroke_b"]); hi, lo = max(a, b), min(a, b)
        if hi >= 0.02 and lo < hi / 3: C.append((hi, lo, f.split("/")[-2], r["id"], r["seed_z"], r["seed_y"], r["seed_x"]))
C.sort(reverse=True); print("status", dict(st)); print("HITS" if C else "NOTHING at this spacing")
for c in C[:3]: print("  %.3f vs %.3f  %s p%04d  seed z %s y %s x %s" % (c[0], c[1], c[2], int(c[3]), c[4], c[5], c[6]))
