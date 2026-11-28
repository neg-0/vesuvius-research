"""Judge a whole-scroll screen by the rule below. Usage: judge_whole.py <runs/whole-SCROLL>
HIT candidate = patch with >= 2% confident one-sided ink (the stroke_a / stroke_b text score columns: confident ink in letter-sized structures; frac_a / frac_b is a looser haze-prone measure and is not used) and the other face under one third of that; the top 3 by that fraction count.
The rule lives in runs.candidates. A folder with no scores.tsv (in it or in shard-*/) stops with an error instead of reading as NOTHING."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import runs
run = sys.argv[1].rstrip("/")
if runs.not_runs([run]): sys.exit(f"no scores.tsv in {run} or its shard-*/ folders: not a scan run")
st, C = runs.candidates([run]); print("status", dict(st)); print("HITS" if C else "NOTHING at this spacing")
for c in C[:3]: print("  %.3f vs %.3f  %s p%04d  seed z %s y %s x %s" % (c["hi"], c["lo"], os.path.basename(c["dir"]), c["id"], *c["seed"]))
