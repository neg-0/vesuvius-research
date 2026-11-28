"""Contact sheet of the top one-sided text-score patches of a screen. Usage: sheet_whole.py <run dir> <n> <out.png>
Candidates are judge_whole.py's (runs.candidates); the sheet also leaves out patches under 0.25 cm2, too small to read
by eye. Prints the table it draws. A folder with no scores.tsv (in it or in shard-*/) stops with an error."""
import os, sys, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import runs
run, n, out = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3]
if runs.not_runs([run]): sys.exit(f"no scores.tsv in {run} or its shard-*/ folders: not a scan run")
R = [c for c in runs.candidates([run])[1] if c["area"] >= 0.25]; ims = []
for c in R[:n]:
    p = f"{c['dir']}/p{c['id']:04d}_{c['side']}.png"
    if not os.path.isfile(p): continue
    print("%.3f vs %.3f side %s %s p%04d z %s y %s x %s area %.2f" % (c["hi"], c["lo"], c["side"], os.path.basename(c["dir"]), c["id"], *c["seed"], c["area"])); ims.append(np.asarray(Image.open(p).convert("L").resize((384, 384))))
print(len(R), "candidates")
if ims:
    while len(ims) % 6: ims.append(np.zeros((384, 384), np.uint8))
    Image.fromarray(np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)])).save(out)
