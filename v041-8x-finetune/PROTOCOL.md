> Public copy of the V-041 registration. Paths such as `experiments/v036/README.md` and `results/...` refer to the private research tree and are kept as written; the code and results are in `code/` and `results/` next to this file.

# V-041: does V-034's fine-tune beat the stock model at 8x the data and 8x the steps? (RunPod)

Registered 2026-10-10 (cloud), after Dustin chose "RunPod larger study" on the decision card (up to $30) and before any V-041
plan, render, training or score exists. Public-safe. Driver: `v041_run.py` (selftest passes), which imports `v039_run.py`, which
imports V-034's `v034_run.py`, all unchanged. Output: `results/v041-p4-scale2/`.

## Run condition (pre-registered, evaluated before any pod exists)

V-041 is only worth its money if V-039 shows a trend. It runs iff V-039 is FAIL (or UNKNOWN is resolved to FAIL by a complete
rerun) **and** V-039's mean gain over zero-shot exceeds V-034's (-0.0131) by at least +0.015, i.e. is at least +0.0019. Otherwise
it is cancelled, nothing is rented, and the form says scale did not help. If V-039 PASSes, V-041 is cancelled and a PHerc0139
replication of V-039 is registered instead. This thread flips the queue tag from `[blocked]` to `[ready]` after reading V-039's
`eval_rows.json` and `compare.json` (not `summary.json`, which a driver defect emptied once; fixed 2026-10-10); Dustin has a standing policy that anything in the queue is approved (2026-10-10), so no per-item ask is made (RUNPOD-LEDGER rule 4); the caps below still bind.

## Design

V-039 with two numbers raised; everything else is V-034's:

- up to **24** densest training patches per training segment per fold (V-039: 9, V-034: 3), same selection rule and 3D guard of
  1014 voxels from every held-out window;
- **16000** iterations (V-039: 6000, V-034: 2000), `finetune_on_scan.py` at 7a5ba5a, seed 0, bf16, `--voxel-um 2.258`;
- same folds, held-out segments, V-031 windows, REPRO re-scores, and rule: PASS iff fine-tuned > zero-shot on at least
  ceil(0.7 n) windows and mean gain at least +0.02; FAIL otherwise; INVALID / UNKNOWN as V-034;
- the plan is written fresh on the pod (a new patch set); **plan stop condition:** fewer than 60 training patches in any fold
  (`PLAN_CHECK FAIL`) stops the item after `plan`; the FOLD lines are reported and nothing is rendered or trained;
- fine-tuned maps are written as `<window>_sc2_ft_ink.npy`, separate from V-034's and V-039's.

`compare` prints, descriptively, V-041's mean gain next to V-034's and V-039's on the windows in all three, split by stock AUC
below / at least 0.75.

## What each outcome means

- PASS: data and steps rescue fine-tuning; replicate on PHerc0139 before any claim (new registration, pod again).
- FAIL with a gain higher than V-039's: still rising; report the dose-response and stop, or step up once more if the budget allows.
- FAIL with a gain no higher than V-039's: scale is not the lever; the form says cross-segment fine-tuning does not beat the stock model.

## Budget and caps

Estimate on one 24 GB pod (V-036's RTX 3090 at $0.50/h): carry 0.75 h, plan 0.2 h, render about 100 patches 2.5 h, train 3 folds of
16000 iterations about 3 h each = 9 h, eval 105 windows 2.5 h; about 15 h, **about $7.5 to $9**. **Hard caps: 24 wall-clock
hours and $20 total (GPU plus disk)**, whichever first; at 80% of either the pod finishes the current stage, copies back and stops.
A stage cut off by a cap leaves the rule UNKNOWN (the pod volume is deleted); a stage that prints `BUDGET reached` is rerun with the
same line. Ledger: add the cap as an open item before the pod exists, a row when it ends.

## RunPod run

Exactly V-036's "RunPod run" section (`experiments/v036/README.md`: approval gate, pod type and price ceiling at most $1.00/h, storage,
what goes to the pod and what never does, environment, acceptance gate, finish), with these differences: the tarball also holds
`experiments/{v039,v041}` and `results/v034-p4-finetune` and `results/v037-p4-seed1` are not needed; the stages are the commands below;
the pod **stops itself (not deletes) when the last stage writes its done marker** (RUNPOD-LEDGER rule 6) and the worker copies back and
then deletes the pod and volume; all `R*` wrappers use `--hypothesis V-041` and `v041_run.py`.

```sh
export V041_RESULTS=$PWD/projects/vesuvius/results/v041-p4-scale2 V039_RESULTS=$PWD/projects/vesuvius/results/v039-p4-scale \
  V034_RESULTS=$PWD/projects/vesuvius/results/v034-p4-finetune
R() { python3 tools/research.py run vesuvius --hypothesis V-041 --seed 0 --timeout 6600 \
  --cwd projects/vesuvius/experiments/v041 --input v041_run.py --input ../v039/v039_run.py $IN -- $P v041_run.py "$@"; }
```

Stages: `check` (`CHECK OK`); `R31 carry PHercParis4`; `R plan` (four `FOLD` lines; `PLAN_CHECK OK` needs at least 60 patches per fold); in the
throwaway repo `git add .../v041-p4-scale2/plan.json && git commit -m "V-041 plan"` and copy plan.json to the desktop; `R prep`;
`R train 0; R train 1; R train 2` (skipped_frac at most 0.01); `R eval 0; R eval 1; R eval 2` (`REPRO FAIL` means stop: copy back, stop the
pod, report); `R rule; R compare; R status`. Copy back `plan.json`, `train_log_fold*.json`, `summary.json`, `RESULT.md`, `eval_rows.json`,
`compare.json`, `timing.json`; commit on the desktop with GPU type, $/h, hours and dollars in the commit message. Post nothing elsewhere.

## Run condition result (2026-10-10)

V-039: FAIL on the window count (58 of 105 windows up, need 74); mean gain over zero-shot +0.0205 (V-034 -0.0131, V-037 +0.0009); stock below 0.75 (15 windows) +0.1671, stock at least 0.75 -0.0040 (V-034: +0.1415 / -0.0389). The mean gain exceeds V-034's by +0.034, so the run condition is **met**. The loss on strong windows nearly vanished at 3x data and steps; V-041 asks whether it turns positive at 8x.
