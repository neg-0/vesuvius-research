# V-033: can a label-free statistic choose the depth offset per window? FAIL

Spec: `experiments/v033/README.md`. Driver: `experiments/v033/v033_chooser.py`. Lines printed by `rule`:

- `RULE label-free depth chooser (V-033, primary: std of the ds8 ink map): mean gain on loss windows -0.0398 (need >= 0.5 x oracle +0.1131 = +0.0566); mean gain on control windows -0.1775 (need >= -0.01); chosen beats offset 0 on 4 of 12 loss windows (need >= 8) -> FAIL`
- `DESCR chooser std: gain L -0.0398, W -0.1775, D 0.0029; chosen beats 0 on 4 of 12 L windows; equals the oracle offset on 1 of 31; chosen offsets L [32, 24, -8, 16, 24, 24, -24, 32, 32, -32, -16, 0]`
- `DESCR chooser top5: gain L 0.0454, W -0.0117, D -0.0576; chosen beats 0 on 9 of 12 L windows; equals the oracle offset on 4 of 31; chosen offsets L [32, 32, -8, -24, 32, 8, -8, 16, 16, 8, 24, -8]`
- `DESCR chooser mean: gain L 0.0306, W 0.0017, D 0.0071; chosen beats 0 on 8 of 12 L windows; equals the oracle offset on 5 of 31; chosen offsets L [8, 32, -8, -8, 0, -8, -24, 8, -8, 8, 32, -8]`
- `DESCR oracle gain L 0.1131; random-offset gain L -0.0396, W -0.0911, D -0.0603`
- `DESCR loss set mean AUC: at 0 0.7227, oracle 0.8358, mrg20736-1um 0.9332; primary chooser gives 0.6829`
- `DESCR control set mean AUC: at 0 0.8966, oracle 0.9256, mrg20736-1um 0.8341; primary chooser gives 0.7191`
- `DESCR PHerc0139 w044/w045 set mean AUC: at 0 0.7503, oracle 0.8987, mrg20736-1um 0.8784; primary chooser gives 0.7532`

<!-- prose -->

## Notes (desktop, 2026-10-09 UTC)

- **Verdict: FAIL.** The registered chooser (the offset with the highest std of the ds8 ink map) does not find the
  depth. On the 12 PHercParis4 loss windows it changes the mean AUC by −0.040 (+0.057 was needed, half the oracle's
  +0.113), it beats offset 0 on 4 of 12 (8 needed), and it costs the controls −0.178 (at most −0.01 allowed). That is
  no better than a random offset on the loss set (−0.040) and worse than random on the controls (−0.091).
- **The two descriptive choosers do better but were not the registered one.** Top-5% mean: +0.045 on the loss set,
  beats offset 0 on 9 of 12, −0.012 on controls. Mean: +0.031, 8 of 12, +0.002 on controls. Neither reaches +0.057, and
  they match the oracle offset on only 4 and 5 of 31 windows.
- **Inputs:** all 279 ink maps and 31 carried masks from V-032 were present (`CHECK OK`); nothing was recomputed and no
  GPU was used. `stats.json` was committed (`V-033 stats: chosen offsets`) before `rule` ran.
- **Runs (V-033):** `20261009T182458-10cc637d47` (stats), `20261009T182501-5013bf22a8` (rule).
