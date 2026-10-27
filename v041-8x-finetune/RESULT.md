# V-041 / V-047 / V-044 / V-049: does a bigger fine-tune beat the stock 2.4 µm ink model on 1.129 µm scans?

All four were registered before they ran (rule: fine-tuned beats zero-shot on at least 70% of held-out windows AND mean gain at least +0.02 AUC). Fine-tune = villa's `finetune_on_scan.py` at `neg-0/villa@7a5ba5a`, labels carried over from the 2.399 µm scan, 3 folds with held-out segments, bf16, `--voxel-um 2.258`. Windows are the V-031 windows scored against the organisers' published ink labels.

| Run | Scroll | Recipe | Held-out windows | Fine-tuned > stock | Mean AUC stock / fine-tuned | Mean gain | Verdict |
|---|---|---|---|---|---|---|---|
| V-041 | PHercParis4 | 24 patches/segment, 16,000 it, seed 0 | 105 | 77 (need 74) | 0.8607 / 0.9111 | +0.0503 | PASS |
| V-047 | PHercParis4 | same, seed 1 (replication) | 105 | 78 (need 74) | 0.8607 / 0.9085 | +0.0478 | PASS |
| V-049 | PHerc0139 | same recipe | 85 | 44 (need 60) | 0.8011 / 0.8064 | +0.0053 | FAIL |
| V-044 | PHerc0139 | 3-member ensemble (stock + two fine-tuned seeds) | 85 | 41 (need 60) | 0.8011 / 0.8019 | +0.0008 | FAIL |

For reference, the organisers' own 1 µm prediction (mrg20736-1um) scores 0.9202 on the PHercParis4 windows. On PHercParis4 the gain is concentrated where the stock model is weak (stock AUC < 0.75: +0.197 on 15 windows) and small where it is strong (+0.026 on 90). On PHerc0139 the same recipe helps weak windows (+0.134 on 23) and hurts strong ones (-0.043 on 62), so the PHercParis4 gain is scroll-specific; we claim nothing beyond it. Smaller recipes (3x and 6000 iterations, V-034/V-039) did not pass the same rule.

## Layout
- `PROTOCOL.md`: V-041's registration (the design, run condition and stop rules; V-047, V-049 and V-044 are the same driver with a seed, scroll or ensemble change, and their rules are in the code header of each `code/vNNN/vNNN_run.py`).
- `results/<run>/`: `RESULT.md` (printed rule lines, per-window table), `summary.json`, `compare.json`, `eval_rows.json`, `plan.json`, per-fold training logs. Note: the V-041 and V-047 `RESULT.md` files keep the `V-034` heading of the template they were generated from.
- `code/`: the drivers as run, with the `..` import layout they expect. They need a `neg-0/villa` checkout at 7a5ba5a (`V031_VILLA`), the canonical ink checkpoint (`CANON_CKPT`) and the open-data bucket; see the header of `code/v031/v031_run.py` and `code/v034/v034_run.py` for the environment variables. Run `python3 -I vNNN_run.py selftest` from a driver's folder only after `V031_VILLA` is set (the import of villa's scan-adaptation module needs it).

Hardware: one rented 24 GB GPU per run (about $10 for V-041). Caveats: one scroll passes and one fails; the stock model differs per scroll; no claim is made that any model reads text.
