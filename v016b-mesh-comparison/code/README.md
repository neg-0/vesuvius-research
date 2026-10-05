# V-016b code

`v016_cross_scan_ink.py` is the V-016 driver with three environment switches added:

- `V016_AMP`: `fp16` (default) or `bf16`.
- `V016_TAG`: a suffix for checkpoint folders, caches and payload names.
- `V016_SEED`: the seed (default 0).

It also has a `compare` subcommand that applies the V-016b rule. With no switches set, it behaves exactly like the
V-016 driver. Setup is in [../../v016-cross-scan-ink/code/README.md](../../v016-cross-scan-ink/code/README.md).
