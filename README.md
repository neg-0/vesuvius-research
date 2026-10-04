# Vesuvius research (neg-0)

Independent experiments on the [Vesuvius Challenge](https://scrollprize.org)
open data: preregistered hypotheses, the code that ran, and the results,
including negative ones. Each folder has a `PROTOCOL.md` (written before the
data were scored), a `RESULT.md`, and machine-readable payloads.

| Folder | Question | Outcome |
|---|---|---|
| [`v013-pherc0139-depth/`](v013-pherc0139-depth/) | Can a label-free depth choice recover ink where PHerc0139's March-2026 2.4 µm meshes sit off the inked surface ([villa #1912](https://github.com/ScrollPrize/villa/issues/1912))? | Mixed. A finer depth grid lifts the label-free mean AUC from 0.803 to 0.817 and the label-oracle ceiling from 0.810 to 0.841, but two aligned segments get worse. Every segment's best offset is on the same side of the mesh. |
| [`v012-pherc0800-seating/`](v012-pherc0800-seating/) | Is PHerc.0800 segment `20251028225813` seated on the eligible 8.64 µm volume (sampled seating score)? | No. It scores 4.07 against a threshold of 15; the cloud port reproduces the earlier segment's 6.6608 exactly. |

## Provenance and credit

- Scroll data: Vesuvius Challenge open data bucket
  (`vesuvius-challenge-open-data`), read in place; no raw CT is redistributed here.
- Ink model: [`scrollprize/ink_canonical_2um`](https://huggingface.co/scrollprize/ink_canonical_2um)
  @ `075855bc`, run through
  [ScrollPrize/villa](https://github.com/ScrollPrize/villa) `ink-detection/optimized_inference`.
- V-013 builds on the benchmark and code of
  [TAUIL-Abd-Elilah/cross-scan-ink-transfer](https://github.com/TAUIL-Abd-Elilah/cross-scan-ink-transfer)
  @ `df9db2f` (MIT). That code is fetched at its pinned commit by
  `setup_local.sh`, not copied here; our only change to it is letting
  `infer_crop.py` run on CPU.
- V-012's seating score reimplements the public method of Herculaneum Scroll
  Tools at commit `c836a52`.

## Licensing

Our code in this repository is MIT licensed (see `LICENSE`). Derived data
(ink-probability maps, AUC tables) come from Vesuvius Challenge data and model
outputs and are shared under the Challenge's data terms, not under MIT. Linked
third-party code keeps its own licence.
