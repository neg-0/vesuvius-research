# Vesuvius research (neg-0)

Independent experiments on the [Vesuvius Challenge](https://scrollprize.org)
open data: preregistered hypotheses, the code that ran, and the results,
including negative ones. Each folder has a `PROTOCOL.md` (written before the
data were scored), a `RESULT.md`, and machine-readable payloads.

| Folder | Question | Outcome |
|---|---|---|
| [`intensity-shift-robustness/`](intensity-shift-robustness/) | Do ink readers keep their accuracy on a labelled 9.362 µm segment when it takes the grey-level distribution of other scans of the same type? | No. On PHerc0500P2 `20250825181859`, Hecate d0 falls from 0.931 to 0.649–0.888 reverse AUC across six source scans, and Reader V3 from 0.896 to 0.793–0.900 (four of six drop by 0.03 or more). Matching the segment to itself changes neither by more than 0.0004. `match_intensity.py` ships the per-layer histogram match. Registered before scoring. |
| [`lasagna-normals-from-m7/`](lasagna-normals-from-m7/) | Can lasagna-format normals be built from the public m7 surface prediction alone, for the six eligible scrolls with no published lasagna normals? | Yes. On PHerc1218 they are 7.8° median (p90 26.1°) from the organisers' normals and cover 100% of the box against their 86%; `make_normals.py` writes the stores the lasagna fit reads. An automatic umbilicus finder did not work (4–18 mm off). |
| [`v031-canonical-vs-mrg-1um/`](v031-canonical-vs-mrg-1um/) | Does the stock 2.4 µm canonical ink model, zero-shot on 1.129 µm renders, match the organisers' `mrg20736-1um` prediction (AUC on identical ds8 pixels, committed windows)? | Not across scrolls. PHerc1667: canonical >= mrg on 76 of 107 windows (PASS). PHercParis4: 24 of 105 (mean 0.861 vs 0.920, FAIL). PHerc0139: 25 of 85 (0.801 vs 0.857, FAIL). Preregistered. |
| [`v032-depth-sweep/`](v032-depth-sweep/) | On PHercParis4, does a depth offset (oracle, -32 to +32 layers) explain the loss to `mrg20736-1um`? | Partly, per window only. Loss windows rise from mean AUC 0.723 to 0.836 (mrg 0.933); the gap closed on 2 of 12 (needed 6), so FAIL, but the loss set gains 0.100 more than controls. Segment- or scroll-level offsets do not carry over. Preregistered. |
| [`b001-real-scan-distillation/`](b001-real-scan-distillation/) | Does fine-tuning the organisers' 9 µm ink model on real 9.36 / 8.64 µm scans of other scrolls, with the 2.4 µm model's published prediction as the teacher, read unseen scrolls better? | Yes. Mean AUC 0.711 → 0.803 on 49 human-labelled windows of three held-out scrolls (44 of 49 better), replicated with a second seed (0.815, 46 of 49); largest on the 8.64 µm scan type. Real input beats simulated 9 µm input; six scrolls of data add nothing over three. Preregistered. |
| [`v016-cross-scan-ink/`](v016-cross-scan-ink/) | Can the 2.4 µm ink model learn to read PHerc1667's 1.129 µm scan from labels drawn on the 2.399 µm scan, using V-014's refit meshes? | Yes. Fine-tuned on four segments, it reads the held-out ones at AUC 0.744–0.941 (mean 0.864) against 0.564 zero-shot, 6 of 6 better, in 3 leave-a-pair-out folds. Preregistered, with one recorded amendment (a training-normalisation bug fixed before any held-out score counted). |
| [`v016b-mesh-comparison/`](v016b-mesh-comparison/) | With one fixed, numerically stable recipe, does a 1.129 µm ink model trained along V-014's refit PHerc1667 meshes beat one trained along the published meshes? | Yes, on 6 of 6 held-out segments in each of two seeds (mean +0.208 and +0.182 AUC; refit 0.859 and 0.858, published 0.651 and 0.676). This is ink-level evidence for #1843. Preregistered. |
| [`v018-cross-scan-placement/`](v018-cross-scan-placement/) | Are a segment's published surface volumes on different scans placed on the same papyrus? Audit of every multi-scan segment in the open-data bucket (231 segments, 290 scan pairs, CPU only). | PHerc1667 1.129 µm fails on all 25 pairs, reproducing #1843. Elsewhere the renders agree (median NCC 0.83–0.93) only after median in-plane shifts of 25–115 µm, part of which a mesh-index convention explains; six of seven low-match pairs on PHercParis4 and PHerc0814 turn out to be offset 250–390 µm, and the seventh stays unexplained. |
| [`v014-pherc1667-transform/`](v014-pherc1667-transform/) | Does the published PHerc1667 1.129 µm → 2.399 µm matrix put the 1.129 µm meshes on the papyrus ([villa #1843](https://github.com/ScrollPrize/villa/issues/1843))? | No. Its own landmarks reject it (51.4 vs 0.93 voxels RMS for a refit). On all 6 labelled segments the refit mesh's CT matches the 2.399 µm reference (NCC 0.59–0.80 on 5) and the published mesh's does not. Only 1 of the 22 published transforms has this problem. |
| [`v013-pherc0139-depth/`](v013-pherc0139-depth/) | Can a label-free depth choice recover ink where PHerc0139's March-2026 2.4 µm meshes sit off the inked surface ([villa #1912](https://github.com/ScrollPrize/villa/issues/1912))? | Mixed. A finer depth grid lifts the label-free mean AUC from 0.803 to 0.817 and the label-oracle ceiling from 0.810 to 0.841, but two aligned segments get worse. Every segment's best offset is on the same side of the mesh. |
| [`v012-pherc0800-seating/`](v012-pherc0800-seating/) | Is PHerc.0800 segment `20251028225813` seated on the eligible 8.64 µm volume (sampled seating score)? | No. It scores 4.07 against a threshold of 15; the cloud port reproduces the earlier segment's 6.6608 exactly. |

## Provenance and credit

- Scroll data: Vesuvius Challenge open data bucket
  (`vesuvius-challenge-open-data`), read in place; no raw CT is redistributed here.
- Ink model: [`scrollprize/ink_canonical_2um`](https://huggingface.co/scrollprize/ink_canonical_2um)
  @ `075855bc`, run through
  [ScrollPrize/villa](https://github.com/ScrollPrize/villa) `ink-detection/optimized_inference`.
- V-013 and V-014 build on the benchmark and code of
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
| [`v033-depth-chooser/`](v033-depth-chooser/) | On PHercParis4, can a label-free statistic of the ink map (std, top-5, mean) choose the depth offset per window, using V-032's nine-offset maps? | No. The primary chooser (std) scores -0.0398 on loss windows against an oracle gain of +0.1131 and -0.1775 on control windows; FAIL as registered. |
| [`d003-bucket-pyramid-audit/`](d003-bucket-pyramid-audit/) | Does the pyramid defect reported in villa #2107 (truncated depth at a coarse level) exist in the open-data bucket copy of the surface volumes? | No. 692 volumes audited, no pyramid defects (BUCKET_CLEAN); 5 sparse volumes had nothing to probe. |
| [`v041-8x-finetune/`](v041-8x-finetune/) | Does a fine-tune with 8x the data and 8x the steps beat the stock 2.4 µm ink model on 1.129 µm PHercParis4 scans, and does it carry to PHerc0139? | PHercParis4: yes, twice (77 and 78 of 105 windows, +0.050 and +0.048 mean AUC, AUC 0.911 vs 0.861); PHerc0139: no (V-049 +0.005, V-044 +0.001). Scroll-specific. |
