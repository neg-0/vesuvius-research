# B-001 results

All three studies ran on one rented RTX 4090 (24 GB) each, 2026-10-11 UTC, about $5.40 of GPU time in total
(B-001 $0.94, B-001r $1.82, B-001s about $2.61). Software: villa `e0bbb8b`, torch 2.12.0+cu130, numpy 2.5.3,
zarr 3.4.0, tifffile 2026.9.20. The base checkpoint's sha256 is checked by the drivers. No checkpoint selection;
each model is the final step of its one fixed recipe.

Fine-tuned checkpoints (sha256 prefixes): B-001 `1674820d`, B-001r R1 `92aed6dd`, B-001r SIM `4d54b986`,
B-001s folds PHerc0841 `8b0ca81d`, PHerc0009B `c7e19b0e`, PHerc0500P2 `5821f8e3`. They are not in this repository
(273 MB each) and are available on request.

Training notes: B-001 trained all 34,546,498 parameters for 6,000 steps of batch 16, final loss (last 100 steps)
0.5152. R1 0.5186, SIM 0.4976. Reproduction checks: the stock model's forward AUCs re-read for B-001r and B-001s match
B-001's committed values on all 49 windows (max difference 0.00000).

## B-001: real-scan distillation for the eligible ~9 µm scan types: PASS

Spec: `experiments/b001/README.md`. Lines printed by `b001_read.py rule`:

- `RULE mean paired gain +0.0914 (need >= +0.05); REAL > S0 on 44 of 49 windows (need 70%); n 49 (need 20)`
- `RULE PHerc0009B: 12 windows, mean gain +0.1044`
- `RULE PHerc0500P2: 12 windows, mean gain +0.1167`
- `RULE PHerc0841: 25 windows, mean gain +0.0730`
- `CONTROL REAL shuffled-layer pooled AUC 0.5613 (need 0.5 +- 0.1)`
- `VERDICT PASS`

### Mean AUC over scored windows (descriptive)

| map | mean AUC |
| --- | --- |
| S0_fwd | 0.7113 |
| REAL_fwd | 0.8027 |
| S0_rev | 0.5073 |
| REAL_rev | 0.515 |
| S0_max | 0.6732 |
| REAL_max | 0.7739 |
| S0_fwd_gridindex_labels | 0.7086 |

### Per window

| window | scroll | S0 fwd | REAL fwd | gain | S0 fwd, grid-index labels | REAL shuffled | pos / neg px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PHerc0009B_20250919125754_w4680_1820 | PHerc0009B | 0.5494 | 0.8026 | +0.2531 | 0.4837 | 0.6474 | 2698 / 113025 |
| PHerc0009B_20250919125754_w5200_1820 | PHerc0009B | 0.6258 | 0.8203 | +0.1946 | 0.6252 | 0.5520 | 75987 / 189412 |
| PHerc0009B_20250919125754_w5200_2340 | PHerc0009B | 0.7850 | 0.8253 | +0.0404 | 0.7462 | 0.5780 | 80908 / 189492 |
| PHerc0009B_20250919125754_w5200_3120 | PHerc0009B | 0.8056 | 0.9439 | +0.1383 | 0.7995 | 0.5139 | 106290 / 164110 |
| PHerc0009B_20250919125754_w5200_3900 | PHerc0009B | 0.8164 | 0.8824 | +0.0660 | 0.7818 | 0.6881 | 97479 / 172921 |
| PHerc0009B_20250919125754_w5200_4420 | PHerc0009B | 0.7904 | 0.7343 | -0.0561 | 0.7860 | 0.5791 | 19757 / 27058 |
| PHerc0009B_20250919125754_w5720_1560 | PHerc0009B | 0.7522 | 0.8605 | +0.1083 | 0.7675 | 0.5157 | 31244 / 82605 |
| PHerc0009B_20250919125754_w5720_4160 | PHerc0009B | 0.4398 | 0.5581 | +0.1183 | 0.4460 | 0.8291 | 613 / 42221 |
| PHerc0009B_20250919125754_w5980_2080 | PHerc0009B | 0.7222 | 0.8238 | +0.1016 | 0.7408 | 0.5802 | 71883 / 179234 |
| PHerc0009B_20250919125754_w5980_2600 | PHerc0009B | 0.7341 | 0.9030 | +0.1689 | 0.7482 | 0.6141 | 67634 / 189412 |
| PHerc0009B_20250919125754_w5980_3120 | PHerc0009B | 0.8432 | 0.9435 | +0.1003 | 0.8409 | 0.6409 | 75919 / 194294 |
| PHerc0009B_20250919125754_w5980_3640 | PHerc0009B | 0.9545 | 0.9733 | +0.0189 | 0.9331 | 0.6639 | 19197 / 42941 |
| PHerc0500P2_20250825181859_w1820_1040 | PHerc0500P2 | 0.7178 | 0.8088 | +0.0910 | 0.7145 | 0.4564 | 36471 / 109202 |
| PHerc0500P2_20250825181859_w2600_1300 | PHerc0500P2 | 0.7773 | 0.8521 | +0.0748 | 0.7764 | 0.6642 | 43363 / 185789 |
| PHerc0500P2_20250825181859_w2600_780 | PHerc0500P2 | 0.7310 | 0.7651 | +0.0341 | 0.7295 | 0.4247 | 75574 / 168707 |
| PHerc0500P2_20250825181859_w2860_2080 | PHerc0500P2 | 0.6125 | 0.7740 | +0.1615 | 0.6043 | 0.6120 | 35714 / 127552 |
| PHerc0500P2_20250825181859_w2860_260 | PHerc0500P2 | 0.6744 | 0.9010 | +0.2266 | 0.6583 | 0.5243 | 16582 / 44785 |
| PHerc0500P2_20250825181859_w3120_1300 | PHerc0500P2 | 0.6820 | 0.7769 | +0.0949 | 0.6738 | 0.4840 | 57109 / 171545 |
| PHerc0500P2_20250825181859_w3380_780 | PHerc0500P2 | 0.7500 | 0.8798 | +0.1298 | 0.7483 | 0.5903 | 54358 / 134317 |
| PHerc0500P2_20250825181859_w3900_780 | PHerc0500P2 | 0.6565 | 0.8910 | +0.2346 | 0.6767 | 0.6442 | 42882 / 147563 |
| PHerc0500P2_20250825181859_w4420_0 | PHerc0500P2 | 0.7323 | 0.7219 | -0.0103 | 0.7241 | 0.4255 | 33280 / 90999 |
| PHerc0500P2_20250825181859_w4420_1560 | PHerc0500P2 | 0.7805 | 0.8888 | +0.1083 | 0.7705 | 0.6566 | 70988 / 168734 |
| PHerc0500P2_20250825181859_w4420_2080 | PHerc0500P2 | 0.8826 | 0.9781 | +0.0955 | 0.8334 | 0.8054 | 11273 / 20684 |
| PHerc0500P2_20250825181859_w4680_1040 | PHerc0500P2 | 0.7352 | 0.8953 | +0.1601 | 0.7225 | 0.5332 | 50371 / 169236 |
| PHerc0841_20260220213127_w1560_2860 | PHerc0841 | 0.6710 | 0.8262 | +0.1552 | 0.6513 | 0.5065 | 36777 / 171818 |
| PHerc0841_20260220213127_w1560_3380 | PHerc0841 | 0.7304 | 0.8576 | +0.1272 | 0.7224 | 0.4571 | 59361 / 179998 |
| PHerc0841_20260220213127_w2080_2600 | PHerc0841 | 0.7784 | 0.7952 | +0.0168 | 0.8196 | 0.6511 | 23934 / 64068 |
| PHerc0841_20260220213127_w2080_3120 | PHerc0841 | 0.6975 | 0.8331 | +0.1356 | 0.6884 | 0.5641 | 65012 / 205378 |
| PHerc0841_20260220213127_w2080_3640 | PHerc0841 | 0.4471 | 0.6933 | +0.2461 | 0.3983 | 0.2566 | 1593 / 35746 |
| PHerc0841_20260220213127_w2600_2600 | PHerc0841 | 0.7230 | 0.8300 | +0.1070 | 0.7359 | 0.5626 | 59204 / 156613 |
| PHerc0841_20260220213127_w2860_3120 | PHerc0841 | 0.7737 | 0.7985 | +0.0248 | 0.7905 | 0.7518 | 33298 / 93396 |
| PHerc0841_20260220213127_w3120_2600 | PHerc0841 | 0.9427 | 0.9592 | +0.0165 | 0.9292 | 0.5895 | 7565 / 75427 |
| PHerc0841_20260220213127_w520_3380 | PHerc0841 | 0.8352 | 0.9061 | +0.0708 | 0.8108 | 0.3823 | 64081 / 92880 |
| PHerc0841_20260220214732_w2600_1560 | PHerc0841 | 0.6830 | 0.8571 | +0.1741 | 0.6806 | 0.5355 | 77962 / 164377 |
| PHerc0841_20260220214732_w2600_2080 | PHerc0841 | 0.6491 | 0.7918 | +0.1427 | 0.6038 | 0.7224 | 19223 / 64225 |
| PHerc0841_20260220214732_w2600_2860 | PHerc0841 | 0.8557 | 0.9346 | +0.0790 | 0.8324 | 0.5885 | 53313 / 84828 |
| PHerc0841_20260220214732_w780_1560 | PHerc0841 | 0.5206 | 0.6109 | +0.0903 | 0.5088 | 0.7988 | 18939 / 33561 |
| PHerc0841_20260220214732_w780_2080 | PHerc0841 | 0.8725 | 0.9028 | +0.0303 | 0.8735 | 0.6292 | 78343 / 164373 |
| PHerc0841_20260220214732_w780_2600 | PHerc0841 | 0.6383 | 0.6397 | +0.0014 | 0.6508 | 0.6108 | 69571 / 138740 |
| PHerc0841_20260221022814_w0_2340 | PHerc0841 | 0.2878 | 0.2938 | +0.0060 | 0.3469 | 0.8768 | 6846 / 26627 |
| PHerc0841_20260221022814_w1040_1560 | PHerc0841 | 0.7121 | 0.8497 | +0.1376 | 0.7227 | 0.5496 | 16715 / 17643 |
| PHerc0841_20260221022814_w1040_2080 | PHerc0841 | 0.7678 | 0.7848 | +0.0170 | 0.7595 | 0.4624 | 62035 / 170039 |
| PHerc0841_20260221022814_w1300_2600 | PHerc0841 | 0.7713 | 0.8714 | +0.1001 | 0.8062 | 0.5832 | 68936 / 191223 |
| PHerc0841_20260221022814_w1300_3120 | PHerc0841 | 0.9183 | 0.9056 | -0.0127 | 0.8872 | 0.4094 | 13375 / 29100 |
| PHerc0841_20260221022814_w1560_780 | PHerc0841 | 0.2205 | 0.6303 | +0.4098 | 0.2958 | 0.5058 | 1722 / 4868 |
| PHerc0841_20260221022814_w1820_1300 | PHerc0841 | 0.7464 | 0.7880 | +0.0416 | 0.7227 | 0.6124 | 53781 / 137736 |
| PHerc0841_20260221022814_w520_1820 | PHerc0841 | 0.4641 | 0.1680 | -0.2961 | 0.5822 | 0.9206 | 3479 / 6823 |
| PHerc0841_20260221022814_w520_2340 | PHerc0841 | 0.7087 | 0.7757 | +0.0670 | 0.6981 | 0.6028 | 58220 / 152138 |
| PHerc0841_20260221022814_w780_2860 | PHerc0841 | 0.8883 | 0.8263 | -0.0620 | 0.8720 | 0.6293 | 1647 / 48965 |

## B-001r: seed replication PASS, simulated-input ablation REAL_INPUT_MATTERS

Spec: `experiments/b001r/README.md`. Lines printed by `b001r_read.py rule`:

- `REPRO S0 forward AUC vs B-001 committed: max abs diff 0.00000 on 49 windows (tol 0.005) -> OK`
- `R1 RULE mean paired gain +0.1035 (need >= +0.05); REAL > S0 on 46 of 49 windows (need 70%); n 49 (need 20)`
- `R1 RULE PHerc0009B: 12 windows, mean gain +0.1045`
- `R1 RULE PHerc0500P2: 12 windows, mean gain +0.1294`
- `R1 RULE PHerc0841: 25 windows, mean gain +0.0906`
- `CONTROL R1 shuffled-layer pooled AUC 0.5608 (need 0.5 +- 0.1)`
- `CONTROL SIM shuffled-layer pooled AUC 0.5496 (need 0.5 +- 0.1)`
- `R1_VERDICT PASS -> B-001 REPLICATED`
- `SIM REAL - SIM mean paired gain +0.0348; REAL > SIM on 39 of 49, SIM > REAL on 10 of 49 (margin 0.02, fraction 60%)`
- `SIM_OUTCOME REAL_INPUT_MATTERS`

### Descriptive

```json
{
 "B001_S0_fwd": 0.7113,
 "B001_REAL_fwd": 0.8027,
 "S0_fwd": 0.7113,
 "R1_fwd": 0.8148,
 "R1_rev": 0.52,
 "SIM_fwd": 0.768,
 "SIM_rev": 0.5315,
 "gain_correlation_REAL_R1": 0.9448,
 "SIM_vs_S0_B001_rule": [
  "RULE mean paired gain +0.0566 (need >= +0.05); REAL > S0 on 35 of 49 windows (need 70%); n 49 (need 20)",
  "RULE PHerc0009B: 12 windows, mean gain +0.0662",
  "RULE PHerc0500P2: 12 windows, mean gain +0.0679",
  "RULE PHerc0841: 25 windows, mean gain +0.0466"
 ]
}
```

### Per window (forward AUC)

| window | scroll | S0 (B-001) | REAL (B-001) | R1 | SIM | S0 re-read |
| --- | --- | --- | --- | --- | --- | --- |
| PHerc0009B_20250919125754_w4680_1820 | PHerc0009B | 0.5494 | 0.8026 | 0.8029 | 0.8095 | 0.5494 |
| PHerc0009B_20250919125754_w5200_1820 | PHerc0009B | 0.6258 | 0.8203 | 0.8022 | 0.7502 | 0.6258 |
| PHerc0009B_20250919125754_w5200_2340 | PHerc0009B | 0.7850 | 0.8253 | 0.7968 | 0.7379 | 0.7850 |
| PHerc0009B_20250919125754_w5200_3120 | PHerc0009B | 0.8056 | 0.9439 | 0.9467 | 0.8415 | 0.8056 |
| PHerc0009B_20250919125754_w5200_3900 | PHerc0009B | 0.8164 | 0.8824 | 0.8649 | 0.8037 | 0.8164 |
| PHerc0009B_20250919125754_w5200_4420 | PHerc0009B | 0.7904 | 0.7343 | 0.7094 | 0.6913 | 0.7904 |
| PHerc0009B_20250919125754_w5720_1560 | PHerc0009B | 0.7522 | 0.8605 | 0.8720 | 0.7514 | 0.7522 |
| PHerc0009B_20250919125754_w5720_4160 | PHerc0009B | 0.4398 | 0.5581 | 0.5906 | 0.7840 | 0.4398 |
| PHerc0009B_20250919125754_w5980_2080 | PHerc0009B | 0.7222 | 0.8238 | 0.8427 | 0.7678 | 0.7222 |
| PHerc0009B_20250919125754_w5980_2600 | PHerc0009B | 0.7341 | 0.9030 | 0.9114 | 0.8370 | 0.7341 |
| PHerc0009B_20250919125754_w5980_3120 | PHerc0009B | 0.8432 | 0.9435 | 0.9487 | 0.8751 | 0.8432 |
| PHerc0009B_20250919125754_w5980_3640 | PHerc0009B | 0.9545 | 0.9733 | 0.9838 | 0.9632 | 0.9545 |
| PHerc0500P2_20250825181859_w1820_1040 | PHerc0500P2 | 0.7178 | 0.8088 | 0.8226 | 0.7244 | 0.7178 |
| PHerc0500P2_20250825181859_w2600_1300 | PHerc0500P2 | 0.7773 | 0.8521 | 0.8401 | 0.7880 | 0.7773 |
| PHerc0500P2_20250825181859_w2600_780 | PHerc0500P2 | 0.7310 | 0.7651 | 0.7897 | 0.7123 | 0.7310 |
| PHerc0500P2_20250825181859_w2860_2080 | PHerc0500P2 | 0.6125 | 0.7740 | 0.8032 | 0.7521 | 0.6125 |
| PHerc0500P2_20250825181859_w2860_260 | PHerc0500P2 | 0.6744 | 0.9010 | 0.9268 | 0.8651 | 0.6744 |
| PHerc0500P2_20250825181859_w3120_1300 | PHerc0500P2 | 0.6820 | 0.7769 | 0.7891 | 0.7875 | 0.6820 |
| PHerc0500P2_20250825181859_w3380_780 | PHerc0500P2 | 0.7500 | 0.8798 | 0.8522 | 0.8830 | 0.7500 |
| PHerc0500P2_20250825181859_w3900_780 | PHerc0500P2 | 0.6565 | 0.8910 | 0.8987 | 0.7442 | 0.6565 |
| PHerc0500P2_20250825181859_w4420_0 | PHerc0500P2 | 0.7323 | 0.7219 | 0.7868 | 0.7129 | 0.7323 |
| PHerc0500P2_20250825181859_w4420_1560 | PHerc0500P2 | 0.7805 | 0.8888 | 0.8898 | 0.8086 | 0.7805 |
| PHerc0500P2_20250825181859_w4420_2080 | PHerc0500P2 | 0.8826 | 0.9781 | 0.9763 | 0.9752 | 0.8826 |
| PHerc0500P2_20250825181859_w4680_1040 | PHerc0500P2 | 0.7352 | 0.8953 | 0.9097 | 0.7939 | 0.7352 |
| PHerc0841_20260220213127_w1560_2860 | PHerc0841 | 0.6710 | 0.8262 | 0.8102 | 0.7349 | 0.6710 |
| PHerc0841_20260220213127_w1560_3380 | PHerc0841 | 0.7304 | 0.8576 | 0.8735 | 0.7538 | 0.7304 |
| PHerc0841_20260220213127_w2080_2600 | PHerc0841 | 0.7784 | 0.7952 | 0.7868 | 0.6488 | 0.7784 |
| PHerc0841_20260220213127_w2080_3120 | PHerc0841 | 0.6975 | 0.8331 | 0.8510 | 0.7557 | 0.6975 |
| PHerc0841_20260220213127_w2080_3640 | PHerc0841 | 0.4471 | 0.6933 | 0.7498 | 0.6084 | 0.4471 |
| PHerc0841_20260220213127_w2600_2600 | PHerc0841 | 0.7230 | 0.8300 | 0.8359 | 0.8084 | 0.7230 |
| PHerc0841_20260220213127_w2860_3120 | PHerc0841 | 0.7737 | 0.7985 | 0.8571 | 0.7367 | 0.7737 |
| PHerc0841_20260220213127_w3120_2600 | PHerc0841 | 0.9427 | 0.9592 | 0.9592 | 0.9858 | 0.9427 |
| PHerc0841_20260220213127_w520_3380 | PHerc0841 | 0.8352 | 0.9061 | 0.9198 | 0.9132 | 0.8352 |
| PHerc0841_20260220214732_w2600_1560 | PHerc0841 | 0.6830 | 0.8571 | 0.8625 | 0.7867 | 0.6830 |
| PHerc0841_20260220214732_w2600_2080 | PHerc0841 | 0.6491 | 0.7918 | 0.7964 | 0.6977 | 0.6491 |
| PHerc0841_20260220214732_w2600_2860 | PHerc0841 | 0.8557 | 0.9346 | 0.9274 | 0.8257 | 0.8557 |
| PHerc0841_20260220214732_w780_1560 | PHerc0841 | 0.5206 | 0.6109 | 0.5351 | 0.5115 | 0.5206 |
| PHerc0841_20260220214732_w780_2080 | PHerc0841 | 0.8725 | 0.9028 | 0.9050 | 0.8888 | 0.8725 |
| PHerc0841_20260220214732_w780_2600 | PHerc0841 | 0.6383 | 0.6397 | 0.6534 | 0.5754 | 0.6383 |
| PHerc0841_20260221022814_w0_2340 | PHerc0841 | 0.2878 | 0.2938 | 0.3697 | 0.4120 | 0.2878 |
| PHerc0841_20260221022814_w1040_1560 | PHerc0841 | 0.7121 | 0.8497 | 0.8230 | 0.8521 | 0.7121 |
| PHerc0841_20260221022814_w1040_2080 | PHerc0841 | 0.7678 | 0.7848 | 0.7924 | 0.7521 | 0.7678 |
| PHerc0841_20260221022814_w1300_2600 | PHerc0841 | 0.7713 | 0.8714 | 0.8611 | 0.8117 | 0.7713 |
| PHerc0841_20260221022814_w1300_3120 | PHerc0841 | 0.9183 | 0.9056 | 0.9096 | 0.8714 | 0.9183 |
| PHerc0841_20260221022814_w1560_780 | PHerc0841 | 0.2205 | 0.6303 | 0.7932 | 0.5898 | 0.2205 |
| PHerc0841_20260221022814_w1820_1300 | PHerc0841 | 0.7464 | 0.7880 | 0.7818 | 0.6496 | 0.7464 |
| PHerc0841_20260221022814_w520_1820 | PHerc0841 | 0.4641 | 0.1680 | 0.2582 | 0.5704 | 0.4641 |
| PHerc0841_20260221022814_w520_2340 | PHerc0841 | 0.7087 | 0.7757 | 0.7657 | 0.7580 | 0.7087 |
| PHerc0841_20260221022814_w780_2860 | PHerc0841 | 0.8883 | 0.8263 | 0.8898 | 0.9709 | 0.8883 |

## B-001s: scaled, leave-one-scroll-out real-scan distillation: NO_CLEAR_DIFFERENCE (vs stock: PASS)

Spec: `experiments/b001s/README.md`. Lines printed by `b001s_read.py rule`:

- `REPRO S0 forward AUC vs B-001 committed: max abs diff 0.00000 on 49 windows (tol 0.005) -> OK`
- `VS_S0 RULE mean paired gain +0.0987 (need >= +0.05); REAL > S0 on 45 of 49 windows (need 70%); n 49 (need 20)`
- `VS_S0 RULE PHerc0009B: 12 windows, mean gain +0.1236`
- `VS_S0 RULE PHerc0500P2: 12 windows, mean gain +0.1264`
- `VS_S0 RULE PHerc0841: 25 windows, mean gain +0.0734`
- `CONTROL fold PHerc0841 shuffled-layer pooled AUC 0.5629 (need 0.5 +- 0.1)`
- `CONTROL fold PHerc0009B shuffled-layer pooled AUC 0.5766 (need 0.5 +- 0.1)`
- `CONTROL fold PHerc0500P2 shuffled-layer pooled AUC 0.5369 (need 0.5 +- 0.1)`
- `VS_S0_VERDICT PASS`
- `SCALE B-001s - B-001 (two-seed mean) mean paired gain +0.0012; higher on 31 of 49, lower on 18 of 49 (margin 0.02, fraction 60%)`
- `SCALE_OUTCOME NO_CLEAR_DIFFERENCE`
- `SCALE PHerc0841: 25 windows, mean B-001s 0.7656 vs B-001 two-seed 0.7739 (-0.0084)`
- `SCALE PHerc0009B: 12 windows, mean B-001s 0.8585 vs B-001 two-seed 0.8393 (+0.0192)`
- `SCALE PHerc0500P2: 12 windows, mean B-001s 0.8540 vs B-001 two-seed 0.8507 (+0.0033)`
- `VERDICT NO_CLEAR_DIFFERENCE (vs stock: PASS)`

### Descriptive mean AUC

```json
{
 "B001_S0_fwd": 0.7113,
 "B001_REAL_fwd": 0.8027,
 "B001_R1_fwd": 0.8148,
 "S_fwd": 0.81,
 "S_rev": 0.5284,
 "S0_fwd": 0.7113
}
```

### Per window (forward AUC)

| window | scroll | S0 | B-001 REAL | B-001 R1 | B-001s |
| --- | --- | --- | --- | --- | --- |
| PHerc0009B_20250919125754_w4680_1820 | PHerc0009B | 0.5494 | 0.8026 | 0.8029 | 0.8306 |
| PHerc0009B_20250919125754_w5200_1820 | PHerc0009B | 0.6258 | 0.8203 | 0.8022 | 0.7959 |
| PHerc0009B_20250919125754_w5200_2340 | PHerc0009B | 0.7850 | 0.8253 | 0.7968 | 0.8470 |
| PHerc0009B_20250919125754_w5200_3120 | PHerc0009B | 0.8056 | 0.9439 | 0.9467 | 0.9390 |
| PHerc0009B_20250919125754_w5200_3900 | PHerc0009B | 0.8164 | 0.8824 | 0.8649 | 0.8586 |
| PHerc0009B_20250919125754_w5200_4420 | PHerc0009B | 0.7904 | 0.7343 | 0.7094 | 0.7240 |
| PHerc0009B_20250919125754_w5720_1560 | PHerc0009B | 0.7522 | 0.8605 | 0.8720 | 0.8608 |
| PHerc0009B_20250919125754_w5720_4160 | PHerc0009B | 0.4398 | 0.5581 | 0.5906 | 0.7296 |
| PHerc0009B_20250919125754_w5980_2080 | PHerc0009B | 0.7222 | 0.8238 | 0.8427 | 0.8650 |
| PHerc0009B_20250919125754_w5980_2600 | PHerc0009B | 0.7341 | 0.9030 | 0.9114 | 0.9007 |
| PHerc0009B_20250919125754_w5980_3120 | PHerc0009B | 0.8432 | 0.9435 | 0.9487 | 0.9608 |
| PHerc0009B_20250919125754_w5980_3640 | PHerc0009B | 0.9545 | 0.9733 | 0.9838 | 0.9894 |
| PHerc0500P2_20250825181859_w1820_1040 | PHerc0500P2 | 0.7178 | 0.8088 | 0.8226 | 0.7881 |
| PHerc0500P2_20250825181859_w2600_1300 | PHerc0500P2 | 0.7773 | 0.8521 | 0.8401 | 0.8292 |
| PHerc0500P2_20250825181859_w2600_780 | PHerc0500P2 | 0.7310 | 0.7651 | 0.7897 | 0.7897 |
| PHerc0500P2_20250825181859_w2860_2080 | PHerc0500P2 | 0.6125 | 0.7740 | 0.8032 | 0.7922 |
| PHerc0500P2_20250825181859_w2860_260 | PHerc0500P2 | 0.6744 | 0.9010 | 0.9268 | 0.9629 |
| PHerc0500P2_20250825181859_w3120_1300 | PHerc0500P2 | 0.6820 | 0.7769 | 0.7891 | 0.8287 |
| PHerc0500P2_20250825181859_w3380_780 | PHerc0500P2 | 0.7500 | 0.8798 | 0.8522 | 0.8902 |
| PHerc0500P2_20250825181859_w3900_780 | PHerc0500P2 | 0.6565 | 0.8910 | 0.8987 | 0.9006 |
| PHerc0500P2_20250825181859_w4420_0 | PHerc0500P2 | 0.7323 | 0.7219 | 0.7868 | 0.7088 |
| PHerc0500P2_20250825181859_w4420_1560 | PHerc0500P2 | 0.7805 | 0.8888 | 0.8898 | 0.8881 |
| PHerc0500P2_20250825181859_w4420_2080 | PHerc0500P2 | 0.8826 | 0.9781 | 0.9763 | 0.9487 |
| PHerc0500P2_20250825181859_w4680_1040 | PHerc0500P2 | 0.7352 | 0.8953 | 0.9097 | 0.9211 |
| PHerc0841_20260220213127_w1560_2860 | PHerc0841 | 0.6710 | 0.8262 | 0.8102 | 0.8119 |
| PHerc0841_20260220213127_w1560_3380 | PHerc0841 | 0.7304 | 0.8576 | 0.8735 | 0.8781 |
| PHerc0841_20260220213127_w2080_2600 | PHerc0841 | 0.7784 | 0.7952 | 0.7868 | 0.8038 |
| PHerc0841_20260220213127_w2080_3120 | PHerc0841 | 0.6975 | 0.8331 | 0.8510 | 0.8784 |
| PHerc0841_20260220213127_w2080_3640 | PHerc0841 | 0.4471 | 0.6933 | 0.7498 | 0.3589 |
| PHerc0841_20260220213127_w2600_2600 | PHerc0841 | 0.7230 | 0.8300 | 0.8359 | 0.8492 |
| PHerc0841_20260220213127_w2860_3120 | PHerc0841 | 0.7737 | 0.7985 | 0.8571 | 0.8320 |
| PHerc0841_20260220213127_w3120_2600 | PHerc0841 | 0.9427 | 0.9592 | 0.9592 | 0.9805 |
| PHerc0841_20260220213127_w520_3380 | PHerc0841 | 0.8352 | 0.9061 | 0.9198 | 0.9295 |
| PHerc0841_20260220214732_w2600_1560 | PHerc0841 | 0.6830 | 0.8571 | 0.8625 | 0.8458 |
| PHerc0841_20260220214732_w2600_2080 | PHerc0841 | 0.6491 | 0.7918 | 0.7964 | 0.7064 |
| PHerc0841_20260220214732_w2600_2860 | PHerc0841 | 0.8557 | 0.9346 | 0.9274 | 0.9329 |
| PHerc0841_20260220214732_w780_1560 | PHerc0841 | 0.5206 | 0.6109 | 0.5351 | 0.6017 |
| PHerc0841_20260220214732_w780_2080 | PHerc0841 | 0.8725 | 0.9028 | 0.9050 | 0.9246 |
| PHerc0841_20260220214732_w780_2600 | PHerc0841 | 0.6383 | 0.6397 | 0.6534 | 0.6587 |
| PHerc0841_20260221022814_w0_2340 | PHerc0841 | 0.2878 | 0.2938 | 0.3697 | 0.2922 |
| PHerc0841_20260221022814_w1040_1560 | PHerc0841 | 0.7121 | 0.8497 | 0.8230 | 0.8198 |
| PHerc0841_20260221022814_w1040_2080 | PHerc0841 | 0.7678 | 0.7848 | 0.7924 | 0.8085 |
| PHerc0841_20260221022814_w1300_2600 | PHerc0841 | 0.7713 | 0.8714 | 0.8611 | 0.8605 |
| PHerc0841_20260221022814_w1300_3120 | PHerc0841 | 0.9183 | 0.9056 | 0.9096 | 0.9246 |
| PHerc0841_20260221022814_w1560_780 | PHerc0841 | 0.2205 | 0.6303 | 0.7932 | 0.5771 |
| PHerc0841_20260221022814_w1820_1300 | PHerc0841 | 0.7464 | 0.7880 | 0.7818 | 0.7950 |
| PHerc0841_20260221022814_w520_1820 | PHerc0841 | 0.4641 | 0.1680 | 0.2582 | 0.3049 |
| PHerc0841_20260221022814_w520_2340 | PHerc0841 | 0.7087 | 0.7757 | 0.7657 | 0.8540 |
| PHerc0841_20260221022814_w780_2860 | PHerc0841 | 0.8883 | 0.8263 | 0.8898 | 0.9105 |

## B-001 post hoc (descriptive, not part of the rule): how close is REAL to the 2.4 µm teacher?

Computed on 2026-10-11, after the verdict. The teacher is the canonical 2.4 µm model's published
prediction of each test segment, read through B-001's CT-only correction at every LR px. The pixels are B-001's
scoring pixels (CT-valid, supervised, teacher in bounds) on the same 49 windows.

| scroll | windows | S0 (stock ink_9um) | REAL (B-001) | teacher (2.4 µm map) |
| --- | --- | --- | --- | --- |
| PHerc0009B (8.64 µm) | 12 | 0.7349 | 0.8393 | 0.9275 |
| PHerc0500P2 | 12 | 0.7277 | 0.8444 | 0.9628 |
| PHerc0841 | 25 | 0.6921 | 0.7652 | 0.8739 |
| all | 49 | 0.7113 | 0.8027 | 0.9088 |

REAL closes about 46% of the gap between the stock 9 µm model and the 2.4 µm teacher (0.0914 of 0.1975). REAL is at
or above the teacher on 6 of 49 windows. Caveat: the human labels were likely drawn with the high-resolution
renders and predictions in view, which favours the teacher.

Also descriptive, from `summary.json`: S0 scored with grid-index (uncorrected) labels averages 0.7086 vs 0.7113 with
corrected labels. The 100-300 µm label misplacement changes window-level AUC only slightly here.

## Side finding: first look at PHerc0800 (8.64 µm Grand Prize scroll), descriptive

No labels exist, so nothing here is scored. Driver: `code/b001g/b001g_read.py`. The maps themselves are not published.

- Read: four whole public `auto_grown` segments on scan `20250521135224` (`20251029010146…642`, `20251028220042…762`,
  `20251028220955…262`, `20251028213516…907`), centre 28 of 31 layers, dz = 0, both directions, stock and B-001 REAL
  and R1.
- Stock ink_9um gives scattered, unorganised blobs. B-001 REAL and R1 (forward) put bright blobs into horizontal rows
  on all four segments. Reverse maps are less organised. **No letters are legible.**
- The papyrus's horizontal fibres also run in rows, and fibre structure is a known false-positive mode. So a
  shuffled-layer control was run: Pearson correlation between REAL's forward map and its map on the layer-shuffled
  stack. If ink on a surface drives the map, shuffling depth should decorrelate it.
  - PHerc0800 mean correlation **0.366** (per segment 0.510 / 0.213 / 0.275 / 0.467).
  - B-001's 49 labelled test windows: p10-p90 **0.023-0.342**.
  - Registered outcome: `MIXED` (between 0.2 and 0.5), and above the test windows' p90.
- **Conclusion:** the rows depend on depth order less than ink does on the labelled scrolls. They are not claimed as
  text.
