# V-012 result — PHerc.0800 segment 225813 seating (cloud)

**Verdict: `partial_sample`, score 4.0733 (threshold 15), coverage 1.000.**
Segment 20251028225813-auto_grown_20251028225813045 does not pass the sampled
seating check, scoring lower than the first segment (6.6608).

[Protocol](PROTOCOL.md) ·
[script](v012_p800_seating_cloud.py) ·
[test payload](2026-10-04-v012-segment-225813.json) ·
[reproduction payload](2026-10-04-v012-reproduction-222030.json)

| Run | Segment | Role | Score | Coverage | Gap (L2 vox) | Centre mean (on) | Chunks | Response bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `20261004T000803-5673034ec2` | 222030 | reproduction control | 6.660833279 | 0.9983 | 8 | 102.50 | 46 | 96,669,613 |
| `20261004T000832-5e03c2398a` | 225813 | test | 4.073333263 | 1.0000 | 12 | 104.77 | 41 | 86,140,097 |

- Reproduction control passed: sample hash `6f56f0cc…` matched V-010 and the
  score differs from V-010's 6.6608332793 by 8.2e-12. Coordinate TIFF bbox
  matched segment metadata exactly in both runs.
- Synthetic controls passed in both runs: seated sheet 120.0, cross-cut 0.0,
  empty -1.0 (coverage branch).
- Test segment: TIFXYZ grid 115×113, 6,433 valid points, sample hash
  `412e7220…`, 41 planned level-2 chunks, 0 missing, 0 out-of-bounds reads.
- Tested 2 / skipped 0 / unknown 0. No GPU, no ink inference, no submission.
  Disk free after: 31.7 GB; memory available 16.2 GB.

## Interpretation and caveats

Both segments lie in the same region of the scroll (overlapping bboxes) and
both sit in bright material (centre ≈103–105, coverage ≈1) with weak contrast
against the gap probe. That pattern fits a mesh tracking dense papyrus but not
a single clean sheet, which is what the method's threshold rewards.

We still have no real-data positive control for this scorer on this scan; the
15 threshold comes from the pinned public method. A low score therefore means
"not shown to be seated", not "proven unseated". The remaining two eligible
segments (220955, 220042) are smaller by bbox projection and in the same
neighbourhood.
