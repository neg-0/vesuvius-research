# D-003 protocol (registered before the audit ran)

**Question.** Villa issue #2107 reports Hugging Face surface volumes whose level 2 holds only the first 17 of 65 depth slices while the metadata declares full depth. Does the same defect exist in the open-data bucket copy of the surface volumes that our experiments read?

**Scope.** Every surface-volume zarr root found under `*/segments/*/surface-volumes/` in the bucket (listing taken at audit time). CPU and network only.

**Checks per volume** (zarr v2 pyramid, probes at fixed fractions of the level-0 extent, first one whose level-0 chunk is non-zero):

| Flag | Meaning |
|---|---|
| META_XY_Lk / META_DEPTH_Lk | level k XY is not ceil(level k-1 / 2) (+-1), or depth differs from the depth implied by the declared z-scale ratio |
| CHUNK_MISSING_Lk | the probed chunk at level k returns 404 |
| DEPTH_Lk | fewer than 90% of depth layers hold any non-zero voxel at level k, while level 0 has at least 90% |
| XLEVEL_L1 | correlation of level 1 with the 2x2-pooled level 0 over the same window is below 0.9 |
| PROBE_EMPTY | no probe position had a non-zero level-0 chunk (nothing to compare; not a defect) |

**Rule.** BUCKET_CLEAN if no META, CHUNK, DEPTH, XLEVEL or ERROR flag on any volume; otherwise BUCKET_AFFECTED with the list.

**Run.** `cd code && python3 -I d003_pyramid.py list && python3 -I d003_pyramid.py audit && python3 -I d003_pyramid.py rule` (run one `audit` at a time; it is resumable). `python3 -I d003_pyramid.py selftest` checks the rule on synthetic pyramids.
