# D-003 result: BUCKET_CLEAN

692 surface volumes audited in the open-data bucket. None showed the villa #2107 defect or any other pyramid defect (no META, CHUNK, DEPTH or XLEVEL flags, no errors). Every level of every probed volume had the full set of depth layers and a level 1 that matched the pooled level 0.

Five volumes were flagged PROBE_EMPTY only, meaning no probe position hit a non-zero level-0 chunk, so there was nothing to compare. These are sparse volumes, not defects:

- four PHerc1447 8.64 um volumes (three `auto_grown` segments and `z_dbg_gen_00320`, volume `8.64um-1.2m-116keV-volume-20250521151220.zarr`)
- one PHerc1667 2.399 um volume (`w013_20240304141531_flatboi`, volume `2.399um-0.22m-78keV-volume-20251217075048.zarr`)

One rule refinement during the run, before the final pass: PHerc1447 volumes are z-pooled pyramids with a declared z-scale of 2.0, so depth is compared against the depth implied by the declared scale ratio, not against a constant. Without that, 692 volumes would not have been the clean set.

Per-volume rows: [`results/volumes.csv`](results/volumes.csv); summary: [`results/summary.json`](results/summary.json); flag list: [`results/RESULT.md`](results/RESULT.md).

**Scope limit.** This covers the bucket copy only. It does not say anything about the Hugging Face copy that #2107 reports, and the checks probe fixed positions per volume, not every chunk.
