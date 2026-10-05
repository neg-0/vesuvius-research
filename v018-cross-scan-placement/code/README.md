# V-018: cross-scan placement audit

Question: is a segment's surface volume on one scan placed on the
same papyrus as its surface volume on another scan? CPU only; reads the open-data bucket over HTTPS.

- `sv_placement.py <segment> [--ref R] [--windows N] [--size-um S] [--out F]`: for every published surface volume
  of the segment, resamples it and the reference scan's surface volume (the scan closest to 2.4 µm) to a shared
  (u, v, depth) µm grid at 2× the coarser voxel size, then reports NCC with no shift and the best NCC over a
  ±8-layer, ±200 µm FFT shift search, per random window inside the data mask (seed 0).
- `audit.py out.jsonl [windows]`: runs the check on every segment with 2+ surface volumes (resumable).
- `summarize.py audit.jsonl [out.md]`: per scroll/scan table and the list of pairs no shift explains.
- `mesh_offset.py <segment> <coarse mesh> <fine mesh> <volume with transform.json> [fine2coarse|coarse2fine]`:
  compares two tifxyz meshes of one segment through the published transform, without rendering.
- `zread.py`: minimal zarr v2 reader (numcodecs) with retries.

Needs numpy, scipy, numcodecs, tifffile. Results and write-up: `../RESULT.md`.
