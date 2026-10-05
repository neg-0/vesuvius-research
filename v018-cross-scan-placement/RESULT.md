# V-018 result: published cross-scan surface volumes are offset 25–115 µm in-plane, some by up to 390 µm

Run 2026-10-05 on cloud CPU. Code: `code/`. Data in `results/`: `audit.jsonl` (raw), `summary.md` (tables),
`mesh_offset.jsonl`, `leads_500um.jsonl`.

## What was measured

Every segment in the open-data bucket with surface volumes on 2 or more scans: 231 segments, 228 checked,
290 scan pairs, 0 errors (3 had too little overlap). For each pair, up to 3 windows of 1.5 mm, comparing the
other scan's render with the reference scan's render of the same segment in shared µm coordinates.

## Findings

1. **The check reproduces #1843.** PHerc1667 1.129 µm fails on all 25 pairs: median NCC 0.05 with no shift
   and 0.19 at the best shift. No placement within ±200 µm explains it, which matches the broken transform
   found in V-014.
2. **Everywhere else, the renders agree only after a shift.** Median NCC rises from 0.17–0.77 with no shift
   to 0.83–0.93 at the best shift, with median in-plane shifts of 25–115 µm:

   | Scroll | Other scan | No shift | Best shift | In-plane shift |
   | --- | --- | --- | --- | --- |
   | PHerc0139 | 1.129 µm | 0.61 | 0.91 | 58 µm |
   | PHercParis4 | 1.129 µm | 0.27 | 0.91 | 114 µm |
   | PHerc0814 | 1.129 µm | 0.39 | 0.93 | 97 µm |
   | PHerc0500P2 | 4.317 µm | 0.77 | 0.92 | 25 µm |
   | 8.6–9.4 µm scans (7 pairs) | | 0.17–0.58 | 0.56–0.88 | 63–113 µm |

   Shift directions are consistent within a scroll (for example PHerc0009B and PHerc0343P at 8.64 µm,
   about dy −69, dx −52 to −69 µm), so this is systematic, not noise. Offsets this size are several sheet
   spacings at 1.129 µm and matter for ink labels carried across scans.
3. **Part of it is a mesh-index convention.** Comparing each segment's coarse and fine tifxyz meshes through
   the published transform, the coarse vertex i lands on fine index i·r + (r−1)/2, as block averaging read with
   a corner convention predicts. Measured vs predicted offset in fine cells (row/col):
   PHerc0139 9.362 vs 2.399 µm 1.46/1.45 vs 1.46/1.45; PHerc0500P2 9.362 vs 2.215 µm 1.61/1.57 vs 1.62/1.60;
   PHerc0500P2 4.317 vs 2.215 µm 0.46/0.47 vs 0.47/0.47; PHerc0814 2.399 vs 1.129 µm 0.56/0.57 vs 0.56/0.56.
4. **The convention does not explain the 1.129 µm offsets.** It predicts about 13 µm there; the audit sees
   58–114 µm. The remainder is unexplained (transform or registration residuals are the likely source; not
   verified).
5. **The low-match segments outside PHerc1667 are larger offsets, not wrong sheets.** Seven pairs had best NCC
   under 0.4 at ±200 µm. Rerun with villa-fork `check_placement.py --search-um 500 --windows 6`
   (`leads_500um.jsonl`), every one but one reaches best NCC 0.72–0.95 at a median in-plane shift of 250–390 µm:
   PHercParis4 1.129 µm w038-045, w046-052_jordi, w064-068, w085-088 (the last two on 1 usable window only) and
   PHerc0814 auto_grown 20250925182632, 20250925204843 (9.362 µm) and 20250926051122 (1.129 µm). The exception
   is PHerc0814 20250926051122 on 9.362 µm (best 0.38 at ±500 µm), which stays unexplained. So outside PHerc1667
   no published cross-scan surface volume is on the wrong papyrus, but some are offset by several hundred µm,
   which is many sheet spacings.

## Not yet reported upstream

villa #1727 (non-reproducible cross-scan renders) and #1912 (depth offsets) don't cover this, as far as
checked on 2026-10-05.

## Caveats

- NCC on 1.5 mm windows at 2× the coarser voxel size; sparse ink or damaged regions lower it regardless of
  placement.
- The search cap is ±200 µm in-plane and ±8 layers; a larger true offset would show as a low best NCC.
- The reference is the scan closest to 2.4 µm, so "offset" is relative, not absolute.

## Next

The check is packaged as `foundation/volume-registration/check_placement.py` on the neg-0/villa branch
claude/project-thread-s6f6nh (commit ca0ed2d, with tests).
