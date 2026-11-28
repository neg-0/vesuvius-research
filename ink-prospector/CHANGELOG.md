# Ink Prospector changelog

Releases are tagged `ink-prospector-vX.Y.Z`; each tag covers this folder only. To cut one, move the
Unreleased notes under a new `## X.Y.Z (date)` heading and merge. Then run the "ink-prospector release" workflow on
main with version X.Y.Z (Actions tab, Run workflow), or push the tag `ink-prospector-vX.Y.Z`. Either way the workflow
publishes that section as the release notes (`python3 release_notes.py X.Y.Z` prints it) with a tar.gz of this folder.

## Unreleased

- `prospect.py --version` prints the release this copy is from (`+dev` when it has changes made after it).
- Release notes join wrapped changelog lines, and re-running the release workflow for a version refreshes its notes (its archive stays the tagged one).

## 0.1.0 (2026-11-28)

First tagged release: everything from the November 2026 public release up to #65.

### Find ink without a mesh
- `prospect.py scan`: seeds a lattice in a box, grows a small mesh-free patch on each uncovered sheet, reads both
  faces with the ink reader and ranks patches by a text score. Resume-safe; `--shard i/n` splits a box between machines.
- `prospect.py rank`, `status`, `stitch`, `viewer`, `spread`: rank and browse runs, check shard progress, stitch top
  patches into sheets, build a static viewer, grow one hit outward.
- Whole-scroll and dense-box runners (`run_whole.sh`, `run_box.sh`) with `judge_whole.py` and `sheet_whole.py`.
- Local zarr input, and OME-Zarr / remote volumes through `LazyVol` with a disk cache and an optional memory cap
  (`--max-cache-gb`, #25).

### Score existing segments
- `score_tifxyz.py` scores tifxyz segments (#21), a whole sheet tile by tile (#23), with `--resume` (#27),
  `--shard` (#28), `--limit` chunks (#49), progress and time left (#48). Large grids are memory-mapped (#29); tiles
  with no CT under them are skipped (#42); a failing tile no longer ends the run (#34).

### Letter-likeness
- `letterscore.py` (#24): a letter-likeness score for ink maps, calibrated gates (#26), a halo term that rejects haze
  cut into letter-sized forms (#41), the pass rule a map failed (#44). `--dirs` batches skip broken folders (#46),
  tell apart folders with the same name (#53), expand parent folders (#54) and run in parallel with `--jobs` (#55).
- `rank --tsv --letters` adds the letterscore columns to the table (#47).

### Hand hits to Volume Cartographer
- `export_tifxyz.py` writes tifxyz that villa's reader loads (#16); `vc_export.py` and VC grow hand hits to VC3D (#20).
- Auto mode with VC grow and fibre snapping (#40, #43).

### App
- `prospect.py serve`: local server and web UI to run and browse scans (#36), with a letter-likeness column (#37),
  sheet stitching (#39) and a read-only demo on GitHub Pages (#38).

### Viewer and ranking
- Full-size ink maps of the best patches (#32), each tile's tifxyz window and a fixed known-text percent (#30),
  `rank --tsv` (#33), `--min-area` in rank, viewer and vc_export (#51, #52), `status` shows what each shard has left and counts rows of each status (#31, #50).

### Speed
- Text and letter scores are remembered per patch while its images are unchanged (#61, #62).
- `render_fast` samples only valid pixels of partly valid tiles: about 1.8x faster with identical output on a
  synthetic patch (#64).

### Robustness
- Bad `--shard`, `--box`, `--spacing` and `--limit` values, missing tifxyz files and folders that hold no scan stop
  with a message before anything is written (#56, #57, #58, #59, #60, #63).
- `judge_whole.py` and `sheet_whole.py` share one candidate rule and read one-shard runs (#65).
- Cut-off or broken cache chunks are fetched again instead of crashing (#35, #45).
- `seed_frame` ignores isolated specks (#22, thanks @Johnsmr429).
- Tests and CI on Ubuntu and macOS (#17).
