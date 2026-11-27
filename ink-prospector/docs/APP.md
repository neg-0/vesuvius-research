# Ink Prospector app: design

Goal: one local app that drives the existing command line, so a run can be started, watched, stopped, resumed and
browsed (ranks, both ink faces, full-size maps, the 3D viewer) from a browser, without changing what the CLI writes.

## Shape

```
browser (app/index.html)  <-- JSON + files -->  prospect.py serve  --spawns-->  prospect.py scan / score_tifxyz.py / spread / stitch / viewer
                                                   |                                   |
                                                   +--- reads ---- <runs>/ (scan folders, run.json, scores.tsv, PNGs) <--- writes
```

- **Server:** `prospect.py serve --runs DIR [--port 8765] [--host 127.0.0.1]`. Python standard library only
  (`http.server`), so it adds no dependency and runs in the same venv or Docker image as the CLI. Binds to
  localhost by default: it starts processes, so it is not meant to face a network.
- **Jobs (`jobs.py`):** each job is one CLI process with its log. A job is stored as `<runs>/.jobs/<id>/job.json`
  and `log.txt`, so jobs survive a server restart. The argument list is built from a fixed set of job kinds and
  typed fields (no shell, no free-form command line). A job whose process is gone when the server starts again is
  marked `stopped`.
- **Resume:** the scan and scoring scripts are already resume-safe. Resuming a job runs the same command again
  (adding `--resume` for `score_tifxyz.py`), so only missing patches are done.
- **Results:** run folders are read with the same code as `prospect.py rank` and `status` (`runs.py`), so the app
  and the CLI cannot disagree. Ranked patches are cached per folder until `scores.tsv` changes.
- **3D viewer:** the app builds the existing static viewer into `<run>/_viewer/` with `build_viewer.py` and opens
  it. The GitHub Pages demo stays a static page built the same way; the app does not change it.

## API

| Method | Path | What |
|---|---|---|
| GET | `/api/scrolls` | scrolls.json |
| GET | `/api/runs` | run folders under `--runs`: patches, cm², shard progress (as `status`), last job |
| GET | `/api/runs/<name>/patches?n=200` | ranked patches, best first (as `rank`), with image URLs |
| POST | `/api/runs/<name>/viewer` | start a job building the 3D viewer for that run |
| GET | `/api/jobs` | all jobs, newest first |
| POST | `/api/jobs` | `{"kind": "scan" \| "score" \| "spread" \| "stitch" \| "viewer", ...fields}` start a job |
| GET | `/api/jobs/<id>` | one job and the last lines of its log (`?tail=200`) |
| POST | `/api/jobs/<id>/stop` | stop it (process group TERM) |
| POST | `/api/jobs/<id>/resume` | run its command again, resume-safe |
| GET | `/files/<path>` | any file under `--runs` (PNGs, TSVs, the built viewer) |

## UI (app/index.html, one file, no build step)

- **Runs:** every run folder with patch count, area, shard progress bar and the job writing it.
- **New run:** forms for a box scan (scroll, box, spacing, half-nodes, shards, device) and for scoring tifxyz
  surfaces (paths, scroll, tile, crop, shards).
- **Run page:** ranked table with both faces side by side, text score, side, seed, tifxyz window; click for
  full-size maps; buttons for rank TSV, contact sheet and 3D viewer.
- **Jobs:** live state and log tail; stop and resume.

## Plan

1. Server, jobs and API with tests (no network, fake jobs).
2. Web UI on top of it.
3. Letter-likeness column (done: `runs.letters`, sortable in the run page).
4. Read-only mode (done: with no server the app shows `demo/` data; on Pages at `ink-prospector/app/`).
5. Stitched sheet view (done: Stitch sheets on the run page runs `stitch` into `<run>/_stitch/`; the sheets show with a letter score).
6. Auto mode (done: `prospect.py auto`, the Auto form and the Auto panel on the run page; `--grow` runs VC's tracer).

## Real runs (cloud CPU, 4 cores, 16 GB, PHerc0306B, 2026-11-27)

- Box scan through the app: `9000-10000 2400-3400 3400-4400`, spacing 250 (64 seeds), 2 shards. It took about 4 min per
  patch per shard. Uncapped, two shards plus a score job were OOM-killed, so jobs now cap the chunk cache at 2 GB.
- Auto on that scan (16 usable patches so far), with `--leads 10 --spread 2 --grow 3` and the villa
  `volume-cartographer:edge` image as tracer, ran unattended in 44 min:

  | stage | time |
  |---|---|
  | leads, stitch, viewer, vc | under 5 s |
  | spread | 33 min (2 leads, 22 mm fused reads) |
  | grow + whole reads | 11 min (3 leads grown to 1.7-2.1 cm², about 50 s each in VC, 3 min to read each) |

  This box shows no letters (best spread letter score 1.1, under the pass mark). The run checks that the pipeline works
  end to end, not that the box holds text.
- Fibre snap (`fibre_snap.py`) on a quarter (0.6 cm²) of a VC-grown 306B segment: 151 AFV blocks in 15 min, mean shift
  0.55 voxels, 16% of points with a confident fibre peak. Read in 9 tiles, ink moved both ways from tile to tile, and the
  total inked share was about the same (0.078 before, 0.070 after). On a segment VC already traced well, snapping does
  not help much. It is meant for rougher surfaces.
