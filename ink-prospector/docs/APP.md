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
