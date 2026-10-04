#!/usr/bin/env python3
"""V-012: self-contained PHerc.0800 seating check for one segment (cloud port).

Combines the V-008 mesh read, V-009 sample/footprint, and V-010 scoring into
one bounded attempt that needs no prior local run directories. The sampling,
normal, gap-probe and score functions are copied unchanged from
v010_p800_seating.py. Usage (needs numpy, tifffile; output in $PUZZLE_RUN_DIR or ./v012-run):

    python3 v012_p800_seating_cloud.py <segment_id>
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import tifffile

ROOT = Path(os.environ.get("V012_ROOT", Path.cwd()))
CWD = Path.cwd()
MANIFEST = json.loads((Path(__file__).resolve().parent / "v012_p800_seating_manifest.json").read_text(encoding="utf-8"))
LIMITS = MANIFEST["limits"]
RUN_DIR = Path(os.environ.get("PUZZLE_RUN_DIR", ROOT / "v012-run"))
EVIDENCE_DIR = RUN_DIR / "evidence"
OUT_PATH = RUN_DIR / "payload/v012-seating.json"
BASE = MANIFEST["bucket_endpoint"]


class Unknown(RuntimeError):
    pass


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, value: object) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def resources() -> dict:
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    return {"disk_free_bytes": shutil.disk_usage(ROOT).free,
            "host_memory_available_bytes": available, "gpu": "none (cloud container)"}


# --- copied unchanged from v010_p800_seating.py (MANIFEST keys identical) ---

def central_difference(array: np.ndarray, axis: int) -> np.ndarray:
    grad = np.zeros_like(array)
    if axis == 1:
        grad[:, 1:-1] = array[:, 2:] - array[:, :-2]
        grad[:, 0] = 2.0 * (array[:, 1] - array[:, 0])
        grad[:, -1] = 2.0 * (array[:, -1] - array[:, -2])
    else:
        grad[1:-1] = array[2:] - array[:-2]
        grad[0] = 2.0 * (array[1] - array[0])
        grad[-1] = 2.0 * (array[-1] - array[-2])
    return grad


def grid_normals(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
    ux, uy, uz = (central_difference(a, 1) for a in (x, y, z))
    vx, vy, vz = (central_difference(a, 0) for a in (x, y, z))
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return nx / norm, ny / norm, nz / norm


def sample_points(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    valid = (x > 0) & (y > 0) & (z > 0)
    indices = np.argwhere(valid)
    if not len(indices):
        raise ValueError("mesh has no positive coordinate points")
    rng = np.random.default_rng(MANIFEST["sample_seed"])
    chosen = indices[rng.choice(len(indices), size=min(MANIFEST["sample_count"], len(indices)), replace=False)]
    r, c = chosen[:, 0], chosen[:, 1]
    nx, ny, nz = grid_normals(x, y, z)
    points = np.stack((x[r, c], y[r, c], z[r, c]), axis=1).astype(np.float64)
    normals = np.stack((nx[r, c], ny[r, c], nz[r, c]), axis=1).astype(np.float64)
    return points, normals, chosen


class ChunkVolume:
    def __init__(self, shape_zyx, chunks_zyx, blocks: dict):
        self.shape = shape_zyx
        self.chunks = chunks_zyx
        self.blocks = blocks  # None means a verified missing Zarr key; use fill_value 0.
        self.missing_reads_as_zero = 0
        self.out_of_bounds_reads = 0

    def at(self, pz: np.ndarray, py: np.ndarray, px: np.ndarray) -> np.ndarray:
        out = np.zeros(len(pz), dtype=np.float32)
        inside = ((pz >= 0) & (pz < self.shape[0]) & (py >= 0) & (py < self.shape[1])
                  & (px >= 0) & (px < self.shape[2]))
        self.out_of_bounds_reads += int((~inside).sum())
        if not np.any(inside):
            return out
        z, y, x = pz[inside], py[inside], px[inside]
        keys = np.stack((z // self.chunks[0], y // self.chunks[1], x // self.chunks[2]), axis=1)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        edges = np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1
        starts, ends = np.r_[0, edges], np.r_[edges, len(sorted_keys)]
        values = np.zeros(len(z), dtype=np.float32)
        zs, ys, xs = z[order], y[order], x[order]
        for (start, end), row in zip(zip(starts, ends), (sorted_keys[s] for s in starts)):
            key = tuple(int(t) for t in row)
            block = self.blocks.get(key, "__UNPLANNED__")
            if isinstance(block, str) and block == "__UNPLANNED__":
                raise Unknown(f"seating scorer attempted an unplanned CT chunk: {key}")
            if block is None:
                self.missing_reads_as_zero += int(end - start)
                continue
            values[start:end] = block[
                zs[start:end] - key[0] * self.chunks[0],
                ys[start:end] - key[1] * self.chunks[1],
                xs[start:end] - key[2] * self.chunks[2],
            ]
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        out[inside] = values[inverse]
        return out


def probe_gap(points: np.ndarray, normals: np.ndarray, volume) -> float:
    divisor = float(2 ** MANIFEST["level"])
    px, py, pz = (points[:, i] * MANIFEST["mesh_scale"] / divisor for i in range(3))
    nx, ny, nz = (normals[:, i] for i in range(3))
    steps = np.arange(2.0, 60.0, 2.0)
    means = []
    for distance in steps:
        plus = volume.at(np.rint(pz + nz * distance).astype(np.int64),
                         np.rint(py + ny * distance).astype(np.int64),
                         np.rint(px + nx * distance).astype(np.int64))
        minus = volume.at(np.rint(pz - nz * distance).astype(np.int64),
                          np.rint(py - ny * distance).astype(np.int64),
                          np.rint(px - nx * distance).astype(np.int64))
        means.append(0.5 * (plus.mean() + minus.mean()))
    smooth = np.convolve(np.asarray(means), np.ones(3) / 3.0, mode="same")
    for index in range(1, len(smooth) - 1):
        if smooth[index] <= smooth[index - 1] and smooth[index] <= smooth[index + 1]:
            return float(steps[index])
    return float(steps[int(np.argmin(smooth))])


def seating_score(points: np.ndarray, normals: np.ndarray, volume) -> tuple[float, float, float, float]:
    divisor = float(2 ** MANIFEST["level"])
    px, py, pz = (points[:, i] * MANIFEST["mesh_scale"] / divisor for i in range(3))
    nx, ny, nz = (normals[:, i] for i in range(3))
    gap = probe_gap(points, normals, volume)
    def read(distance: float) -> np.ndarray:
        return volume.at(np.rint(pz + nz * distance).astype(np.int64),
                         np.rint(py + ny * distance).astype(np.int64),
                         np.rint(px + nx * distance).astype(np.int64))
    centre, before, after = read(0.0), read(-gap), read(gap)
    on = centre > 40
    coverage = float(on.mean())
    if coverage < MANIFEST["min_coverage"]:
        return -1.0, float(centre.mean()), coverage, gap
    contrast = float((centre[on] - 0.5 * (before[on] + after[on])).mean())
    return contrast * coverage, float(centre[on].mean()), coverage, gap


class SyntheticVolume:
    def __init__(self, mode: str, center: int = 1000):
        self.mode = mode
        self.center = center
    def at(self, pz, py, px):
        if self.mode == "empty":
            return np.zeros(len(pz), dtype=np.float32)
        delta = np.abs(pz - self.center)
        if self.mode == "sheet":
            return np.where(delta <= 1, 120.0, 0.0).astype(np.float32)
        if self.mode == "crosscut":
            return np.where(delta <= 1000, 100.0, 0.0).astype(np.float32)
        raise ValueError(self.mode)

# --- end of copied functions ---


def test_controls() -> dict:
    points = np.tile(np.asarray([[4000.0, 4000.0, 4000.0]]), (600, 1))
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (600, 1))
    pos = seating_score(points, normals, SyntheticVolume("sheet"))
    neg = seating_score(points, normals, SyntheticVolume("crosscut"))
    empty = seating_score(points, normals, SyntheticVolume("empty"))
    if pos[0] < MANIFEST["threshold_seated"] or pos[2] < MANIFEST["min_coverage"]:
        raise RuntimeError("synthetic seated-sheet positive control failed")
    if neg[0] >= MANIFEST["threshold_seated"]:
        raise RuntimeError("synthetic cross-cut null control failed")
    if empty[2] >= MANIFEST["min_coverage"] or empty[0] != -1.0:
        raise RuntimeError("synthetic empty-volume null control failed")
    return {"synthetic_seated_sheet_score": pos[0], "synthetic_crosscut_score": neg[0],
            "synthetic_empty_score": empty[0], "all_pass": True}


class Client:
    def __init__(self):
        self.transferred = 0
        self.requests = []
        self.started = time.monotonic()

    def get(self, key: str, max_bytes: int, allow_404: bool = False):
        if time.monotonic() - self.started >= LIMITS["hard_runner_timeout_seconds"] - 15:
            raise Unknown("internal stop before hard run timeout")
        if self.transferred + max_bytes > LIMITS["max_total_response_bytes"]:
            raise Unknown("next request could exceed aggregate response cap")
        url = urllib.parse.urljoin(BASE, urllib.parse.quote(key, safe="/"))
        req = urllib.request.Request(url, headers={"Accept-Encoding": "identity",
                                                   "User-Agent": "Writrunner-V012-p800-seating/1"})
        record = {"key": key}
        response = None
        body = bytearray()
        try:
            try:
                response = urllib.request.urlopen(req, timeout=LIMITS["request_timeout_seconds"])
            except urllib.error.HTTPError as error:
                if not (allow_404 and error.code == 404):
                    record["status"] = error.code
                    raise Unknown(f"HTTP {error.code} for {key}")
                response = error
            record.update(status=response.code, etag=response.headers.get("ETag"))
            if urllib.parse.urlsplit(response.geturl()).netloc != urllib.parse.urlsplit(BASE).netloc:
                raise Unknown(f"unexpected redirect host for {key}")
            if response.headers.get("Content-Encoding") not in (None, "identity"):
                raise Unknown(f"unexpected content encoding for {key}")
            limit = LIMITS["max_missing_chunk_body_bytes"] if response.code == 404 else max_bytes
            while True:
                block = response.read(min(64 * 1024, limit + 1 - len(body)))
                if not block:
                    break
                body.extend(block)
                if len(body) > limit:
                    raise Unknown(f"response exceeds bound for {key}")
            record["bytes"] = len(body)
            record["sha256"] = hashlib.sha256(body).hexdigest()
            return response.code, bytes(body)
        except urllib.error.URLError as error:
            raise Unknown(f"network error for {key}: {error.reason}") from error
        finally:
            self.transferred += len(body)
            self.requests.append(record)
            if response is not None:
                response.close()


def main() -> int:
    segment = sys.argv[1]
    if segment not in MANIFEST["allowed_segments"]:
        raise RuntimeError(f"segment not registered in V-012 manifest: {segment}")
    expected = MANIFEST["allowed_segments"][segment]
    result = {"schema_version": 1, "hypothesis": "V-012", "segment_id": segment,
              "eligible_volume_id": MANIFEST["eligible_volume_id"], "status": "running",
              "resources_before": resources()}
    client = Client()
    try:
        if result["resources_before"]["disk_free_bytes"] < LIMITS["minimum_free_disk_bytes"]:
            raise Unknown("disk floor not met")
        result["controls"] = test_controls()
        mesh_prefix = f"PHerc0800/segments/{segment}/mesh/{segment.split('-')[0]}-on-{MANIFEST['eligible_volume_id']}-8.64um.tifxyz/"
        status, meta_raw = client.get(mesh_prefix + "meta.json", 4096)
        meta = json.loads(meta_raw)
        atomic_write(EVIDENCE_DIR / "meta.json", meta_raw)
        if meta.get("scale") != MANIFEST["expected_mesh_scale_yx"]:
            raise Unknown(f"unexpected mesh scale {meta.get('scale')}")
        arrays, tiffs = {}, {}
        for axis in "xyz":
            _, raw = client.get(mesh_prefix + f"{axis}.tif", expected["coordinate_tif_bytes"])
            if len(raw) != expected["coordinate_tif_bytes"]:
                raise Unknown(f"{axis}.tif length {len(raw)} differs from listing")
            path = EVIDENCE_DIR / f"{axis}.tif"
            atomic_write(path, raw)
            arrays[axis] = np.asarray(tifffile.imread(path))
            tiffs[axis] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                           "shape": list(arrays[axis].shape), "dtype": str(arrays[axis].dtype)}
        result["coordinate_tiffs"] = tiffs
        x, y, z = arrays["x"], arrays["y"], arrays["z"]
        valid = (x > 0) & (y > 0) & (z > 0)
        result["mesh"] = {"grid_shape": list(x.shape), "valid_points": int(valid.sum()),
                          "bbox_xyz_min": [float(a[valid].min()) for a in (x, y, z)],
                          "bbox_xyz_max": [float(a[valid].max()) for a in (x, y, z)],
                          "meta_bbox": meta.get("bbox")}
        _, header_raw = client.get(MANIFEST["volume_prefix"] + f"{MANIFEST['level']}/.zarray", 32768)
        header = json.loads(header_raw)
        atomic_write(EVIDENCE_DIR / "level2.zarray", header_raw)
        if (header.get("shape") != MANIFEST["level_shape_zyx"] or header.get("chunks") != MANIFEST["level_chunks_zyx"]
                or header.get("dtype") != "|u1" or header.get("compressor") is not None):
            raise Unknown("level-2 header differs from frozen V-010 header")
        points, normals, chosen = sample_points(x, y, z)
        sample_hash = hashlib.sha256(chosen.astype("<i8", copy=False).tobytes()).hexdigest()
        result["sample"] = {"count": len(points), "seed": MANIFEST["sample_seed"], "indices_sha256": sample_hash}
        if expected.get("sample_indices_sha256") and sample_hash != expected["sample_indices_sha256"]:
            raise Unknown("sample hash differs from the registered reproduction value")
        # Footprint: every chunk the scorer can touch (centre plus every probe distance).
        div = float(2 ** MANIFEST["level"])
        p = points * MANIFEST["mesh_scale"] / div
        shape, chunks = tuple(MANIFEST["level_shape_zyx"]), tuple(MANIFEST["level_chunks_zyx"])
        planned = set()
        for d in [0.0] + [s * sign for s in MANIFEST["probe_steps_level_voxels"] for sign in (-1.0, 1.0)]:
            ix, iy, iz = (np.rint(p[:, i] + normals[:, i] * d).astype(np.int64) for i in range(3))
            inside = (iz >= 0) & (iz < shape[0]) & (iy >= 0) & (iy < shape[1]) & (ix >= 0) & (ix < shape[2])
            for row in np.unique(np.stack([iz[inside] // chunks[0], iy[inside] // chunks[1], ix[inside] // chunks[2]], 1), axis=0):
                planned.add(tuple(int(t) for t in row))
        planned = sorted(planned)
        chunk_bytes = int(np.prod(chunks))
        result["footprint"] = {"chunk_count": len(planned), "max_bytes": len(planned) * chunk_bytes,
                               "chunks_zyx": [list(k) for k in planned]}
        atomic_json(EVIDENCE_DIR / "planned-chunks.json", result["footprint"])
        if len(planned) * chunk_bytes > LIMITS["max_ct_chunk_bytes"]:
            raise Unknown("planned footprint exceeds registered CT budget; no CT requested")
        blocks, missing = {}, 0
        for key in planned:
            k = f"{MANIFEST['volume_prefix']}{MANIFEST['level']}/{key[0]}/{key[1]}/{key[2]}"
            code, raw = client.get(k, chunk_bytes, allow_404=True)
            if code == 404:
                blocks[key] = None
                missing += 1
                continue
            if len(raw) != chunk_bytes:
                raise Unknown(f"chunk body length mismatch for {k}")
            blocks[key] = np.frombuffer(raw, dtype=np.uint8).reshape(chunks)
        volume = ChunkVolume(shape, chunks, blocks)
        score, centre_mean, coverage, gap = seating_score(points, normals, volume)
        if score >= MANIFEST["threshold_seated"] and coverage >= MANIFEST["min_coverage"]:
            verdict = "seated_sample"
        elif score > MANIFEST["threshold_no"]:
            verdict = "partial_sample"
        else:
            verdict = "not_seated_sample"
        result.update(status="complete", verdict=verdict, score=score, centre_mean_on=centre_mean,
                      coverage=coverage, gap_level_voxels=gap, missing_chunks=missing,
                      missing_reads_as_zero=volume.missing_reads_as_zero,
                      out_of_bounds_reads=volume.out_of_bounds_reads)
        if expected.get("reference_score") is not None:
            diff = abs(score - expected["reference_score"])
            result["reproduction"] = {"reference_score": expected["reference_score"], "abs_diff": diff,
                                      "pass": diff <= 1e-8}
        return_code = 0
    except Unknown as error:
        result.update(status="unknown", error=str(error))
        return_code = 2
    finally:
        result["requests"] = len(client.requests)
        result["response_bytes"] = client.transferred
        result["resources_after"] = resources()
        atomic_json(EVIDENCE_DIR / "requests.json", client.requests)
        atomic_json(OUT_PATH, result)
    print(json.dumps({k: result.get(k) for k in ("status", "verdict", "score", "coverage", "gap_level_voxels",
                                                    "missing_chunks", "requests", "response_bytes", "reproduction",
                                                    "error")}, indent=1))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
