"""Tests of the geometry, volume reading and file-format code, on small synthetic volumes. No network, no torch.
Run from this folder: python -m pytest -q
These check the code paths; what the tool finds on real scroll data is in the README ("What has been checked").
"""
import json, os, subprocess, sys
import numpy as np
import pytest
import tifffile
from lazyvol import LazyVol
import grow, runs

HERE = os.path.dirname(os.path.abspath(__file__))


def write_zarr(folder, vol, chunk=32, level=0, dtype="|u1"):
    """Minimal OME-Zarr v2 level, uncompressed, '.'-separated keys, as LazyVol reads it."""
    d = os.path.join(folder, str(level)); os.makedirs(d, exist_ok=True)
    json.dump(dict(zarr_format=2, shape=list(vol.shape), chunks=[chunk] * 3, dtype=dtype, compressor=None, fill_value=0,
                   order="C", filters=None), open(os.path.join(d, ".zarray"), "w"))
    for i in range(0, vol.shape[0], chunk):
        for j in range(0, vol.shape[1], chunk):
            for k in range(0, vol.shape[2], chunk):
                c = np.zeros((chunk,) * 3, np.uint8); b = vol[i:i + chunk, j:j + chunk, k:k + chunk]; c[:b.shape[0], :b.shape[1], :b.shape[2]] = b
                if c.any(): open(os.path.join(d, f"{i // chunk}.{j // chunk}.{k // chunk}"), "wb").write(c.tobytes())
    return folder


def sheet(shape=(96, 96, 96), z0=48.0, tilt=0.15, width=1.6):
    """m7-like volume: one bright ridge on the plane z = z0 + tilt * (y - 48)."""
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    d = (z - (z0 + tilt * (y - 48))) / np.sqrt(1 + tilt ** 2)
    return (255 * np.exp(-(d / width) ** 2)).astype(np.uint8), lambda p: (p[..., 0] - (z0 + tilt * (p[..., 1] - 48))) / np.sqrt(1 + tilt ** 2)


def test_lazyvol_local_zarr_matches_array(tmp_path):
    rng = np.random.default_rng(0); vol = rng.integers(1, 255, (70, 50, 40), dtype=np.uint8)   # shape not a chunk multiple
    lv = LazyVol(write_zarr(str(tmp_path / "v.zarr"), vol))
    assert list(lv.shape) == [70, 50, 40]
    pts = np.array([[0, 0, 0], [69, 49, 39], [33, 17, 5], [40, 31, 32]], float).T
    assert np.array_equal(lv.sample(pts), vol[tuple(pts.astype(int))].astype(np.float32))
    assert np.array_equal(lv.block((-2, 10, 30), (5, 20, 45))[2:, :, :10], vol[0:5, 10:20, 30:40])
    assert lv.block((-2, 10, 30), (5, 20, 45))[:2].sum() == 0                    # outside the volume reads 0
    mid = lv.sample(np.array([[10.5], [3.0], [3.0]]))[0]
    assert mid == pytest.approx((float(vol[10, 3, 3]) + float(vol[11, 3, 3])) / 2)   # trilinear


def test_lazyvol_missing_chunk_is_zero_and_dtype_checked(tmp_path):
    vol = np.zeros((64, 64, 64), np.uint8); vol[40:, 40:, 40:] = 7
    lv = LazyVol(write_zarr(str(tmp_path / "v.zarr"), vol))                       # only chunk 1.1.1 is written
    assert lv.sample(np.array([[5.0], [5.0], [5.0]]))[0] == 0 and lv.sample(np.array([[50.0], [50.0], [50.0]]))[0] == 7
    write_zarr(str(tmp_path / "f.zarr"), vol.astype(np.uint8), dtype="<f4")
    with pytest.raises(ValueError): LazyVol(str(tmp_path / "f.zarr"))
    with pytest.raises(FileNotFoundError): LazyVol(str(tmp_path / "f.zarr"), level=3)


def test_seed_frame_snaps_to_sheet_and_none_off_sheet(tmp_path):
    vol, dist = sheet(); m7 = LazyVol(write_zarr(str(tmp_path / "m7.zarr"), vol))
    p0, n, U, V, *_ = grow.seed_frame(m7, np.array([52.0, 48.0, 48.0]))
    assert abs(dist(p0)) < 1.0                                                    # snapped onto the ridge
    want = np.array([1.0, -0.15, 0]) / np.sqrt(1 + 0.15 ** 2)
    assert abs(abs(n @ want) - 1) < 0.02                                          # normal fitted by PCA
    assert abs(U @ n) < 1e-5 and abs(V @ n) < 1e-5 and abs(U @ V) < 1e-5
    assert grow.seed_frame(m7, np.array([15.0, 48.0, 48.0])) is None             # 30 vx from any sheet


def test_grow_tracks_a_tilted_sheet(tmp_path):
    vol, dist = sheet(); m7 = LazyVol(write_zarr(str(tmp_path / "m7.zarr"), vol))
    p0, n, U, V, *_ = grow.seed_frame(m7, np.array([50.0, 48.0, 48.0]))
    P, ok, N = grow.grow(m7, p0, n, U, V, half_nodes=8)
    assert P.shape == (17, 17, 3) and ok.mean() > 0.95
    assert np.abs(dist(P[ok])).max() < 1.0                                        # every node on the sheet
    gap = np.linalg.norm(np.diff(P, axis=1), axis=-1); gap = gap[ok[:, 1:] & ok[:, :-1]]
    assert 3.5 < np.median(gap) < 4.6                                             # 4 vx node spacing
    D, valid, Nd = grow.densify(P, ok, 4)
    assert valid.mean() > 0.9 and np.abs(dist(D[valid])).max() < 1.0


def test_render_fast_samples_along_normal(tmp_path):
    vol, dist = sheet(); m7 = LazyVol(write_zarr(str(tmp_path / "m7.zarr"), vol))
    p0, n, U, V, *_ = grow.seed_frame(m7, np.array([50.0, 48.0, 48.0]))
    P, ok, N = grow.grow(m7, p0, n, U, V, half_nodes=6); D, valid, Nd = grow.densify(P, ok, 4)
    st = grow.render_fast(m7, D, Nd, 17, valid=valid)                             # render the ridge itself as "CT"
    prof = st[:, valid].mean(1)
    assert st.shape[0] == 17 and prof.argmax() == 8 and prof[8] > 200 and prof[0] < 20 and prof[16] < 20


def test_export_tifxyz_roundtrip(tmp_path):
    G = 9; P = np.zeros((G, G, 3), np.float32); ii, jj = np.mgrid[:G, :G]
    P[..., 0] = 500 + 0.5 * ii; P[..., 1] = 1000 + 4 * ii; P[..., 2] = 2000 + 4 * jj
    ok = np.zeros((G, G), bool); ok[2:7, 1:8] = True; ok[4, 4] = False
    np.savez(tmp_path / "p0001.npz", P=P, ok=ok, n0=np.array([1.0, 0, 0]))
    out = tmp_path / "p0001.tifxyz"
    subprocess.check_call([sys.executable, os.path.join(HERE, "export_tifxyz.py"), str(tmp_path / "p0001.npz"), str(out), "--scroll", "S"])
    x, y, z = (tifffile.imread(out / f"{a}.tif") for a in "xyz")
    assert x.shape == (5, 7) and x.dtype == np.float32                            # cropped to the tracked rows/cols
    m = ok[2:7, 1:8]
    assert np.array_equal(x > 0, m) and np.array_equal(z > 0, m) and (x[~m] == -1).all()
    assert np.array_equal(z[m], P[2:7, 1:8, 0][m]) and np.array_equal(y[m], P[2:7, 1:8, 1][m]) and np.array_equal(x[m], P[2:7, 1:8, 2][m])
    meta = json.load(open(out / "meta.json"))
    assert meta["format"] == "tifxyz" and meta["scale"] == [0.25, 0.25] and meta["scroll"] == "S"
    assert meta["bbox"][0] == [2004.0, 1008.0, 501.0]


def test_empty_scan_folder(tmp_path):
    d = tmp_path / "scan"; d.mkdir()
    (d / "scores.tsv").write_text("id\tseed_z\tseed_y\tseed_x\tstatus\ttracked\tarea_cm2\tfrac_a\tfrac_b\n0\t1\t2\t3\tno_sheet\n")
    assert runs.patches([str(d)]) == []
    r = subprocess.run([sys.executable, os.path.join(HERE, "prospect.py"), "rank", str(d), "--sheet", str(tmp_path / "s.png")],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0 and "0 usable patches" in r.stdout and not (tmp_path / "s.png").exists()


def test_judge_whole_rule(tmp_path):
    d = tmp_path / "whole" / "shard-0"; d.mkdir(parents=True)
    head = "id\tseed_z\tseed_y\tseed_x\tstatus\ttracked\tarea_cm2\tfrac_a\tfrac_b\tp995_a\tp995_b\tstroke_a\tstroke_b\tsecs\n"
    rows = ["1\t10\t20\t30\tok\t0.9\t0.5\t0.1\t0.1\t1\t1\t0.0300\t0.0050\t9",    # hit: 3% one side, other under a third
            "2\t11\t21\t31\tok\t0.9\t0.5\t0.1\t0.1\t1\t1\t0.0300\t0.0200\t9",    # both faces: not a hit
            "3\t12\t22\t32\tok\t0.9\t0.5\t0.1\t0.1\t1\t1\t0.0100\t0.0000\t9",    # under 2%: not a hit
            "4\t13\t23\t33\tno_sheet"]
    (d / "scores.tsv").write_text(head + "\n".join(rows) + "\n")
    out = subprocess.run([sys.executable, os.path.join(HERE, "judge_whole.py"), str(tmp_path / "whole")], capture_output=True, text=True, check=True).stdout
    assert "HITS" in out and "p0001" in out and "p0002" not in out and "p0003" not in out and "'no_sheet': 1" in out


def _fake_run(folder, pid, z0, ts_blobs):
    """A scan folder with one ok patch: a flat 41x41 node grid on z = z0 and face images from test_smoke's blobs."""
    from test_smoke import blobs, blank, save
    os.makedirs(folder, exist_ok=True); G = 41; ii, jj = np.mgrid[:G, :G]
    P = np.stack([np.full((G, G), z0, np.float32), 1000 + 4 * ii, 2000 + 4 * jj], -1).astype(np.float32); ok = np.ones((G, G), bool); ok[:3] = False
    np.savez(os.path.join(folder, f"p{pid:04d}.npz"), P=P, ok=ok, n0=np.array([1.0, 0, 0]))
    save(os.path.join(folder, f"p{pid:04d}_a.png"), blobs(ts_blobs)); save(os.path.join(folder, f"p{pid:04d}_b.png"), blank())
    with open(os.path.join(folder, "scores.tsv"), "a") as f:
        if f.tell() == 0: f.write("id\tseed_z\tseed_y\tseed_x\tstatus\ttracked\tarea_cm2\tfrac_a\tfrac_b\n")
        f.write(f"{pid}\t{z0}\t1080\t2080\tok\t0.9\t0.5\t0.20\t0.01\n")


def test_export_every_keeps_vc_step(tmp_path):
    G = 41; ii, jj = np.mgrid[:G, :G]; P = np.stack([np.full((G, G), 9.0), 4.0 * ii, 4.0 * jj], -1); ok = np.ones((G, G), bool); ok[:2] = False
    from export_tifxyz import write_tifxyz
    n = write_tifxyz(P, ok, str(tmp_path / "s"), every=5)
    x = tifffile.imread(tmp_path / "s" / "x.tif"); y = tifffile.imread(tmp_path / "s" / "y.tif")
    assert json.load(open(tmp_path / "s" / "meta.json"))["scale"] == [0.05, 0.05]
    assert np.allclose(np.diff(x, axis=1), 20) and np.allclose(np.diff(y, axis=0), 20) and n == x.size   # 20 vx apart, all tracked


def test_vc_export_and_run_ids(tmp_path):
    for run, z0 in (("runA", 500), ("runB", 700)):
        _fake_run(str(tmp_path / run / "shard-0"), 1, z0, [(64, 64), (64, 192), (192, 64), (192, 192)])
    ids = [p["id"] for p in runs.patches([str(tmp_path / "runA"), str(tmp_path / "runB")])]
    assert sorted(ids) == ["runA-s0-1", "runB-s0-1"]                                # no clash between runs
    out = tmp_path / "vc"
    subprocess.check_call([sys.executable, os.path.join(HERE, "vc_export.py"), str(out), str(tmp_path / "runA"), str(tmp_path / "runB"), "--m7", "S/m7.zarr"])
    pc = json.load(open(out / "prospector_hits.json"))
    assert pc["vc_pointcollections_json_version"] == "1"
    pts = pc["collections"]["1"]["points"]; assert len(pts) == 2
    for p in pts.values():
        x, y, z = p["p"]; assert z in (500.0, 700.0) and 1000 <= y <= 1160 and 2000 <= x <= 2160   # x y z, on the sheet
    sh = (out / "grow.sh").read_text()
    assert "--resume paths/prospector_runA-s0-1 " in sh and "vesuvius-challenge-open-data" in sh and "S/m7.zarr" in sh
    assert json.load(open(out / "paths" / "prospector_runA-s0-1" / "meta.json"))["scale"] == [0.05, 0.05]
