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
    cd = tmp_path / "cache"; z = write_zarr(str(tmp_path / "v.zarr"), vol)       # disk cache: reused, and a broken file is refetched
    LazyVol(z, cache_dir=str(cd)).block((0, 0, 0), (64, 64, 64))
    assert sorted(os.listdir(cd)) == [f"{i}_{j}_{k}.npy" for i in (0, 1) for j in (0, 1) for k in (0, 1)]   # no .tmp left
    (cd / "1_1_1.npy").write_bytes(b"\x93NUMPY trunc")                          # a run killed mid-write
    assert np.array_equal(LazyVol(z, cache_dir=str(cd)).block((0, 0, 0), (64, 64, 64)), vol)
    assert np.load(cd / "1_1_1.npy").shape == (32, 32, 32)


def test_lazyvol_refetches_a_cut_off_chunk_and_names_a_broken_one(tmp_path):
    vol = np.random.default_rng(2).integers(1, 255, (32, 32, 32), dtype=np.uint8)
    lv = LazyVol(write_zarr(str(tmp_path / "v.zarr"), vol)); calls = []
    def flaky(url):
        calls.append(url); raw = open(url, "rb").read(); return raw[:len(raw) // 2] if len(calls) == 1 else raw
    lv._read = flaky; lv.local = False                                            # act like a remote store
    assert np.array_equal(lv.block((0, 0, 0), (32, 32, 32)), vol) and len(calls) == 2
    lv2 = LazyVol(write_zarr(str(tmp_path / "w.zarr"), vol)); lv2._read = lambda url: b"junk"
    with pytest.raises(IOError, match="does not decode"): lv2.block((0, 0, 0), (32, 32, 32))


def test_seed_frame_snaps_to_sheet_and_none_off_sheet(tmp_path):
    vol, dist = sheet(); m7 = LazyVol(write_zarr(str(tmp_path / "m7.zarr"), vol))
    p0, n, U, V, *_ = grow.seed_frame(m7, np.array([52.0, 48.0, 48.0]))
    assert abs(dist(p0)) < 1.0                                                    # snapped onto the ridge
    want = np.array([1.0, -0.15, 0]) / np.sqrt(1 + 0.15 ** 2)
    assert abs(abs(n @ want) - 1) < 0.02                                          # normal fitted by PCA
    assert abs(U @ n) < 1e-5 and abs(V @ n) < 1e-5 and abs(U @ V) < 1e-5
    assert grow.seed_frame(m7, np.array([15.0, 48.0, 48.0])) is None             # 30 vx from any sheet


def test_seed_frame_is_none_next_to_a_lone_sheet_voxel(tmp_path):
    vol = np.zeros((64, 64, 64), np.uint8); vol[40, 32, 32] = 255                # an isolated speck in the prediction
    m7 = LazyVol(write_zarr(str(tmp_path / "m7.zarr"), vol))
    assert grow.seed_frame(m7, np.array([36.0, 32.0, 32.0])) is None


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



def test_lazyvol_cache_cap_keeps_output_and_bounds_memory(tmp_path):
    vol, _ = sheet((96, 160, 160)); rng = np.random.default_rng(1); vol = np.maximum(vol, rng.integers(0, 40, vol.shape, dtype=np.uint8))
    z = write_zarr(str(tmp_path / "ct.zarr"), vol, chunk=16)
    yy, xx = np.mgrid[2:150, 2:150].astype(np.float64); D = np.stack([48 + 0.15 * (yy - 48), yy, xx], -1)
    Nd = np.broadcast_to(np.array([1.0, -0.15, 0]) / np.sqrt(1.0225), D.shape).copy(); valid = np.ones(D.shape[:2], bool)
    full = LazyVol(z); a = grow.render_fast(full, D, Nd, 9, tile=32, valid=valid)
    cap = 40 * 16 ** 3 / 1e9; small = LazyVol(z, max_cache_gb=cap); b = grow.render_fast(small, D, Nd, 9, tile=32, valid=valid)
    assert np.array_equal(a, b)
    assert small.max_chunks == 40 and len(small.cache) <= 40 < len(full.cache)
    assert np.array_equal(small.block((0, 0, 0), (96, 160, 160)), vol)            # a read larger than the cap is still whole

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


def test_score_tifxyz_tiles_cover_a_sheet_once(tmp_path):
    # Tiles share an edge row/column, so their densified patches abut: together they cover the sheet exactly once.
    import score_tifxyz
    ok = np.ones((23, 31), bool); ok[:, 25:] = False                           # a ragged sheet edge
    W = score_tifxyz.tiles(ok.shape, 8, ok)
    cover = np.zeros((22 * 4, 30 * 4), int)                                   # densified at step 4
    for r0, r1, c0, c1 in W:
        assert r1 - r0 <= 12 and c1 - c0 <= 12
        cover[r0 * 4:(r1 - 1) * 4, c0 * 4:(c1 - 1) * 4] += 1
    assert (cover[:, :24 * 4] == 1).all()                                     # every cell of the valid part once
    assert all(ok[r0:r1, c0:c1].sum() >= 16 for r0, r1, c0, c1 in W)          # empty tiles beyond the edge skipped
    assert len(score_tifxyz.tiles((23, 31), 8)) == 3 * 4                       # short last strips merged


def test_score_tifxyz_loads_and_densifies_any_step(tmp_path):
    # A VC-style tifxyz: 20-voxel grid (scale 0.05) on the plane z = 300, two invalid nodes (-1 and z = 0).
    from export_tifxyz import write_tifxyz
    import score_tifxyz
    G0, G1 = 6, 9; ii, jj = np.mgrid[:G0, :G1]
    P = np.stack([np.full((G0, G1), 300.0), 1000 + 20.0 * ii, 2000 + 20.0 * jj], -1); ok = np.ones((G0, G1), bool)
    write_tifxyz(P, ok, str(tmp_path / "seg"), step=20.0)
    z = tifffile.imread(tmp_path / "seg" / "z.tif"); z[0, 0] = 0; tifffile.imwrite(tmp_path / "seg" / "z.tif", z)
    Q, qok, step = score_tifxyz.load_tifxyz(str(tmp_path / "seg"))
    assert step == pytest.approx(20.0) and Q.shape == (G0, G1, 3) and not qok[0, 0] and qok.sum() == G0 * G1 - 1
    D, valid, Nd = score_tifxyz.to_dense(Q, qok, step)
    assert D.shape == ((G0 - 1) * 20, (G1 - 1) * 20, 3)                          # rectangular grids work
    assert np.allclose(D[valid][:, 0], 300, atol=0.5) and abs(abs(Nd[valid][:, 0]).mean() - 1) < 1e-3
    Q2, ok2, _ = score_tifxyz.load_tifxyz(str(tmp_path / "seg"), crop=(1, 4, 2, 7))
    assert Q2.shape == (3, 5, 3) and ok2.all()
    assert isinstance(score_tifxyz._read(str(tmp_path / "seg"))[2], np.memmap)            # uncompressed: memory-mapped
    assert np.array_equal(score_tifxyz.valid_nodes(str(tmp_path / "seg"), rows=2), qok)   # row-block mask = whole mask
    assert np.array_equal(score_tifxyz.valid_nodes(str(tmp_path / "seg"), (1, 4, 2, 7)), ok2)
    for a in "xyz":                                                                       # compressed tifs still load
        tifffile.imwrite(tmp_path / "seg" / f"{a}.tif", tifffile.imread(tmp_path / "seg" / f"{a}.tif"), compression="zlib")
    score_tifxyz._read.cache_clear(); Q3, qok3, _ = score_tifxyz.load_tifxyz(str(tmp_path / "seg"))
    assert np.array_equal(Q3, Q) and np.array_equal(qok3, qok)



def test_score_tifxyz_resume_scores_only_missing_patches(tmp_path, monkeypatch, capsys):
    # A stub reader (no torch) that counts calls: a resumed run must redo only the patch whose score row is missing.
    import types
    from export_tifxyz import write_tifxyz
    import score_tifxyz
    calls = []
    stub = types.SimpleNamespace(load_reader=lambda *a: None,
                                 run_stack=lambda cm, st, **kw: calls.append(1) or st[8].astype(np.float32) / (510 if kw.get("reverse") else 255))
    monkeypatch.setitem(sys.modules, "reader", stub)
    vol, _ = sheet(); ct = write_zarr(str(tmp_path / "ct.zarr"), vol)
    G = 13; ii, jj = np.mgrid[:G, :G]
    P = np.stack([48 + 0.15 * (16 + 4.0 * ii - 48), 16 + 4.0 * ii, 16 + 4.0 * jj], -1)
    write_tifxyz(P, np.ones((G, G), bool), str(tmp_path / "seg"), step=4.0)
    out = str(tmp_path / "scan"); args = [str(tmp_path / "seg"), "--out", out, "--ct", ct, "--tile", "7", "--um", "8"]
    score_tifxyz.main(args); first = open(out + "/scores.tsv").readlines(); n = len(first) - 1
    assert n == 4 and len(calls) == 2 * n
    log = capsys.readouterr().out; assert "1 of 4 this run" in log and "4 of 4 this run, ~0 min left" in log   # progress and time left
    o2 = str(tmp_path / "chunked"); a2 = [x if x != out else o2 for x in args]; calls.clear()    # --limit: time-boxed chunks
    score_tifxyz.main(a2 + ["--limit", "3"]); assert len(open(o2 + "/scores.tsv").readlines()) == 4 and len(calls) == 6
    assert "1 patch(es) of this shard left" in capsys.readouterr().out
    score_tifxyz.main(a2 + ["--resume", "--limit", "3"]); assert len(calls) == 8
    cut = lambda L: sorted(r.split("\t")[:-1] for r in L[1:])                          # same rows as one full run (minus timing)
    assert cut(open(o2 + "/scores.tsv").readlines()) == cut(first)
    open(out + "/scores.tsv", "w").writelines(first[:-1])                    # cut off before the last patch's score
    calls.clear(); score_tifxyz.main(args + ["--resume"])
    again = open(out + "/scores.tsv").readlines()
    assert len(calls) == 2 and len(again) == len(first)
    assert [l.split("\t")[:13] for l in again] == [l.split("\t")[:13] for l in first]   # same rows, same values
    assert len(open(out + "/surfaces.tsv").readlines()) == n + 1
    calls.clear(); score_tifxyz.main(args + ["--resume"]); assert not calls      # nothing left to do
    import runs, build_viewer                                                    # rank and viewer name each tile's tifxyz window
    P = runs.patches([out]); assert len(P) == n and all(p["src"].startswith("seg rows ") for p in P)
    cut = sorted(p["area"] for p in P)[1]                                          # --min-area drops the smaller patches
    assert [p["id"] for p in runs.patches([out], min_area=cut)] == [p["id"] for p in P if p["area"] >= cut] and len(runs.patches([out], min_area=1e9)) == 0
    runs.write_tsv(P, str(tmp_path / "ranked.tsv")); import csv
    T = list(csv.DictReader(open(tmp_path / "ranked.tsv"), delimiter="\t"))
    assert [t["id"] for t in T] == [p["id"] for p in P] and [int(t["rank"]) for t in T] == list(range(1, n + 1))
    assert all(os.path.exists(t["png"]) and t["src"] == p["src"] for t, p in zip(T, P))
    runs.write_tsv(P, str(tmp_path / "ranked_l.tsv"), with_letters=True)          # letterscore columns on request
    TL = list(csv.DictReader(open(tmp_path / "ranked_l.tsv"), delimiter="\t"))
    assert [t["id"] for t in TL] == [t["id"] for t in T] and all(t["letters"] in ("True", "False") and t["fails"] for t in TL)
    assert all(t["fails"] == "-" if t["letters"] == "True" else t["fails"] != "-" for t in TL)
    assert all(float(t["letter_score"]) >= 0 for t in TL) and "letters" not in T[0]
    build_viewer.main([str(tmp_path / "view"), "t", out])
    D = json.load(open(tmp_path / "view" / "data.json")); assert {p["src"] for p in D["patches"]} == {p["src"] for p in P}
    with pytest.raises(AssertionError, match="no usable"):                       # viewer honours --min-area too
        build_viewer.main([str(tmp_path / "view_min"), "t", out, "--min-area", str(max(p["area"] for p in P) * 1.01)])
    assert all(len(p["full"]) == 2 and (tmp_path / "view" / p["full"][0]).exists() for p in D["patches"])   # full-size maps copied
    build_viewer.main([str(tmp_path / "view2"), "t", out, "--full", "1"])
    D = json.load(open(tmp_path / "view2" / "data.json")); assert [len(p["full"]) for p in D["patches"]] == [2] + [0] * (n - 1)
    assert len(os.listdir(tmp_path / "view2" / "full")) == 2
    with pytest.raises(SystemExit):                                              # a different tiling is refused
        score_tifxyz.main(args[:-3] + ["5", "--um", "8", "--resume"])
    rows = {}                                                                    # two shards: every patch once, same values
    for k in (0, 1):
        calls.clear(); score_tifxyz.main(args[:2] + [out + f"-s{k}"] + args[3:] + ["--shard", f"{k}/2"])
        got = [l.split("\t") for l in open(out + f"-s{k}/scores.tsv").readlines()[1:]]
        assert len(calls) == 2 * len(got) and all(int(r[0]) % 2 == k for r in got)
        rows.update({r[0]: r[:13] for r in got})
    assert rows == {l.split("\t")[0]: l.split("\t")[:13] for l in first[1:]}
    R, Gs = runs.status([out + "-s0", out + "-s1"])                              # status: both shards complete
    assert [(r["done"], r["expected"], r["missing"]) for r in R] == [(2, 2, []), (2, 2, [])]
    assert len(Gs) == 1 and Gs[0]["jobs"] == n and Gs[0]["missing"] == []
    s1 = open(out + "-s1/scores.tsv").readlines(); open(out + "-s1/scores.tsv", "w").writelines(s1[:-1])   # shard 1 cut off
    R, Gs = runs.status([out + "-s0", out + "-s1"]); last = int(s1[-1].split("\t")[0])
    assert R[1]["missing"] == [last] and Gs[0]["missing"] == [last]
    R, Gs = runs.status([out + "-s0"])                                           # shard 1 forgotten: its ids are missing
    assert R[0]["missing"] == [] and Gs[0]["missing"] == [1, 3] and Gs[0]["have"] == [0]
    calls.clear(); stub.run_stack = lambda cm, st, **kw: calls.append(1) or st[8].astype(np.float32) / (510 if kw.get("reverse") else 255)
    far = np.stack([5048 + 0.15 * (16 + 4.0 * ii - 48), 16 + 4.0 * ii, 16 + 4.0 * jj], -1); write_tifxyz(far, np.ones((G, G), bool), str(tmp_path / "far"), step=4.0)   # a sheet off the scan
    score_tifxyz.main([str(tmp_path / "far"), "--out", out + "-far", "--ct", ct, "--tile", "7", "--um", "8"])
    rows_far = [l.split("\t") for l in open(out + "-far/scores.tsv").readlines()[1:]]
    assert not calls and len(rows_far) == n and all(r[4].strip() == "noct" for r in rows_far)   # skipped without a reader call
    assert runs.status([out + "-far"])[0][0]["missing"] == [] and runs.patches([out + "-far"]) == []
    assert runs.status([out + "-far"])[0][0]["kinds"] == {"noct": n} and runs.status([out + "-s0"])[0][0]["kinds"] == {"ok": R[0]["done"]}
    boom = {"n": 0}                                                              # a patch that throws does not end the run
    def flaky(cm, st, **kw):
        boom["n"] += 1
        if boom["n"] == 3: raise OSError("chunk fetch failed")
        return st[8].astype(np.float32) / (510 if kw.get("reverse") else 255)
    stub.run_stack = flaky; bad = out + "-flaky"
    with pytest.raises(SystemExit, match="1 patch"):
        score_tifxyz.main(args[:2] + [bad] + args[3:])
    got = {l.split("\t")[0] for l in open(bad + "/scores.tsv").readlines()[1:]}
    assert got == {"0", "2", "3"} and runs.status([bad])[0][0]["missing"] == [1]
    score_tifxyz.main(args[:2] + [bad] + args[3:] + ["--resume"])                # --resume retries only the failed one
    assert [l.split("\t")[:13] for l in sorted(open(bad + "/scores.tsv").readlines()[1:])] == [l.split("\t")[:13] for l in sorted(first[1:])]
    keep = open(out + "/scores.tsv").read()                                      # a mistyped option leaves a finished run alone
    for extra in (["--shard", "2/2"], ["--shard", "x"], ["--limit", "0"]):
        with pytest.raises(SystemExit): score_tifxyz.main(args + extra)
        assert open(out + "/scores.tsv").read() == keep
    calls.clear()                                                                # a mistyped surface path: named, nothing read or written
    with pytest.raises(SystemExit, match="segX.*x.tif is missing"): score_tifxyz.main([str(tmp_path / "segX")] + args[1:])
    assert open(out + "/scores.tsv").read() == keep and not calls

def test_letterscore_passes_a_row_and_rejects_haze_blobs_bars(tmp_path, capsys):
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    import letterscore
    um, N = 16.0, 1300; px = lambda mm: int(mm * 1000 / um)
    strokes = [[(0, 1, .5, 0), (.5, 0, 1, 1), (.2, .6, .8, .6)], [(0, 0, 0, 1), (0, 0, 1, 0), (1, 0, 1, 1)], [(0, 0, 1, 0), (.5, 0, .5, 1)],
               [(0, 0, 0, 1), (0, 0, .9, 0), (0, .5, .7, .5), (0, 1, .9, 1)], [(0, 0, 1, 1), (1, 0, 0, 1)]]
    def glyphs(spots):
        im = Image.new("F", (N, N), 0); d = ImageDraw.Draw(im); H = px(3.2)
        for k, (y, x) in enumerate(spots):
            for a, b, c, e in strokes[k % 5]:
                d.line([(px(x) + a * H * .8, px(y) + b * H), (px(x) + c * H * .8, px(y) + e * H)], fill=0.95, width=25)
        return np.asarray(im, np.float32)
    z = np.zeros((N, N), np.float32); yy, xx = np.mgrid[:N, :N]
    row = glyphs([(3, 1 + 4 * k) for k in range(5)])
    ok = letterscore.score_maps(row, z, None, um)
    assert ok["letters"] and ok["row"] == 5 and ok["form_share"] > 0.9 and ok["halo"] < 0.05
    fog = np.maximum(row, np.float32(0.5))                                                 # same strokes standing in half-confident haze
    hz = letterscore.score_maps(fog, z, None, um)
    assert hz["row"] == 5 and hz["halo"] > 0.9 and not hz["letters"] and hz["score"] < 0.1 * ok["score"]
    assert ok["fails"] == "-" and hz["fails"] == "halo"                                     # says which rule failed
    assert "two-sided" in letterscore.score_maps(row, row, None, um)["fails"]              # two-sided
    haze = np.clip(ndi.gaussian_filter(np.random.default_rng(0).normal(0, 1, (N, N)), 20) * 25 + 0.5, 0, 1).astype(np.float32)
    assert not letterscore.score_maps(haze, z, None, um)["letters"]
    blobs = np.zeros_like(z); bars = np.zeros_like(z)
    for k in range(5):
        blobs[(yy - px(4.5)) ** 2 + (xx - px(2.5 + 4 * k)) ** 2 < px(1.5) ** 2] = 0.95
        bars[px(3):px(7.5), px(2 + 4 * k):px(2 + 4 * k) + 30] = 0.95
    assert not letterscore.score_maps(blobs, z, None, um)["letters"]
    assert not letterscore.score_maps(bars, z, None, um)["letters"]
    scattered = glyphs([(1, 1), (6, 8), (2, 13), (7, 1.5), (5, 4.2)])
    assert not letterscore.score_maps(scattered, z, None, um)["letters"]
    np.save(tmp_path / "prob_v3_fwd.npy", z); np.save(tmp_path / "prob_v3_rev.npy", row)       # ink on the rev face
    r = letterscore.score_dir(str(tmp_path), um)
    assert r["face"] == "rev" and r["letters"]
    (tmp_path / "empty").mkdir()                                                          # a broken folder in a batch
    with pytest.raises(SystemExit, match="1 folder"): letterscore.main(["--dirs", str(tmp_path / "empty"), str(tmp_path), "--um", str(um)])
    out = capsys.readouterr()
    assert "skipped" in out.err and out.out.splitlines()[1].startswith(tmp_path.name + "\tTrue")
    for s in ("a", "b"):                                                                  # same folder name under two runs
        (tmp_path / s / "w").mkdir(parents=True)
        for f in ("prob_v3_fwd.npy", "prob_v3_rev.npy"): (tmp_path / s / "w" / f).write_bytes((tmp_path / f).read_bytes())
    letterscore.main(["--dirs", str(tmp_path / "a" / "w"), str(tmp_path / "b" / "w") + "/", "--um", str(um)])
    assert sorted(l.split("\t")[0] for l in capsys.readouterr().out.splitlines()[1:]) == ["a/w", "b/w"]
    (tmp_path / "runs").mkdir(); os.rename(tmp_path / "a", tmp_path / "runs" / "a"); os.rename(tmp_path / "b", tmp_path / "runs" / "b")
    letterscore.main(["--dirs", str(tmp_path / "runs"), "--um", str(um)])                # a parent folder ranks every follow-up under it
    assert sorted(l.split("\t")[0] for l in capsys.readouterr().out.splitlines()[1:]) == ["a/w", "b/w"]
    letterscore.main(["--dirs", str(tmp_path / "runs"), "--um", str(um)]); one = capsys.readouterr().out
    letterscore.main(["--dirs", str(tmp_path / "runs"), "--um", str(um), "--jobs", "2"])  # parallel gives the same table
    assert capsys.readouterr().out == one


def test_bad_shard_exits_before_any_file_is_written(tmp_path):
    import runs, subprocess
    assert runs.shard("1/4") == (1, 4) and runs.shard("0/1") == (0, 1)
    for bad in ("2/2", "-1/3", "x", "1/2/3", "3"):
        with pytest.raises(SystemExit, match="--shard"): runs.shard(bad)
    out = tmp_path / "run"; out.mkdir(); (out / "run.json").write_text('{"keep": 1}')    # a scan run already in this folder
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "prospect.py"), "scan", "--ct", "c", "--m7", "m", "--box", "0", "1", "0", "1", "0", "1",
                        "--out", str(out), "--shard", "2/2"], capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode != 0 and "--shard 2/2" in r.stderr
    assert (out / "run.json").read_text() == '{"keep": 1}' and sorted(os.listdir(out)) == ["run.json"]


def test_rank_and_status_name_a_folder_that_is_not_a_run(tmp_path):
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__)); pr = os.path.join(here, "prospect.py")
    ok = tmp_path / "run"; ok.mkdir()
    (ok / "scores.tsv").write_text("id\tseed_z\tseed_y\tseed_x\tstatus\ttracked\tarea_cm2\tfrac_a\tfrac_b\tp995_a\tp995_b\tstroke_a\tstroke_b\tsecs\n")
    typo = str(tmp_path / "rnu")
    for cmd in ("rank", "status"):
        r = subprocess.run([sys.executable, pr, cmd, typo], capture_output=True, text=True)
        assert r.returncode != 0 and "rnu" in r.stderr and "not a scan run" in r.stderr      # only a typo: stop
        r = subprocess.run([sys.executable, pr, cmd, str(ok), typo], capture_output=True, text=True)
        assert r.returncode == 0 and "rnu" in r.stderr                                       # a typo among real runs: named, rest read
    for cmd in ("viewer", "stitch"):                                                         # same for the commands that write
        r = subprocess.run([sys.executable, pr, cmd, typo, "--out", str(tmp_path / ("o-" + cmd))], capture_output=True, text=True)
        assert r.returncode != 0 and "not a scan run" in r.stderr and "Traceback" not in r.stderr and not (tmp_path / ("o-" + cmd)).exists()
