"""Tests for the app server and jobs (prospect.py serve). No network: jobs here are short python one-liners."""
import json, os, sys, threading, time, urllib.error, urllib.request
import numpy as np
import pytest
import jobs as joblib
import server
from test_smoke import blank, blobs, _write_patch


def _scan(folder, ids=(1, 2)):
    os.makedirs(folder)
    with open(os.path.join(folder, "scores.tsv"), "w") as f:
        f.write("id\tseed_z\tseed_y\tseed_x\tstatus\ttracked\tarea_cm2\tfrac_a\tfrac_b\n")
        for i in ids:
            _write_patch(folder, i, blobs([(64, 64), (192, 192)][: i]), blank())
            f.write(f"{i}\t{10000 + i}\t3000\t5000\tok\t0.9\t0.5\t0.20\t0.01\n")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setitem(joblib.ARGV, "fake", lambda runs, f: ([sys.executable, "-c", f["code"]], joblib.inside(runs, f["out"])))
    runs = tmp_path / "runs"; runs.mkdir(); _scan(str(runs / "demo"))
    s = server.make_server(str(runs), port=0, home=str(tmp_path)); t = threading.Thread(target=s.serve_forever, daemon=True); t.start()
    base = f"http://127.0.0.1:{s.server_address[1]}"

    def call(path, body=None, headers=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers or {},
                                     method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req) as r: return r.status, r.read()
        except urllib.error.HTTPError as e: return e.code, e.read()
    yield call, runs
    for j in s.RequestHandlerClass.app.jobs.list():
        if j["state"] == "running": s.RequestHandlerClass.app.jobs.stop(j["id"])
    s.shutdown()


def J(r): return json.loads(r[1])


def wait(call, jid, states, secs=20):
    t = time.time()
    while time.time() - t < secs:
        j = J(call(f"/api/jobs/{jid}"))
        if j["state"] in states: return j
        time.sleep(0.1)
    raise AssertionError(f"job {jid} still {j['state']}")


def test_runs_and_ranked_patches(app):
    call, runs = app
    code, body = call("/api/runs"); assert code == 200
    (r,) = json.loads(body); assert r["name"] == "demo" and r["patches"] == 2 and r["scored"] == 2
    P = J(call("/api/runs/demo/patches"))
    assert [p["id"] for p in P] == ["demo-2", "demo-1"] and P[0]["side"] == "a"   # same order as prospect.py rank
    assert all(isinstance(p["ls"], float) and p["letters"] in (True, False) for p in P)   # letterscore.py on the ink face
    code, png = call(P[0]["a"]); assert code == 200 and png[:4] == b"\x89PNG"
    rows = call("/api/runs/demo/tsv")[1].decode().splitlines()
    assert rows[0].startswith("rank\tid") and rows[1].split("\t")[1] == "demo-2" and rows[1].endswith("demo/p0002_a.png")


def test_files_stay_inside_the_runs_folder(app):
    call, runs = app
    (runs.parent / "secret.txt").write_text("x")
    assert call("/files/../secret.txt")[0] in (403, 404)
    assert call("/files/%2e%2e/secret.txt")[0] == 403
    assert call("/files/.jobs/x")[0] == 403


def test_job_runs_logs_and_records_exit(app):
    call, runs = app
    j = J(call("/api/jobs", dict(kind="fake", out="demo", code="print('hello from job')")))
    j = wait(call, j["id"], ("done", "failed"))
    assert j["state"] == "done" and j["exit"] == 0 and any("hello from job" in l for l in j["log"])
    j = J(call("/api/jobs", dict(kind="fake", out="demo", code="import sys; sys.exit(3)")))
    assert wait(call, j["id"], ("done", "failed"))["state"] == "failed"
    assert J(call("/api/runs"))[0]["jobs"][0]["state"] == "failed"


def test_stop_and_resume(app):
    call, runs = app
    j = J(call("/api/jobs", dict(kind="fake", out="demo", code="import time; print('start', flush=True); time.sleep(60)")))
    wait(call, j["id"], ("running",))
    code, body = call("/api/jobs", dict(kind="fake", out="demo", code="import time; print('start', flush=True); time.sleep(60)"))
    assert code == 400 and b"already running" in body
    assert wait(call, J(call(f"/api/jobs/{j['id']}/stop", {}))["id"], ("stopped",))["state"] == "stopped"
    j = J(call(f"/api/jobs/{j['id']}/resume", {})); assert j["attempts"] == 2
    wait(call, j["id"], ("running",)); call(f"/api/jobs/{j['id']}/stop", {})
    log = wait(call, j["id"], ("stopped",))["log"]
    assert sum("=== attempt" in l for l in log) == 2


def test_bad_requests_refused(app):
    call, runs = app
    assert call("/api/jobs", dict(kind="rm", out="x"))[0] == 400
    assert call("/api/jobs", dict(kind="scan", scroll="PHerc0306B", box=[1, 2, 3], out="x"))[0] == 400
    assert call("/api/jobs", dict(kind="scan", scroll="PHerc0306B", box=[0] * 6, out="../x"))[0] == 400
    assert call("/api/jobs", dict(kind="scan", scroll="PHerc0306B", box=[0] * 6, out="x", device="rm -rf"))[0] == 400
    assert call("/api/jobs", dict(kind="fake", out="demo", code="1"), headers={"Origin": "https://evil.example"})[0] == 403
    assert call("/api/runs", headers={"Host": "evil.example"})[0] == 403
    assert call("/api/jobs/nope")[0] in (400, 404) and call("/api/runs/nope/patches")[0] == 404


def test_argv_matches_the_command_line(tmp_path):
    runs = str(tmp_path)
    a, out = joblib.argv_scan(runs, dict(scroll="PHerc0306B", box=[1, 2, 3, 4, 5, 6], out="b", shard="1/4", spacing=150))
    assert a[2:] == ["scan", "--scroll", "PHerc0306B", "--box", "1", "2", "3", "4", "5", "6", "--out", out, "--shard", "1/4", "--device", "cpu",
                     "--spacing", "150", "--half-nodes", "128", "--max-cache-gb", "2.0"] and out == os.path.join(runs, "b")
    a, out = joblib.argv_score(runs, dict(scroll="PHerc1545", surfaces=["s1"], out="hx", shard="2/3", tile=64))
    assert a[a.index("--out") + 1] == os.path.join(runs, "hx", "shard-2") and out == os.path.join(runs, "hx") and a[-1] == os.path.abspath("s1")
    J = joblib.Jobs(runs); j = J.create(dict(kind="score", scroll="PHerc1545", surfaces=["missing"], out="hx2"))
    t = time.time()
    while J.get(j["id"])["state"] == "running" and time.time() - t < 30: time.sleep(0.1)
    assert J.get(j["id"])["state"] == "failed" and "--resume" not in j["run_argv"]          # no such surface
    assert "--resume" in J.resume(j["id"])["run_argv"]                                   # resuming a score job keeps what is scored
    t = time.time()
    while J.get(j["id"])["state"] == "running" and time.time() - t < 30: time.sleep(0.1)
    d = os.path.join(runs, ".jobs", j["id"]); job = json.load(open(os.path.join(d, "job.json")))   # a job from before the cache cap
    job["argv"] = [v for v in job["argv"] if v not in ("--max-cache-gb", "2.0")]; json.dump(job, open(os.path.join(d, "job.json"), "w"))
    assert "--max-cache-gb" in J.resume(j["id"])["run_argv"]                             # resume rebuilds the command from the fields
    with pytest.raises(joblib.BadJob): joblib.argv_spread(runs, dict(scroll="x", seed=[1, 2], out="s"))


def test_stitched_sheets_listed_and_stitch_job(app):
    call, runs = app
    st = runs / "demo" / "_stitch"; st.mkdir()
    from PIL import Image
    Image.fromarray(blobs([(64, 64), (64, 160)])).save(st / "sheet1_ink.png"); Image.fromarray(blank()).save(st / "sheet1_back.png")
    (st / "sheets.json").write_text(json.dumps([dict(sheet=1, patches=["demo-2", "demo-1"], root="demo-2", height_mm=8.8, width_mm=8.8, area_cm2=0.7)]))
    (s,) = J(call("/api/runs/demo"))["sheets"]
    assert s["root"] == "demo-2" and isinstance(s["ls"], float) and call(s["ink"])[1][:4] == b"\x89PNG"
    j = J(call("/api/runs/demo/stitch", {}))
    assert j["kind"] == "stitch" and j["argv"][j["argv"].index("--out") + 1] == str(st) and j["argv"][2:4] == ["stitch", str(runs / "demo")]
    wait(call, j["id"], ("done", "failed"))
    assert J(call("/api/runs/demo"))["jobs"] == []                       # a stitch job is not the run's own job


def test_scan_runs_report_progress_from_box_and_shards(tmp_path):
    import runs as runlib
    run = tmp_path / "box"
    for i in (0, 1):
        _scan(str(run / f"shard-{i}"), ids=(i, i + 2) if i == 0 else (1,))
    box = dict(scroll="X", box=[0, 1000, 0, 500, 0, 500], spacing=250, half_nodes=128)  # 4 x 2 x 2 = 16 lattice seeds
    (run / "run.json").write_text(json.dumps(dict(box, shards=2)))
    rows, groups = runlib.status([str(run)])
    assert [(r["done"], r["expected"]) for r in rows] == [(2, 8), (1, 8)] and len(groups[0]["missing"]) == 13
    (run / "run.json").write_text(json.dumps(box))                     # older run.json: no shard count, done only
    rows, groups = runlib.status([str(run)])
    assert [(r["done"], r["expected"]) for r in rows] == [(2, None), (1, None)] and groups == []


def test_auto_runs_every_stage_on_an_existing_sweep(tmp_path):
    import auto
    src = tmp_path / "runs" / "scan"; _scan(str(src), ids=(1, 2, 3))
    for i in (1, 2, 3):                                                  # node grids: flat 32 x 32 sheets, 4 voxels apart
        g = np.stack(np.meshgrid(np.arange(32) * 4.0, np.arange(32) * 4.0, indexing="ij"), -1)
        P = np.concatenate([np.full((32, 32, 1), 1000.0 * i), g + 3000], -1).astype(np.float32)
        np.savez_compressed(src / f"p{i:04d}.npz", P=P, ok=np.ones((32, 32), bool), n0=np.array([1.0, 0, 0]))
    (src / "run.json").write_text(json.dumps(dict(scroll="PHerc0306B", um=8.64)))
    auto.main(["--scroll", "PHerc0306B", "--out", "t", "--runs", str(tmp_path / "runs"), "--sweep", str(src), "--spread", "0", "--leads", "2"])
    run = tmp_path / "runs" / "t-PHerc0306B"; A = json.loads((run / "auto" / "auto.json").read_text())
    assert all(A["stages"][k]["state"] == "done" for k in auto.STAGES), A["stages"]
    best = A["stages"]["leads"]["best"]; assert {l["id"] for l in best} == {"scan-2", "scan-3"} and best[0]["ts"] >= best[1]["ts"]   # 2 best of 3
    assert (run / "_viewer" / "index.html").exists() and len(list((run / "auto" / "vc" / "paths").iterdir())) == 2
    assert f"| 1 | {best[0]['id']} |" in (run / "auto" / "REPORT.md").read_text()
    auto.main(["--scroll", "PHerc0306B", "--out", "t", "--runs", str(tmp_path / "runs"), "--sweep", str(src), "--spread", "0"])   # resumes: all done
    s = server.App(str(tmp_path / "runs")); info = s.auto(str(run))
    assert info["report"] and [l["id"] for l in info["leads"]] == [l["id"] for l in best] and info["leads"][0]["png"].startswith("/files/")
