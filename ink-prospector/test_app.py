"""Tests for the app server and jobs (prospect.py serve). No network: jobs here are short python one-liners."""
import json, os, sys, threading, time, urllib.error, urllib.request
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
                     "--spacing", "150", "--half-nodes", "128"] and out == os.path.join(runs, "b")
    a, out = joblib.argv_score(runs, dict(scroll="PHerc1545", surfaces=["s1"], out="hx", shard="2/3", tile=64))
    assert a[a.index("--out") + 1] == os.path.join(runs, "hx", "shard-2") and out == os.path.join(runs, "hx") and a[-1] == os.path.abspath("s1")
    J = joblib.Jobs(runs); j = J.create(dict(kind="score", scroll="PHerc1545", surfaces=["missing"], out="hx2"))
    t = time.time()
    while J.get(j["id"])["state"] == "running" and time.time() - t < 30: time.sleep(0.1)
    assert J.get(j["id"])["state"] == "failed" and "--resume" not in j["run_argv"]          # no such surface
    assert "--resume" in J.resume(j["id"])["run_argv"]                                   # resuming a score job keeps what is scored
    with pytest.raises(joblib.BadJob): joblib.argv_spread(runs, dict(scroll="x", seed=[1, 2], out="s"))
