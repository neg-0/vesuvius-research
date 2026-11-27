"""Jobs for the Ink Prospector app: each job is one command line process, stored on disk so it outlives the server.

<runs>/.jobs/<id>/job.json   kind, fields, argv, started, attempts
<runs>/.jobs/<id>/log.txt    the process's output (appended on resume)
<runs>/.jobs/<id>/pid        pid of the wrapper (a session leader; the command runs in its process group)
<runs>/.jobs/<id>/exit       exit code, written by the wrapper when the command ends
<runs>/.jobs/<id>/stop       present when a stop was asked for

The wrapper (`python jobs.py _run <jobdir>`) runs the command and writes its exit code, so a job keeps running and
still records how it ended if the server is stopped or restarted. Argument lists are built from typed fields per job
kind (ARGV); nothing is passed through a shell.
"""
import json, os, re, signal, subprocess, sys, time, uuid
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


class BadJob(ValueError):
    pass


def inside(runs, name):
    """Absolute path of a run folder name under runs; refuses absolute paths, '..' and hidden names."""
    if not isinstance(name, str) or not name or os.path.isabs(name) or any(p in ("", ".", "..") or p.startswith(".") for p in name.split("/")):
        raise BadJob(f"bad run name {name!r}: use a relative folder name inside the runs folder")
    return os.path.join(runs, name)


def _num(f, k, typ=int, default=None, n=None):
    v = f.get(k, default)
    if v is None or v == "": return default
    try:
        return [typ(x) for x in v] if n else typ(v)
    except (TypeError, ValueError):
        raise BadJob(f"{k}: expected {'a list of ' if n else ''}{typ.__name__}, got {v!r}")


def _shard(f):
    s = str(f.get("shard") or "0/1")
    try:
        i, n = (int(v) for v in s.split("/"))
    except ValueError:
        raise BadJob(f"shard {s!r}: expected i/n")
    if not 0 <= i < n: raise BadJob(f"shard {s}: need 0 <= i < n")
    return s, i, n


def _scroll(f, need_m7=True):
    out = []
    for k in ("scroll", "ct") + (("m7",) if need_m7 else ()):
        if f.get(k): out += ["--" + k, str(f[k])]
    if f.get("um"): out += ["--um", str(_num(f, "um", float))]
    if not (f.get("scroll") or f.get("ct")): raise BadJob("name a scroll from scrolls.json, or give ct (and m7 for a scan)")
    return out


def _device(f):
    d = f.get("device") or "cpu"
    if d not in ("cpu", "cuda", "mps"): raise BadJob(f"device {d!r}: cpu, cuda or mps")
    return ["--device", d]


def _cache(f):
    """Chunk cache cap per process: 2 GB unless the job names one (0 = uncapped). Uncapped, one scan passed 6 GB and
    two shards plus a score job got OOM-killed on a 16 GB machine."""
    g = _num(f, "max_cache_gb", float, 2.0)
    return ["--max-cache-gb", str(g)] if g else []


def argv_scan(runs, f):
    box = _num(f, "box", int, n=6)
    if not box or len(box) != 6: raise BadJob("box: six integers z0 z1 y0 y1 x0 x1")
    out = inside(runs, f.get("out")); s, _, _ = _shard(f)
    a = [PY, os.path.join(HERE, "prospect.py"), "scan", *_scroll(f), "--box", *map(str, box), "--out", out, "--shard", s, *_device(f),
         "--spacing", str(_num(f, "spacing", int, 250)), "--half-nodes", str(_num(f, "half_nodes", int, 128))]
    if f.get("max_patches"): a += ["--max-patches", str(_num(f, "max_patches"))]
    a += _cache(f)
    if f.get("fuse"): a.append("--fuse")
    return a, out


def argv_score(runs, f):
    surf = f.get("surfaces")
    if isinstance(surf, str): surf = surf.split()
    if not surf or not all(isinstance(p, str) and p for p in surf): raise BadJob("surfaces: one or more tifxyz folders")
    run = inside(runs, f.get("out")); s, i, n = _shard(f); out = run if n == 1 else os.path.join(run, f"shard-{i}")
    a = [PY, os.path.join(HERE, "score_tifxyz.py"), *_scroll(f, need_m7=False), "--out", out, "--shard", s, *_device(f)]
    if f.get("tile"): a += ["--tile", str(_num(f, "tile"))]
    if f.get("min_nodes"): a += ["--min-nodes", str(_num(f, "min_nodes"))]
    if f.get("crop"):
        c = _num(f, "crop", int, n=4)
        if len(c) != 4: raise BadJob("crop: four integers r0 r1 c0 c1")
        a += ["--crop", *map(str, c)]
    a += _cache(f)
    if f.get("fuse"): a.append("--fuse")
    return a + [os.path.abspath(p) for p in surf], run


def argv_spread(runs, f):
    seed = _num(f, "seed", float, n=3)
    if not seed or len(seed) != 3: raise BadJob("seed: three numbers z y x")
    side = f.get("side") or "a"
    if side not in ("a", "b"): raise BadJob("side: a or b")
    out = inside(runs, f.get("out"))
    return [PY, os.path.join(HERE, "prospect.py"), "spread", *_scroll(f), "--seed", *map(str, seed), "--side", side, "--out", out,
            "--half-nodes", str(_num(f, "half_nodes", int, 320))], out


def _run_list(runs, f):
    r = f.get("runs")
    if isinstance(r, str): r = [r]
    if not r: raise BadJob("runs: one or more run folder names")
    return [inside(runs, x) for x in r]


def argv_stitch(runs, f):
    src = _run_list(runs, f); out = inside(runs, f.get("out"))
    return [PY, os.path.join(HERE, "prospect.py"), "stitch", *src, "--out", out, "--top", str(_num(f, "top", int, 30))], out


def argv_viewer(runs, f):
    src = _run_list(runs, f); out = os.path.join(src[0], "_viewer") if not f.get("out") else inside(runs, f["out"])
    return [PY, os.path.join(HERE, "prospect.py"), "viewer", *src, "--out", out, "--title", str(f.get("title") or os.path.basename(src[0])),
            "--full", str(_num(f, "full", int, 50))], src[0]


def argv_auto(runs, f):
    sc = f.get("scrolls") or f.get("scroll")
    if isinstance(sc, str): sc = sc.split()
    if not sc and not (f.get("ct") and f.get("m7")): raise BadJob("scrolls: one or more names from scrolls.json (or ct and m7)")
    name = f.get("out"); inside(runs, name)
    a = [PY, os.path.join(HERE, "auto.py"), "--out", name, "--runs", runs, *_device(f)]
    if sc: a += ["--scroll", *map(str, sc)]
    for k in ("ct", "m7"):
        if f.get(k): a += ["--" + k, str(f[k])]
    if f.get("box"):
        b = _num(f, "box", int, n=6)
        if len(b) != 6: raise BadJob("box: six integers z0 z1 y0 y1 x0 x1")
        a += ["--box", *map(str, b)]
    for k, d in (("spacing", 800), ("workers", 2), ("leads", 10), ("spread", 3)):
        a += ["--" + k, str(_num(f, k, int, d))]
    a += _cache(f)
    if f.get("sweep"): a += ["--sweep", inside(runs, f["sweep"])]
    if _num(f, "grow", int, 0):
        a += ["--grow", str(_num(f, "grow", int))]
        img = f.get("vc_image")                                         # the API takes an image name, not a command line
        if img:
            if not isinstance(img, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+)?(@sha256:[0-9a-f]{64})?", img):
                raise BadJob("vc_image: a Docker image name such as ghcr.io/scrollprize/villa/volume-cartographer:edge")
            a += ["--tracer", f"docker run --rm --network host -e HTTPS_PROXY -e HTTP_PROXY -e NO_PROXY -v {{dir}}:{{dir}} -w {{dir}} {img} vc_grow_seg_from_seed"]
        if f.get("snap"): a.append("--snap")
    return a, os.path.join(runs, f"{name}-{sc[0] if sc else 'scroll'}")


ARGV = dict(auto=argv_auto, scan=argv_scan, score=argv_score, spread=argv_spread, stitch=argv_stitch, viewer=argv_viewer)


def _alive(pid):
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Jobs:
    def __init__(self, runs, home=None):
        self.runs = os.path.abspath(runs); self.dir = os.path.join(self.runs, ".jobs"); os.makedirs(self.dir, exist_ok=True)
        self.home = os.path.abspath(home or os.environ.get("PROSPECTOR_HOME", "."))
        self.procs = {}                                                  # id -> Popen of wrappers this server started (reaped on poll)

    def _d(self, jid):
        if not isinstance(jid, str) or not jid.isalnum(): raise BadJob(f"bad job id {jid!r}")
        d = os.path.join(self.dir, jid)
        if not os.path.isdir(d): raise KeyError(jid)
        return d

    def create(self, fields):
        kind = fields.get("kind")
        if kind not in ARGV: raise BadJob(f"kind {kind!r}: one of {', '.join(ARGV)}")
        argv, out = ARGV[kind](self.runs, fields)
        if kind != "viewer":
            busy = [j for j in self.list() if j["state"] == "running" and j["out"] == out and j["argv"] == argv]
            if busy: raise BadJob(f"job {busy[0]['id']} is already running this command")
        jid = time.strftime("%Y%m%d%H%M%S") + f"{time.time_ns() % 10**9:09d}" + uuid.uuid4().hex[:4]   # sorts by start time
        d = os.path.join(self.dir, jid); os.makedirs(d)
        job = dict(id=jid, kind=kind, fields=fields, argv=argv, out=out, created=time.time(), attempts=0)
        json.dump(job, open(os.path.join(d, "job.json"), "w"), indent=1)
        return self.start(jid)

    def start(self, jid, resume=False):
        d = self._d(jid); job = json.load(open(os.path.join(d, "job.json")))
        if self.get(jid)["state"] == "running": raise BadJob(f"job {jid} is running")
        argv = list(job["argv"])
        if resume:                                                      # rebuild from the fields, so fixes since the job began apply
            try: argv = ARGV[job["kind"]](self.runs, job["fields"])[0]
            except (BadJob, KeyError): pass
        if resume and job["kind"] == "score" and "--resume" not in argv: argv.insert(2, "--resume")
        for f in ("exit", "stop", "pid"):
            if os.path.exists(os.path.join(d, f)): os.remove(os.path.join(d, f))
        job["attempts"] += 1; job["started"] = time.time(); job["run_argv"] = argv
        json.dump(job, open(os.path.join(d, "job.json"), "w"), indent=1)
        with open(os.path.join(d, "log.txt"), "a") as log:
            log.write(f"=== attempt {job['attempts']} {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(argv)}\n")
        p = subprocess.Popen([PY, os.path.abspath(__file__), "_run", d], cwd=self.home, start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        open(os.path.join(d, "pid"), "w").write(str(p.pid)); self.procs[jid] = p
        return self.get(jid)

    def resume(self, jid): return self.start(jid, resume=True)

    def stop(self, jid):
        d = self._d(jid); j = self.get(jid)
        if j["state"] != "running": return j
        open(os.path.join(d, "stop"), "w").write(str(time.time()))
        try: os.killpg(j["pid"], signal.SIGTERM)
        except ProcessLookupError: pass
        return self.get(jid)

    def get(self, jid, tail=0):
        d = self._d(jid); job = json.load(open(os.path.join(d, "job.json")))
        if jid in self.procs and self.procs[jid].poll() is not None: del self.procs[jid]
        rd = lambda f: open(os.path.join(d, f)).read().strip() if os.path.exists(os.path.join(d, f)) else None
        pid, code, stop = rd("pid"), rd("exit"), rd("stop")
        job["pid"] = int(pid) if pid else None; job["exit"] = int(code) if code not in (None, "") else None
        if job["exit"] is not None: job["state"] = "done" if job["exit"] == 0 else ("stopped" if stop else "failed")
        elif job["pid"] and _alive(job["pid"]): job["state"] = "running"
        else: job["state"] = "stopped" if job["attempts"] else "new"     # process gone without an exit code (killed, machine restarted)
        if tail:
            with open(os.path.join(d, "log.txt"), "rb") as f:
                f.seek(0, 2); f.seek(max(0, f.tell() - 200 * tail))
                job["log"] = f.read().decode("utf-8", "replace").splitlines()[-tail:]
        return job

    def list(self):
        out = []
        for jid in sorted(os.listdir(self.dir), reverse=True):
            try: out.append(self.get(jid))
            except (KeyError, BadJob, FileNotFoundError, json.JSONDecodeError): pass
        return sorted(out, key=lambda j: -j["created"])                  # newest first, also for jobs made in the same second


def _wrapper(d):
    """Run the job's command in this process group, append its output to log.txt, write its exit code."""
    job = json.load(open(os.path.join(d, "job.json")))
    with open(os.path.join(d, "log.txt"), "ab") as log:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.Popen(job.get("run_argv") or job["argv"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        signal.signal(signal.SIGTERM, lambda *_: p.terminate())
        code = p.wait()
        log.write(f"=== exit {code}\n".encode())
    open(os.path.join(d, "exit"), "w").write(str(code))


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "_run":
    _wrapper(sys.argv[2])
