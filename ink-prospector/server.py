"""Ink Prospector app server: a JSON API over the run folders and the command line, plus the web UI (app/index.html).

  prospect.py serve --runs runs [--port 8765] [--host 127.0.0.1]

Standard library only. It starts processes on this machine, so it listens on localhost unless told otherwise.
API and layout: docs/APP.md.
"""
import json, mimetypes, os, sys, threading, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import runs as runlib
import jobs as joblib


class App:
    def __init__(self, runs, home=None):
        self.runs = os.path.abspath(runs); os.makedirs(self.runs, exist_ok=True)
        self.jobs = joblib.Jobs(self.runs, home); self.cache = {}; self.lock = threading.Lock()

    def _stamp(self, d):
        return tuple((f, os.path.getmtime(f), os.path.getsize(f)) for f in (os.path.join(s, "scores.tsv") for s in runlib.scan_dirs([d])))

    def run_names(self):
        out = []
        for n in sorted(os.listdir(self.runs)):
            d = os.path.join(self.runs, n)
            if n[0] in "._" or not os.path.isdir(d): continue
            if runlib.scan_dirs([d]) or any(j["out"] == d for j in self.jobs.list()): out.append(n)
        return out

    def patches(self, name):
        d = joblib.inside(self.runs, name)
        if not os.path.isdir(d): raise KeyError(name)
        st = self._stamp(d)
        with self.lock:
            hit = self.cache.get(d)
            if hit and hit[0] == st: return hit[1]
        P = runlib.patches([d])
        for p in P:
            L = runlib.letters(p) or {}; p["ls"] = L.get("score"); p["lrow"] = L.get("row"); p["letters"] = L.get("letters")
        with self.lock: self.cache[d] = (st, P)
        return P

    def url(self, path): return "/files/" + urllib.parse.quote(os.path.relpath(path, self.runs))

    def run_info(self, name):
        d = joblib.inside(self.runs, name)
        if not os.path.isdir(d) and any(j["out"] == d for j in self.jobs.list()): os.makedirs(d, exist_ok=True)   # a job just started on it
        P = self.patches(name)
        rows, groups = runlib.status([d]) if runlib.scan_dirs([d]) else ([], [])
        done = sum(r["done"] for r in rows); exp = [r["expected"] for r in rows]
        js = [j for j in self.jobs.list() if j["out"] == d and j["kind"] not in ("viewer", "stitch")]
        m = runlib.meta(runlib.scan_dirs([d])[0]) if runlib.scan_dirs([d]) else {}
        return dict(name=name, patches=len(P), area=round(sum(p["area"] for p in P), 2), best=round(P[0]["ts"], 4) if P else None,
                    scored=done, expected=sum(exp) if exp and None not in exp else None, shards=len(rows),
                    missing=sum(len(g["missing"]) for g in groups), scroll=m.get("scroll"), source=m.get("source", "scan"),
                    viewer=self.url(os.path.join(d, "_viewer", "index.html")) if os.path.exists(os.path.join(d, "_viewer", "index.html")) else None,
                    jobs=[dict(id=j["id"], state=j["state"], kind=j["kind"]) for j in js], sheets=self.sheets(d), auto=self.auto(d))

    def auto(self, d):
        """Auto mode state (auto.py): stages, leads with their maps, spreads, report and the VC3D folder."""
        f = os.path.join(d, "auto", "auto.json")
        if not os.path.exists(f): return None
        try: A = json.load(open(f))
        except json.JSONDecodeError: return None                        # being rewritten
        S = A.get("stages", {}); sp = {x["id"]: x for x in S.get("spread", {}).get("spreads", [])}
        url = lambda p: self.url(p) if p and os.path.realpath(p).startswith(os.path.realpath(self.runs) + os.sep) and os.path.exists(p) else None
        leads = [dict(l, png=url(l.get("png")), spread=dict(sp[l["id"]], png=url(sp[l["id"]].get("png"))) if l["id"] in sp else None)
                 for l in S.get("leads", {}).get("best", [])]
        rep = os.path.join(d, "auto", "REPORT.md"); vc = os.path.join(d, "auto", "vc")
        return dict(stages={k: dict(state=v.get("state"), secs=v.get("secs"), error=v.get("error")) for k, v in S.items()}, leads=leads,
                    sweep=S.get("sweep", {}), report=self.url(rep) if os.path.exists(rep) else None, vc=vc if os.path.isdir(vc) else None)

    def sheets(self, d):
        """Stitched sheets of a run (prospect.py stitch into <run>/_stitch), with image URLs and letterscore on each."""
        f = os.path.join(d, "_stitch", "sheets.json")
        if not os.path.exists(f): return []
        um = (runlib.meta(runlib.scan_dirs([d])[0]) if runlib.scan_dirs([d]) else {}).get("um") or 8.64; out = []
        for s in json.load(open(f)):
            base = os.path.join(d, "_stitch", f"sheet{s['sheet']}")
            if not os.path.exists(base + "_ink.png"): continue
            L = runlib.letters(dict(base=base, side="ink", um=um), faces=("ink", "back")) or {}
            out.append(dict(s, ink=self.url(base + "_ink.png"), back=self.url(base + "_back.png"), ls=L.get("score"), lrow=L.get("row"), letters=L.get("letters")))
        return out

    def tsv(self, name):
        """The ranked table of `prospect.py rank --tsv`, with PNG paths relative to the runs folder."""
        import tempfile
        with tempfile.NamedTemporaryFile("r", suffix=".tsv") as f:
            P = [dict(p, base=os.path.relpath(p["base"], self.runs)) for p in self.patches(name)]
            runlib.write_tsv(P, f.name); return f.read().encode()

    def patch_rows(self, name, n):
        out = []
        for i, p in enumerate(self.patches(name)[:n]):
            out.append(dict(rank=i + 1, id=p["id"], ts=round(p["ts"], 4), side=p["side"], fa=round(p["fa"], 4), fb=round(p["fb"], 4),
                            area=round(p["area"], 3), ls=p.get("ls"), lrow=p.get("lrow"), letters=p.get("letters"), tracked=round(p["tracked"], 3), seed=p["seed"], src=p.get("src", ""),
                            a=self.url(p["base"] + "_a.png"), b=self.url(p["base"] + "_b.png")))
        return out


class Handler(BaseHTTPRequestHandler):
    app = None
    server_version = "InkProspector"

    def log_message(self, fmt, *a): pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes): body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def file(self, path):
        if not os.path.isfile(path): return self.send(404, dict(error="not found"))
        self.send(200, open(path, "rb").read(), mimetypes.guess_type(path)[0] or "application/octet-stream")

    def route(self, method):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query); parts = [urllib.parse.unquote(p) for p in u.path.strip("/").split("/") if p]
        A = self.app
        if method == "GET" and not parts: return self.file(os.path.join(HERE, "app", "index.html"))
        if method == "GET" and parts[0] == "files":
            rel = "/".join(parts[1:]); f = os.path.realpath(os.path.join(A.runs, rel))
            if not rel or not f.startswith(os.path.realpath(A.runs) + os.sep) or "/.jobs/" in f + "/": return self.send(403, dict(error="outside the runs folder"))
            return self.file(f)
        if parts[0] != "api": return self.send(404, dict(error="not found"))
        r = parts[1:]
        if method == "GET" and r == ["scrolls"]: return self.send(200, json.load(open(os.path.join(HERE, "scrolls.json"))))
        if method == "GET" and r == ["runs"]: return self.send(200, [A.run_info(n) for n in A.run_names()])
        if len(r) >= 2 and r[0] == "runs":
            name = "/".join(r[1:-1]) if r[-1] in ("patches", "viewer", "tsv", "stitch") else "/".join(r[1:])
            if method == "GET" and r[-1] == "patches": return self.send(200, A.patch_rows(name, int(q.get("n", ["200"])[0])))
            if method == "GET" and r[-1] == "tsv": return self.send(200, A.tsv(name), "text/tab-separated-values; charset=utf-8")
            if method == "POST" and r[-1] == "stitch":
                return self.send(200, A.jobs.create(dict(kind="stitch", runs=[name], out=name + "/_stitch", top=self.body().get("top", 30))))
            if method == "POST" and r[-1] == "viewer": return self.send(200, A.jobs.create(dict(kind="viewer", runs=[name], **self.body())))
            if method == "GET": return self.send(200, A.run_info(name))
        if r == ["jobs"]:
            if method == "GET": return self.send(200, A.jobs.list())
            if method == "POST": return self.send(200, A.jobs.create(self.body()))
        if len(r) == 2 and r[0] == "jobs" and method == "GET": return self.send(200, A.jobs.get(r[1], tail=int(q.get("tail", ["200"])[0])))
        if len(r) == 3 and r[0] == "jobs" and method == "POST" and r[2] in ("stop", "resume"):
            return self.send(200, getattr(A.jobs, r[2])(r[1]))
        return self.send(404, dict(error="not found"))

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n: return {}
        b = json.loads(self.rfile.read(n))
        if not isinstance(b, dict): raise joblib.BadJob("expected a JSON object")
        return b

    def handle_one(self, method):
        try: self.route(method)
        except joblib.BadJob as e: self.send(400, dict(error=str(e)))
        except KeyError as e: self.send(404, dict(error=f"no such run or job: {e.args[0]}"))
        except json.JSONDecodeError as e: self.send(400, dict(error=f"bad JSON: {e}"))
        except Exception as e: self.send(500, dict(error=f"{type(e).__name__}: {e}"))

    def host_ok(self):
        """On a loopback address, answer only requests addressed to localhost (blocks DNS rebinding from web pages)."""
        if self.server.server_address[0] not in ("127.0.0.1", "::1", "localhost"): return True
        h = urllib.parse.urlparse("//" + (self.headers.get("Host") or "")).hostname
        return h in ("localhost", "127.0.0.1", "::1")

    def do_GET(self):
        if not self.host_ok(): return self.send(403, dict(error="bad Host"))
        self.handle_one("GET")

    def do_POST(self):
        if not self.host_ok(): return self.send(403, dict(error="bad Host"))
        if self.headers.get("Origin") and urllib.parse.urlparse(self.headers["Origin"]).netloc != self.headers.get("Host"):
            return self.send(403, dict(error="cross-origin request refused"))   # a web page elsewhere must not start jobs here
        self.handle_one("POST")


def make_server(runs, host="127.0.0.1", port=8765, home=None):
    h = type("H", (Handler,), dict(app=App(runs, home)))
    return ThreadingHTTPServer((host, port), h)


def main(runs, host="127.0.0.1", port=8765):
    s = make_server(runs, host, port)
    print(f"Ink Prospector app: http://{host}:{s.server_address[1]}/  (runs folder {os.path.abspath(runs)}; Ctrl-C to stop; running jobs keep going)", flush=True)
    try: s.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(); main(a.runs, a.host, a.port)
