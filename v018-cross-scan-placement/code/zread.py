"""Minimal zarr v2 reader over HTTPS (blosc or raw chunks), no zarr dependency."""
import json, sys, itertools, urllib.request, concurrent.futures as cf
import numpy as np, numcodecs

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"

def _get(url, tries=5):
    import time, http.client
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if t == tries - 1:
                raise
        except (urllib.error.URLError, http.client.IncompleteRead, ConnectionError, TimeoutError):
            if t == tries - 1:
                raise
        time.sleep(2 ** t)

def meta(path):
    return json.loads(_get(BUCKET + path.rstrip("/") + "/.zarray"))

def read(path, lo, hi, workers=16):
    """Read array[lo:hi] (3D, voxel coords at this level)."""
    m = meta(path); cs = m["chunks"]; dt = np.dtype(m["dtype"]); sep = m.get("dimension_separator", ".")
    comp = numcodecs.get_codec(m["compressor"]) if m["compressor"] else None
    lo = [max(0, a) for a in lo]; hi = [min(s, b) for s, b in zip(m["shape"], hi)]
    out = np.full([b - a for a, b in zip(lo, hi)], m["fill_value"] or 0, dt)
    idx = [range(a // c, (b - 1) // c + 1) for a, b, c in zip(lo, hi, cs)]
    def job(ci):
        raw = _get(BUCKET + path.rstrip("/") + "/" + sep.join(map(str, ci)))
        if raw is None:
            return ci, None
        buf = comp.decode(raw) if comp else raw
        return ci, np.frombuffer(buf, dt).reshape(cs)
    with cf.ThreadPoolExecutor(workers) as ex:
        for ci, a in ex.map(job, itertools.product(*idx)):
            if a is None:
                continue
            c0 = [i * c for i, c in zip(ci, cs)]
            s = tuple(slice(max(l, o) - o, min(h, o + c) - o) for l, h, o, c in zip(lo, hi, c0, cs))
            d = tuple(slice(max(l, o) - l, min(h, o + c) - l) for l, h, o, c in zip(lo, hi, c0, cs))
            out[d] = a[s]
    return out
