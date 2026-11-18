"""Lazy, chunk-cached reader for public OME-zarr v2 uint8 volumes with trilinear point sampling."""
import json, os, urllib.request, urllib.error, numpy as np
from concurrent.futures import ThreadPoolExecutor
BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"

class LazyVol:
    def __init__(self, path, level=0, base=BASE, cache_dir=None, threads=24):
        self.url = f"{base}{path}/{level}"
        meta = json.load(urllib.request.urlopen(f"{self.url}/.zarray"))
        self.shape = np.array(meta["shape"]); self.chunk = np.array(meta["chunks"])
        self.sep = meta.get("dimension_separator", "."); self.comp = meta["compressor"]
        self.codec = None
        if self.comp is not None:
            import numcodecs; self.codec = numcodecs.get_codec(self.comp)
        self.cache = {}; self.threads = threads; self.cache_dir = cache_dir; self.bytes = 0
        if cache_dir: os.makedirs(cache_dir, exist_ok=True)
        self.nch = (self.shape + self.chunk - 1) // self.chunk

    def _get(self, key):
        i, j, k = key
        fn = os.path.join(self.cache_dir, f"{i}_{j}_{k}.npy") if self.cache_dir else None
        if fn and os.path.exists(fn): return key, np.load(fn)
        url = f"{self.url}/{i}{self.sep}{j}{self.sep}{k}"; raw = None
        for _ in range(5):
            try: raw = urllib.request.urlopen(url, timeout=120).read(); break
            except urllib.error.HTTPError as e:
                if e.code in (403, 404): break
            except Exception: pass
        if raw is None: arr = None
        else:
            self.bytes += len(raw)
            if self.codec is not None: raw = self.codec.decode(raw)
            arr = np.frombuffer(raw, np.uint8).reshape(self.chunk)
        if fn: np.save(fn, arr if arr is not None else np.zeros(0, np.uint8))
        return key, arr

    def ensure(self, keys):
        need = [k for k in keys if k not in self.cache]
        if not need: return
        with ThreadPoolExecutor(self.threads) as ex:
            for key, arr in ex.map(self._get, need):
                self.cache[key] = None if (arr is None or arr.size == 0) else arr

    def _corner(self, idx):
        """idx: (3,N) int -> uint8 values (0 outside / missing)."""
        N = idx.shape[1]; out = np.zeros(N, np.float32)
        inb = np.all((idx >= 0) & (idx < self.shape[:, None]), axis=0)
        if not inb.any(): return out
        ii = idx[:, inb]; c = ii // self.chunk[:, None]; loc = ii - c * self.chunk[:, None]
        key = (c[0] * self.nch[1] + c[1]) * self.nch[2] + c[2]
        order = np.argsort(key, kind="stable"); ks = key[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1; starts = np.r_[0, cuts]; ends = np.r_[cuts, len(ks)]
        ukeys = [(int(k // (self.nch[1] * self.nch[2])), int((k // self.nch[2]) % self.nch[1]), int(k % self.nch[2])) for k in ks[starts]]
        self.ensure(ukeys)
        vals = np.zeros(len(ks), np.float32)
        for uk, s, e in zip(ukeys, starts, ends):
            ch = self.cache[uk]
            if ch is None: continue
            o = order[s:e]; vals[s:e] = ch[loc[0, o], loc[1, o], loc[2, o]]
        res = np.zeros(len(ks), np.float32); res[order] = vals
        out[inb] = res
        return out

    def sample(self, pts):
        """Trilinear sample at (3,N) float (z,y,x) points."""
        pts = np.asarray(pts, np.float32); f = np.floor(pts).astype(np.int64); t = pts - f
        out = np.zeros(pts.shape[1], np.float32)
        for dz in (0, 1):
            wz = t[0] if dz else 1 - t[0]
            for dy in (0, 1):
                wy = t[1] if dy else 1 - t[1]
                for dx in (0, 1):
                    wx = t[2] if dx else 1 - t[2]
                    out += wz * wy * wx * self._corner(f + np.array([dz, dy, dx])[:, None])
        return out

    def block(self, lo, hi):
        """Dense uint8 array for the voxel box [lo, hi) (clipped reads; zeros outside / missing)."""
        lo = np.asarray(lo, np.int64); hi = np.asarray(hi, np.int64); out = np.zeros(hi - lo, np.uint8)
        a = np.maximum(lo, 0); b = np.minimum(hi, self.shape)
        if np.any(b <= a): return out
        c0 = a // self.chunk; c1 = (b - 1) // self.chunk
        keys = [(i, j, k) for i in range(c0[0], c1[0] + 1) for j in range(c0[1], c1[1] + 1) for k in range(c0[2], c1[2] + 1)]
        self.ensure(keys)
        for key in keys:
            ch = self.cache[key]
            if ch is None: continue
            o = np.array(key) * self.chunk; s = np.maximum(a, o); e = np.minimum(b, o + self.chunk)
            out[s[0]-lo[0]:e[0]-lo[0], s[1]-lo[1]:e[1]-lo[1], s[2]-lo[2]:e[2]-lo[2]] = ch[s[0]-o[0]:e[0]-o[0], s[1]-o[1]:e[1]-o[1], s[2]-o[2]:e[2]-o[2]]
        return out

    def sample_dense(self, pts):
        """Trilinear sample via one dense bounding block (fast when the points are spatially compact)."""
        from scipy import ndimage as ndi
        pts = np.asarray(pts, np.float32); lo = np.floor(pts.min(1)).astype(np.int64) - 1; hi = np.ceil(pts.max(1)).astype(np.int64) + 2
        blk = self.block(lo, hi)
        return ndi.map_coordinates(blk, pts - lo[:, None].astype(np.float32), order=1, mode="constant", cval=0.0, output=np.float32)

    def drop(self, keep=0):
        """Forget cached chunks (memory)."""
        if len(self.cache) > keep: self.cache.clear()
