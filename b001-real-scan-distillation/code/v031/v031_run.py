"""Open-data bucket access helpers, trimmed for this repository to what the B-001 drivers import: one-level S3
listing, fetch with retries, and TIFF shape over HTTP range reads. Function bodies are unchanged from the version
that ran.
"""
import io
import time
import urllib.error
import urllib.parse
import urllib.request

B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'

def fetch(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except Exception:                                              # noqa: BLE001  (resets, IncompleteRead)
            if t == tries - 1:
                raise
            time.sleep(2 ** (t + 1))


def ls(prefix):
    """S3 listing (one level): (sub-prefixes, [(key, size)])."""
    import re
    ps, ks, tok = [], [], None
    while True:
        q = {'list-type': '2', 'prefix': prefix, 'delimiter': '/'}
        if tok:
            q['continuation-token'] = tok
        x = fetch(B + '?' + urllib.parse.urlencode(q)).decode()
        ps += [p for p in re.findall(r'<Prefix>([^<]*)</Prefix>', x) if p != prefix]
        ks += [(m.group(1), int(m.group(2))) for m in re.finditer(r'<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>', x)]
        m = re.search(r'<NextContinuationToken>([^<]*)</NextContinuationToken>', x)
        if not m:
            return ps, ks
        tok = m.group(1)


class RangeFile(io.RawIOBase):
    """Read-only file over HTTP range requests (tiff headers and tiles without downloading the file)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60)
        self.size = int(r.headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        rq = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}'})
        for t in range(5):
            try:
                d = urllib.request.urlopen(rq, timeout=120).read()
                break
            except Exception:                                          # noqa: BLE001
                if t == 4:
                    raise
                time.sleep(2 ** (t + 1))
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def tif_shape(url):
    import tifffile
    with tifffile.TiffFile(RangeFile(url)) as t:
        p = t.pages[0]
        return list(p.shape), str(p.dtype), int(p.compression), bool(p.is_tiled)
