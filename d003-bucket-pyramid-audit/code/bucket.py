"""Minimal open-data bucket helpers for the D-003 audit: HTTP fetch with retries and S3 one-level listing."""
import re
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
