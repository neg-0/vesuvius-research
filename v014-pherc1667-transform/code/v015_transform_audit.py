"""V-015: audit every published volume transform JSON in the Vesuvius open-data bucket against its own landmarks.

For each `<sample>/volumes/<vol>.zarr/transform*.json`, the matrix (3x4, moving -> fixed, xyz) is applied to its
moving landmarks and compared with its fixed landmarks. Reported per transform:
  - published RMS / max landmark residual (fixed-volume voxels)
  - least-squares affine RMS on the same landmarks, and leave-one-out error (if >= 5 landmarks)
  - singular values of the linear part (scale check)
No CT data is read. Output: JSON + CSV in $PUZZLE_RUN_DIR/payload (default ./payload).
usage: v015_transform_audit.py
"""
import csv
import json
import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

B = 'https://vesuvius-challenge-open-data.s3.amazonaws.com'
RUN = os.environ.get('PUZZLE_RUN_DIR', '.')


def get(url, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            if i == tries - 1:
                raise


def listing(prefix):
    """(subprefixes, keys) directly under prefix, following continuation tokens."""
    pre, keys, tok = [], [], None
    while True:
        u = f'{B}/?list-type=2&delimiter=/&prefix={urllib.parse.quote(prefix)}'
        if tok:
            u += '&continuation-token=' + urllib.parse.quote(tok)
        x = get(u).decode()
        pre += [p for p in re.findall(r'<Prefix>([^<]+)</Prefix>', x) if p != prefix]
        keys += re.findall(r'<Key>([^<]+)</Key>', x)
        m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', x)
        if not m:
            return pre, keys
        tok = m.group(1)


def audit(key):
    d = json.loads(get(f'{B}/{key}'))
    r = {'key': key, 'fixed_volume': d.get('fixed_volume')}
    M = np.array(d.get('transformation_matrix', []), float)
    F = np.array(d.get('fixed_landmarks') or [], float)
    Mv = np.array(d.get('moving_landmarks') or [], float)
    if M.shape == (4, 4):
        M = M[:3]
    r['n_landmarks'] = int(len(F))
    if M.shape != (3, 4):
        r['status'] = 'no 3x4 matrix'
        return r
    r['singular_values'] = np.linalg.svd(M[:, :3])[1].tolist()
    if len(F) == 0 or len(F) != len(Mv):
        r['status'] = 'no landmarks' if len(F) == 0 else 'landmark count mismatch'
        return r
    H = np.c_[Mv, np.ones(len(Mv))]
    res = np.linalg.norm(H @ M.T - F, axis=1)
    r['pub_rms'] = float(np.sqrt((res ** 2).mean())); r['pub_max'] = float(res.max())
    if len(F) >= 4:
        X = np.linalg.lstsq(H, F, rcond=None)[0].T
        rl = np.linalg.norm(H @ X.T - F, axis=1)
        r['lsq_rms'] = float(np.sqrt((rl ** 2).mean()))
        r['lsq_singular_values'] = np.linalg.svd(X[:, :3])[1].tolist()
    if len(F) >= 5:
        loo = []
        for i in range(len(F)):
            m = np.ones(len(F), bool); m[i] = False
            Xi = np.linalg.lstsq(H[m], F[m], rcond=None)[0]
            loo.append(float(np.linalg.norm(H[i] @ Xi - F[i])))
        r['lsq_loo_rms'] = float(np.sqrt(np.mean(np.square(loo))))
    r['status'] = 'ok'
    return r


def main():
    samples, _ = listing('')
    vols = []
    for s in samples:
        sub, _ = listing(s)
        if s + 'volumes/' in sub:
            vols += listing(s + 'volumes/')[0]
    with ThreadPoolExecutor(16) as ex:
        keys = [k for _, ks in ex.map(listing, vols) for k in ks if re.search(r'/transform[^/]*\.json$', k)]
        rows = list(ex.map(audit, sorted(keys)))
    os.makedirs(os.path.join(RUN, 'payload'), exist_ok=True)
    json.dump({'volumes_scanned': len(vols), 'transforms': rows}, open(os.path.join(RUN, 'payload', 'v015-transform-audit.json'), 'w'), indent=1)
    cols = ['key', 'fixed_volume', 'n_landmarks', 'status', 'pub_rms', 'pub_max', 'lsq_rms', 'lsq_loo_rms']
    with open(os.path.join(RUN, 'payload', 'v015-transform-audit.csv'), 'w', newline='') as h:
        w = csv.DictWriter(h, cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print('volumes', len(vols), 'transforms', len(rows))
    for r in sorted(rows, key=lambda r: -(r.get('pub_rms') or 0)):
        print(f"{r.get('pub_rms', float('nan')):9.2f} {r.get('lsq_rms', float('nan')):8.2f} {r.get('lsq_loo_rms', float('nan')):8.2f} n={r['n_landmarks']:3d} {r['status']:12s} {r['key']}")


if __name__ == '__main__':
    main()
