"""B-001s data: B-001's pipeline at larger scale for leave-one-scroll-out folds. Spec: experiments/b001s/README.md.

usage (B001_RESULTS must point at projects/vesuvius/results/b001s):
  b001s_data.py tiles        cloud, CPU: up to T_PER_SEG = 60 teacher-labelled tiles per PLACE_OK segment of all six
                             paired scrolls, except the labelled test segments -> results/b001s/tiles.json, tile_labels/
  b001s_data.py windows      cloud: B-001's 49 windows and labels, copied unchanged from results/b001
  b001s_data.py fetch        pod: LR crops of every tile and window, B001_WORKERS in parallel (default 8); resumable
  b001s_data.py status       counts per scroll and per fold
Placement: B-001's place/ results plus b001_data.place_one on the test scrolls' unlabelled segments (same rule).
"""
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'b001'))
import b001_data as bd                                              # noqa: E402

T_PER_SEG = 60
TEST_SCROLLS = ('PHerc0841', 'PHerc0009B', 'PHerc0500P2')
B001 = os.path.join(HERE, '..', '..', 'results', 'b001')


def guard():
    assert os.path.basename(os.path.normpath(bd.RESULTS)) == 'b001s', f'B001_RESULTS must be results/b001s, not {bd.RESULTS}'


def train_segments():
    S = bd.segs()['segments']
    return [k for k, v in S.items() if v['status'] == 'ok' and not (v['role'] == 'test' and v.get('labels'))]


def tiles():
    guard()
    bd.T_PER_SEG = T_PER_SEG
    names = sorted(train_segments())
    S = bd.segs()['segments']
    assert not any(S[n].get('labels') and S[n]['role'] == 'test' for n in names)
    bd.tiles(names)


def windows():
    guard()
    for f in ('windows.json',):
        shutil.copy(os.path.join(B001, f), os.path.join(bd.RESULTS, f))
    shutil.copytree(os.path.join(B001, 'window_labels'), os.path.join(bd.RESULTS, 'window_labels'), dirs_exist_ok=True)
    print('WINDOWS copied from results/b001:', len(json.load(open(os.path.join(bd.RESULTS, 'windows.json')))['windows']))


def fetch():
    guard()
    t0, jobs = time.time(), []
    for kind, f in (('windows', 'windows.json'), ('tiles', 'tiles.json')):
        reg = bd.jload(os.path.join(bd.RESULTS, f))
        jobs += [(kind, n, it) for n, it in sorted(reg[kind].items())]
    done = [0]

    def one(j):
        bd.fetch_one(*j)
        done[0] += 1
        if done[0] % 200 == 0:
            print(f'FETCH {done[0]}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)

    with ThreadPoolExecutor(int(os.environ.get('B001_WORKERS', '8'))) as ex:
        list(ex.map(one, jobs))
    have = sum(os.path.exists(os.path.join(bd.DATA, k, n + '.npy')) for k, n, _ in jobs)
    print(f'FETCH done: {have} of {len(jobs)} crops present {time.time() - t0:.0f}s', flush=True)
    if have != len(jobs):
        raise SystemExit('FETCH incomplete: rerun')


def status():
    guard()
    reg = bd.jload(os.path.join(bd.RESULTS, 'tiles.json'))
    per = {}
    for t in reg['tiles'].values():
        per[t['scroll']] = per.get(t['scroll'], 0) + 1
    print('TILES per scroll', json.dumps(per))
    for sc in TEST_SCROLLS:
        print(f'FOLD {sc}: {sum(v for k, v in per.items() if k != sc)} training tiles from {len(per) - (sc in per)} scrolls')


def main():
    a = sys.argv[1:]
    cmds = {'tiles': tiles, 'windows': windows, 'fetch': fetch, 'status': status}
    if len(a) != 1 or a[0] not in cmds:
        sys.exit(__doc__)
    cmds[a[0]]()


if __name__ == '__main__':
    main()
