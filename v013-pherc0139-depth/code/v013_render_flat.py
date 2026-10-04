"""V-013 CT-only stage: for each segment, the march_bench window render (161 layers along
smoothed mesh normals) and refine_surface.py flattening (xcorr, 145 layers), exactly as
march_bench.run() does them. No ink model. Keeps march.npy and march_flat*.npy."""
import json, os, subprocess, sys
import numpy as np
from scipy import ndimage
import march_bench as mb


def window(seg, d):
    lab = mb.labels(seg, d)
    va = mb.mesh(seg, '20260319133554-2.403um', os.path.join(d, 'mesh_march'))
    vb = mb.mesh(seg, '20260102150214-2.399um', os.path.join(d, 'mesh_2399'))
    Hc, Wc = lab.shape[0] // 20, lab.shape[1] // 20
    cells = lab[:Hc * 20, :Wc * 20].reshape(Hc, 20, Wc, 20).mean((1, 3), dtype=np.float32)
    dens = ndimage.uniform_filter(cells, (mb.WIN_H, mb.WIN_W), mode='constant')
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    r0, c0 = int(np.clip(cy - mb.WIN_H // 2, 0, Hc - mb.WIN_H)), int(np.clip(cx - mb.WIN_W // 2, 0, Wc - mb.WIN_W))
    A, Bm = va.astype(np.float32), vb.astype(np.float32)
    best = (-1, 0, 0)
    Hh, Ww = min(A.shape[0], Bm.shape[0]) - 20, min(A.shape[1], Bm.shape[1]) - 20
    for dy in range(-10, 11):
        for dx in range(-10, 11):
            s = (A[10 + dy:10 + dy + Hh, 10 + dx:10 + dx + Ww] == Bm[10:10 + Hh, 10:10 + Ww]).mean()
            if s > best[0]:
                best = (s, dy, dx)
    return r0, c0, best[1], best[2]


summary = {}
for seg in sys.argv[1:]:
    d = os.path.join(mb.OUT, seg); os.makedirs(d, exist_ok=True)
    r0, c0, gdy, gdx = window(seg, d)
    summary[seg] = {'window_cells': [r0, r0 + mb.WIN_H, c0, c0 + mb.WIN_W], 'mesh_grid_offset': [gdy, gdx]}
    mar = os.path.join(d, 'march')
    if not os.path.exists(mar + '.npy'):
        subprocess.check_call([mb.PY, os.path.join(mb.HERE, 'render_crop.py'), os.path.join(d, 'mesh_march'), mb.VOL_MARCH, mar,
                               '--rows', str(r0 + gdy), str(r0 + gdy + mb.WIN_H), '--cols', str(c0 + gdx), str(c0 + gdx + mb.WIN_W),
                               '--layers', '161', '--level', '1', '--oversample', '2', '--smooth-normals', '1.0',
                               '--cache-chunks', '1500'])
    if not os.path.exists(mar + '_flat.npy'):
        subprocess.check_call([mb.PY, os.path.join(mb.HERE, 'refine_surface.py'), mar, '--out-layers', os.environ.get('V013_FLAT_LAYERS', '145')])
    print('done', seg, flush=True)
    os.makedirs(os.environ.get('PUZZLE_RUN_DIR', '.') + '/payload', exist_ok=True)
    json.dump(summary, open(os.environ.get('PUZZLE_RUN_DIR', '.') + '/payload/v013-render-flat.json', 'w'), indent=1)
