"""V-013 reproduction control: mesh-as-published AUC for one segment on the March scan.
Same steps and parameters as march_bench.run() up to 'march_mesh_as_is'; skips the
2.399 reference and the flattened depth windows. CPU via INK_DEVICE=cpu."""
import json, os, subprocess, sys
import numpy as np
from scipy import ndimage
import march_bench as mb

seg = sys.argv[1]
d = os.path.join(mb.OUT, seg); os.makedirs(d, exist_ok=True)
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
_, gdy, gdx = best
lab8 = mb.down(lab); e0 = (r0 * 20 // mb.DS, c0 * 20 // mb.DS)
res = {'segment': seg, 'window_cells': [r0, r0 + mb.WIN_H, c0, c0 + mb.WIN_W], 'mesh_grid_offset': [gdy, gdx]}
json.dump(res, open(os.path.join(d, 'window.json'), 'w'), indent=1)
mar = os.path.join(d, 'march')
if not os.path.exists(mar + '.npy'):
    subprocess.check_call([mb.PY, os.path.join(mb.HERE, 'render_crop.py'), os.path.join(d, 'mesh_march'), mb.VOL_MARCH, mar,
                           '--rows', str(r0 + gdy), str(r0 + gdy + mb.WIN_H), '--cols', str(c0 + gdx), str(c0 + gdx + mb.WIN_W),
                           '--layers', '161', '--level', '1', '--oversample', '2', '--smooth-normals', '1.0',
                           '--cache-chunks', '1500'])
if not os.path.exists(mar + '_ink_o+0.npy'):
    subprocess.check_call([mb.PY, os.path.join(mb.HERE, 'infer_crop.py'), mar, '--offsets', '0'])
res['march_mesh_as_is'] = mb.eval_map(np.load(mar + '_ink_o+0.npy'), lab8, e0)
ref = json.load(open(os.path.join(mb.HERE, '..', 'results', 'C2_march_scan_8_segments.json'))) if os.path.exists(os.path.join(mb.HERE, '..', 'results', 'C2_march_scan_8_segments.json')) else {}
res['reporter'] = ref.get(seg, {}).get('march_mesh_as_is')
os.makedirs(os.environ.get('PUZZLE_RUN_DIR', '.') + '/payload', exist_ok=True)
json.dump(res, open(os.environ.get('PUZZLE_RUN_DIR', '.') + '/payload/v013-repro.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
