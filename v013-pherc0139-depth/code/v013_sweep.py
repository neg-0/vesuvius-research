"""V-013 fine depth sweep (GPU). For each segment: march_bench window render (161 layers),
refine_surface flatten to V013_FLAT_LAYERS=161 layers, then ink_canonical_2um at 62-layer
windows centred at offsets -48..+48 step 4 layers (~9.6 um). Per offset: full AUC vs labels
(march_bench.eval_map, label offset searched as in the reporter's benchmark) and the ds8 map.
Writes <run>/payload/v013-sweep.json and <run>/payload/<seg>_ds8_maps.npz (float16).
Checkpoint: per-offset ink maps are cached on disk; reruns skip finished offsets."""
import json, os, subprocess, sys
import numpy as np
import march_bench as mb

OFFS = list(range(-48, 49, 4))
RUN = os.environ.get('PUZZLE_RUN_DIR', '.')
os.makedirs(RUN + '/payload', exist_ok=True)
os.environ.setdefault('V013_FLAT_LAYERS', '161')
out = {}
for seg in sys.argv[1:]:
    subprocess.check_call([mb.PY, 'v013_render_flat.py', seg])
    d = os.path.join(mb.OUT, seg); mar = os.path.join(d, 'march')
    win = json.load(open(os.path.join(RUN, 'payload', 'v013-render-flat.json')))[seg]
    r0, c0 = win['window_cells'][0], win['window_cells'][2]
    lab8 = mb.down(mb.labels(seg, d)); e0 = (r0 * 20 // mb.DS, c0 * 20 // mb.DS)
    need = [o for o in OFFS if not os.path.exists(mar + f'_flat_ink_o{o:+d}.npy')]
    if need:
        subprocess.check_call([mb.PY, 'infer_crop.py', mar + '_flat', '--offsets'] + [str(o) for o in need])
    res, maps = {}, {}
    for o in OFFS:
        ink = np.load(mar + f'_flat_ink_o{o:+d}.npy')
        res[str(o)] = mb.eval_map(ink, lab8, e0)
        maps[f'o{o:+d}'] = mb.down(ink).astype(np.float16)
    np.savez_compressed(os.path.join(RUN, 'payload', f'{seg}_ds8_maps.npz'), lab8_window=lab8.astype(np.float16), e0=np.array(e0), **maps)
    np.save(os.path.join(RUN, 'payload', f'{seg}_flat_d.npy'), np.load(mar + '_flat_d.npy'))
    out[seg] = {'window': win, 'auc_by_offset_layers': res}
    json.dump(out, open(os.path.join(RUN, 'payload', 'v013-sweep.json'), 'w'), indent=1)
    best = max(res, key=lambda k: res[k]['auc'])
    print(seg, 'best offset', best, round(res[best]['auc'], 4), 'o+0', round(res['0']['auc'], 4), flush=True)
