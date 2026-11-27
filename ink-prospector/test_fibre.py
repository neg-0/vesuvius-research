"""fibre_snap moves a surface onto the fibre ply: a flat sheet 3 voxels below a synthetic fibre plane moves up by 3."""
import numpy as np
import fibre_snap


def test_flat_sheet_snaps_to_fibre_plane():
    H = W = 120; D = np.zeros((H, W, 3)); D[..., 0] = 50.0; D[..., 1], D[..., 2] = np.mgrid[0:H, 0:W]
    valid = np.ones((H, W), bool); Nd = np.zeros((H, W, 3)); Nd[..., 0] = 1.0
    prob = lambda q: np.exp(-0.5 * ((q[:, 0] - 53.0) / 1.0) ** 2).astype(np.float32)   # fibres at z = 53
    field, sv, conf, peak = fibre_snap.snap_offsets(D, valid, Nd, prob, radius=6, stride=4, sigma=16)
    assert conf.mean() > 0.95 and np.allclose(field[sv], 3.0, atol=0.05)
    P = D[::20, ::20].copy(); ok = np.ones(P.shape[:2], bool); ok[0, 0] = False
    P2 = fibre_snap.move_nodes(P, ok, Nd, field, 20, 4)
    assert np.allclose(P2[ok][:, 0], 53.0, atol=0.05) and P2[0, 0, 0] == 50.0                 # invalid nodes stay put


def test_no_fibres_means_no_shift():
    H = W = 64; D = np.zeros((H, W, 3)); D[..., 1], D[..., 2] = np.mgrid[0:H, 0:W]
    Nd = np.zeros((H, W, 3)); Nd[..., 0] = 1.0
    field, sv, conf, _ = fibre_snap.snap_offsets(D, np.ones((H, W), bool), Nd, lambda q: np.zeros(len(q), np.float32))
    assert not conf.any() and np.all(field == 0)
