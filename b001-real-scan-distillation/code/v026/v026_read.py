"""Shared 9 um inference helpers, trimmed for this repository to what the B-001 drivers import (villa import shim,
model loading with the official inference flags, checkpoint sha256, the shuffled-layer control order, zarr writer).
Function bodies are unchanged from the version that ran.
Env: V026_VILLA_SRC (villa <checkout>/vesuvius/src at e0bbb8b), V026_CKPT (scrollprize/ink_9um
hybrid_3d2d-seed42/step-075000.pth), V026_AMP (auto|default|fp16|bf16; default auto).
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = ['--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '4', '--direction', 'both',
         '--no-compile', '--num-workers', '0']
CKPT_SHA256 = 'e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab'   # step-075000.pth, HF 7109667

STUBBED = []


def _stub(name, attrs):
    m = types.ModuleType(name)
    for a in attrs:
        def missing(*_a, _n=a, **_k):
            raise RuntimeError(f'{_n} needs an optional villa dependency that is not installed')
        setattr(m, a, missing)
    sys.modules[name] = m
    STUBBED.append(name)


def villa_infer():
    """Import villa's flat inference without vesuvius/__init__ (which pulls optional heavy extras). If an optional
    dependency of unrelated network builders (timm, einops, huggingface_hub) is missing, those builder modules are
    replaced by stubs that raise if called; the hybrid_3d2d ink model does not use them. Stubs are listed in --check."""
    src = os.environ['V026_VILLA_SRC']
    for name in ('vesuvius', 'vesuvius.data', 'vesuvius.utils', 'vesuvius.image_proc', 'vesuvius.image_proc.intensity',
                 'vesuvius.models'):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(src, *name.split('.'))]
            sys.modules[name] = m
    try:
        import vesuvius.models.build.build_network_from_config  # noqa: F401  (imported lazily by make_model)
        import vesuvius.ink_detection.inference.infer as inf
    except ModuleNotFoundError as e:
        if e.name not in ('timm', 'einops', 'huggingface_hub'):
            raise
        for k in [k for k in sys.modules if k.startswith(('vesuvius.ink_detection', 'vesuvius.models.build'))]:
            del sys.modules[k]
        _stub('vesuvius.models.build.primus_wrapper', ['PrimusEncoder', 'PrimusDecoder'])
        _stub('vesuvius.models.build.pretrained_backbones.dinov2', ['build_dinov2_backbone', 'build_dinov2_decoder'])
        import vesuvius.models.build.build_network_from_config  # noqa: F401,F811
        import vesuvius.ink_detection.inference.infer as inf
    return inf


def load_model():
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    inf = villa_infer()
    ckpt = os.environ['V026_CKPT']
    amp = os.environ.get('V026_AMP', 'auto')
    args = inf.parse_args(['in.zarr', ckpt, 'out.tif', *FLAGS, '--amp-dtype', amp])
    cm = inf.configure_model(args)
    model, dev = inf.prepare_model_for_inference(cm.model, gpu_ids=args.gpu_ids, compile_model=False,
                                                 compile_mode=args.compile_mode)
    from dataclasses import replace
    cm = replace(cm, model=model)
    info = {'checkpoint': os.path.abspath(ckpt), 'patch': cm.patch_size, 'input_depth': cm.input_depth,
            'preprocessing': cm.preprocessing, 'amp': str(cm.amp_dtype), 'device': str(dev), 'flags': FLAGS,
            'villa_src': os.environ['V026_VILLA_SRC'], 'stubbed_modules': list(STUBBED)}
    return inf, args, cm, dev, info


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def shuffled_layers(depth, inf, input_depth):
    sel = inf.select_layer_indices(depth, layer_start=None, layer_end=None, output_depth=input_depth,
                                   direction='forward')
    perm = np.random.default_rng(0).permutation(len(sel))
    order = np.arange(depth)
    order[sel] = sel[perm]
    return order, sel


def write_zarr(path, arr):
    import zarr
    kw = {'zarr_format': 2} if int(zarr.__version__.split('.')[0]) >= 3 else {}
    z = zarr.open(path, mode='w', shape=arr.shape, chunks=(arr.shape[0], 128, 128), dtype=np.uint8, **kw)
    z[:] = arr
