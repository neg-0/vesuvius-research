"""The AFV V2 papyrus-fibre network (Qualzz20/afv_fiber_9um_v2 on Hugging Face, Apache-2.0), defined here from its
published architecture so no downloaded code is imported. Weights are loaded as data (weights_only=True) after a sha256
check. Needs torch and dynamic_network_architectures (setup.sh --fibre installs both and fetches the weights).

Output classes of `seg`: 0 background, 1 vertical fibres, 2 horizontal fibres, 3 other."""
import hashlib, importlib, json
from pathlib import Path
import torch
from torch import nn

REPO, REV = "Qualzz20/afv_fiber_9um_v2", "2a8a26a70754a5d90aeca91f9ef8e7130e6e4f81"
SHA = "b8519a4ce4044c93ed98af0c278628fdcdfdd8808275b3c8becd7211f38b7967"
ALLOWED = {"torch.nn.modules.conv.Conv3d", "torch.nn.modules.instancenorm.InstanceNorm3d", "torch.nn.modules.activation.LeakyReLU", "torch.nn.LeakyReLU",
           "torch.nn.modules.dropout.Dropout3d", "torch.nn.modules.batchnorm.BatchNorm3d", "torch.nn.modules.activation.ReLU"}


class FiberDetector(nn.Module):
    def __init__(self, plans):
        super().__init__()
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
        a = plans["configurations"]["3d_fullres"]["architecture"]; args = dict(a["arch_kwargs"])
        for k in a["_kw_requires_import"]:
            if args[k] is not None:
                if args[k] not in ALLOWED: raise ValueError(f"plans.json names an unexpected module: {args[k]}")
                mod, name = args[k].rsplit(".", 1); args[k] = getattr(importlib.import_module(mod), name)
        self.backbone = ResidualEncoderUNet(input_channels=1, num_classes=4, deep_supervision=False, **args)
        c = args["features_per_stage"][0]
        self.centre = nn.Conv3d(c, 1, 1); self.affinity = nn.Conv3d(c, 6, 1); self.direction = nn.Conv3d(c, 3, 1)

    def forward(self, x):
        sk = self.backbone.encoder(x); dec = self.backbone.decoder; x = sk[-1]
        for i, (up, st) in enumerate(zip(dec.transpconvs, dec.stages)): x = st(torch.cat((up(x), sk[-i - 2]), 1))
        return {"seg": dec.seg_layers[-1](x)}


def normalize_ct(ct):
    ct = ct.float(); return (ct - ct.mean(dim=(2, 3, 4), keepdim=True)) / ct.std(dim=(2, 3, 4), keepdim=True).clamp_min(10.0)


def load_model(folder):
    f = Path(folder); ck = f / "checkpoint_final.pth"
    h = hashlib.sha256(ck.read_bytes()).hexdigest()
    if h != SHA: raise ValueError(f"{ck}: sha256 {h} is not the pinned AFV V2 checkpoint")
    m = FiberDetector(json.loads((f / "plans.json").read_text()))
    c = torch.load(ck, map_location="cpu", weights_only=True)
    st = {"backbone." + k: v for k, v in c["network_weights"].items()}; st.update(c["auxiliary_weights"])
    m.load_state_dict(st, strict=True); return m.eval()
