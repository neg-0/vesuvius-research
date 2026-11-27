#!/usr/bin/env bash
# Build a CPU work dir for the Ink Prospector outside the repo. Usage: bash setup.sh [--fibre] [workdir]   (default ~/prospector)
# --fibre also installs the AFV V2 papyrus-fibre model for fibre_snap.py (dynamic_network_architectures + 566 MB of weights)
set -euo pipefail
FIBRE=0; [ "${1:-}" = --fibre ] && { FIBRE=1; shift; }
VILLA_COMMIT=47a9c37f6f8e99afdab47bb46ad425aa726f5dc0   # ScrollPrize/villa, branch merge-ink-pipelines
W=${1:-$HOME/prospector}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"/{code,data/weights,ext}; cp "$HERE"/*.py "$HERE"/scrolls.json "$W"/code/; mkdir -p "$W"/code/viewer "$W"/code/app; cp "$HERE"/viewer/index.html "$W"/code/viewer/; cp "$HERE"/app/index.html "$W"/code/app/
cd "$W"
[ -d venv ] || python3 -m venv venv
venv/bin/pip -q install --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" torch torchvision
# numpy 2.2.0 on Python 3.14 evaluates ~mask into the mask's own buffer (silently flips local arrays); 2.2.6 does not
venv/bin/pip -q install "numpy>=2.2.6,<2.3" scipy "zarr==2.18.7" "numcodecs==0.15.1" tifffile imagecodecs timm einops tqdm pillow \
  opencv-python-headless PyYAML fsspec scikit-image requests pynrrd
if [ ! -d ext/villa-mip ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa ext/villa-mip
  (cd ext/villa-mip && git sparse-checkout set ink-detection vesuvius/src && git checkout -q $VILLA_COMMIT)
fi
if [ ! -f data/weights/reader-v3.pth ]; then
  curl -sSL -m 900 -o data/weights/reader-v3.pth \
    "https://huggingface.co/domenicor046/reader-v3/resolve/39f626e595304c35c6ef574ffc0b6cc1e0ed44cb/reader-v3.pth"
fi
echo "ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a  data/weights/reader-v3.pth" | sha256sum -c -
# install any module the reader still misses, then smoke test (prints: 384 17 tifxyz_robust, then seconds per tile)
for i in 1 2 3 4 5 6; do
  out=$(venv/bin/python code/reader.py data/weights/reader-v3.pth ext/villa-mip 2>&1 | tail -3) || true
  m=$(echo "$out" | grep -oE "No module named '[^']+'" | sed "s/No module named '//;s/'//;s/\..*//" || true)
  [ -z "$m" ] && { echo "$out"; break; }
  case $m in cv2) m=opencv-python-headless;; yaml) m=PyYAML;; skimage) m=scikit-image;; nrrd) m=pynrrd;; PIL) m=pillow;; esac
  venv/bin/pip -q install "$m"
done
if [ $FIBRE = 1 ]; then
  venv/bin/pip -q install "dynamic_network_architectures==0.4.4"
  mkdir -p data/weights/afv-v2
  for f in checkpoint_final.pth plans.json; do
    [ -f data/weights/afv-v2/$f ] || curl -sSL -m 1800 -o data/weights/afv-v2/$f \
      "https://huggingface.co/Qualzz20/afv_fiber_9um_v2/resolve/2a8a26a70754a5d90aeca91f9ef8e7130e6e4f81/$f"
  done
  (cd data/weights/afv-v2 && printf '%s\n' \
    "b8519a4ce4044c93ed98af0c278628fdcdfdd8808275b3c8becd7211f38b7967  checkpoint_final.pth" \
    "a81fd986d0812688de753a72360706897b0bd29e55fd1d6d0cb29f7e31063f65  plans.json" | sha256sum -c -)
fi
echo SETUP_OK "$W"
