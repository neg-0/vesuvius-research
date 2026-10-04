#!/usr/bin/env bash
# Copies the pinned, MIT-licensed reporter tree (TAUIL-Abd-Elilah/cross-scan-ink-transfer @ df9db2f)
# into ./work, adds our v013_*.py drivers, clones villa (for the model code) and downloads
# scrollprize/ink_canonical_2um @ 075855bc (checkpoint SHA-256 verified).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=${V013_WORK:-$PWD/work}
mkdir -p "$W/models"
[ -d "$W/xscan" ] || git clone -q https://github.com/TAUIL-Abd-Elilah/cross-scan-ink-transfer "$W/xscan"
git -C "$W/xscan" checkout -q df9db2f4ea7e63fb6584f1aa3548f2773b927617
cp "$HERE"/v013_*.py "$W/xscan/src/"
[ -d "$W/villa" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/ScrollPrize/villa "$W/villa"
CK=$W/models/r152_3ddec_v2_l5_epoch13.ckpt
[ -f "$CK" ] || curl -sL -o "$CK" https://huggingface.co/scrollprize/ink_canonical_2um/resolve/075855bc69317ef6febf39a0d9d687b27d2b7c29/r152_3ddec_v2_l5_epoch13.ckpt
echo "36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0  $CK" | sha256sum -c -
