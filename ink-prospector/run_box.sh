#!/usr/bin/env bash
# Dense pass over one box of a scroll, one shard per call. Resume-safe; MAXP limits new patches per call.
# Usage: bash run_box.sh <workdir> <scroll in scrolls.json> <tag> <shard i/n> <spacing> <z0> <z1> <y0> <y1> <x0> <x1>
set -uo pipefail
W=$1; SC=$2; TAG=$3; SH=$4; SP=$5; shift 5; BOX="$*"; HERE=$(cd "$(dirname "$0")" && pwd)
cp "$HERE"/*.py "$HERE"/scrolls.json "$W"/code/; cd "$W"; OUT=data/runs/$TAG/shard-${SH%/*}; mkdir -p "$OUT"
CT=$(venv/bin/python -c "import json;print(json.load(open('code/scrolls.json'))['$SC']['ct'])"); M7=$(venv/bin/python -c "import json;print(json.load(open('code/scrolls.json'))['$SC']['m7'])")
venv/bin/python code/scan.py --ct "$CT" --m7 "$M7" --out "$OUT" --box $BOX --spacing "$SP" --half-nodes 128 --shard "$SH" ${MAXP:+--max-patches $MAXP} 2>&1 | tee "$OUT/last_raw.txt" | grep -E "^\[" | tee -a "$OUT/log.txt" || true
grep -q Traceback "$OUT/last_raw.txt" && { echo "SCAN_CRASHED:"; tail -8 "$OUT/last_raw.txt"; }
N=$(grep -oE "[0-9]+ in shard" "$OUT/log.txt" | tail -1 | grep -oE "^[0-9]+"); D=$(($(wc -l < "$OUT/scores.tsv") - 1))
echo "BOX_PASS_DONE $TAG shard $SH: $D of ${N:-?} seeds visited, $(grep -c "	ok	" "$OUT/scores.tsv" || true) patches"
[ "${N:-x}" = "$D" ] && echo ALL_SEEDS_DONE
