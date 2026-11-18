#!/usr/bin/env bash
# Sparse whole-scroll pass: seeds a lattice through the full volume of a scroll and scans one shard of it.
# Resume-safe; MAXP limits new patches per call, so loop the same line until it prints ALL_SEEDS_DONE.
# Usage: bash run_whole.sh <workdir> <scroll name in scrolls.json> <shard i/n> [spacing=800]      (after setup.sh <workdir>)
set -uo pipefail
W=$1; SC=$2; SH=$3; SP=${4:-800}; HERE=$(cd "$(dirname "$0")" && pwd)
cp "$HERE"/*.py "$HERE"/scrolls.json "$W"/code/; cd "$W"; OUT=data/runs/whole-$SC/shard-${SH%/*}; mkdir -p "$OUT"
read CT M7 BOX <<< "$(venv/bin/python - "$SC" <<'PY'
import json, sys, urllib.request
s = json.load(open("code/scrolls.json"))[sys.argv[1]]
shp = json.load(urllib.request.urlopen("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/" + s["ct"] + "/0/.zarray"))["shape"]
print(s["ct"], s["m7"], " ".join(f"{200} {n - 200}" for n in shp))
PY
)"
[ -n "$BOX" ] || { echo "could not read the volume shape"; exit 2; }
venv/bin/python code/scan.py --ct "$CT" --m7 "$M7" --out "$OUT" --box $BOX --spacing "$SP" --half-nodes 128 --shard "$SH" ${MAXP:+--max-patches $MAXP} 2>&1 | tee "$OUT/last_raw.txt" | grep -E "^\[" | tee -a "$OUT/log.txt" || true
grep -q Traceback "$OUT/last_raw.txt" && { echo "SCAN_CRASHED:"; tail -8 "$OUT/last_raw.txt"; }
N=$(grep -oE "[0-9]+ in shard" "$OUT/log.txt" | tail -1 | grep -oE "^[0-9]+"); D=$(($(wc -l < "$OUT/scores.tsv") - 1))
echo "WHOLE_PASS_DONE $SC shard $SH: $D of ${N:-?} seeds visited, $(grep -c "	ok	" "$OUT/scores.tsv" || true) patches"
[ "${N:-x}" = "$D" ] && echo ALL_SEEDS_DONE
