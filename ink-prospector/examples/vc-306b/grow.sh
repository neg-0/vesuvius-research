#!/usr/bin/env bash
# Grow each Prospector hit into a larger segment with VC's tracer, on the surface prediction.
# Run from this folder. Output: grown/<id>_grown/ (tifxyz, opens in VC3D).
set -u
cd "$(dirname "$0")"
M7="${M7:-https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/representations/predictions/surfaces/20250521133212-surface-20260413222639-surface-m7-L0-th0.2.zarr}"
mkdir -p grown
vc_grow_seg_from_seed -v "$M7" -t grown/prospector_demo-1_grown -p params.json --resume paths/prospector_demo-1 --segment-name prospector_demo-1_grown || true
vc_grow_seg_from_seed -v "$M7" -t grown/prospector_demo-2_grown -p params.json --resume paths/prospector_demo-2 --segment-name prospector_demo-2_grown || true
vc_grow_seg_from_seed -v "$M7" -t grown/prospector_demo-0_grown -p params.json --resume paths/prospector_demo-0 --segment-name prospector_demo-0_grown || true
