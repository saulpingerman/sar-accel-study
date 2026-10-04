#!/usr/bin/env bash
# Download the binary figure inputs (135 MB) from the GitHub release into results/v2r/.
set -eu
cd "$(dirname "$0")"
REL=https://github.com/saulpingerman/sar-accel-study/releases/download/v2-data
mkdir -p results/v2r/figs results/v2r/ship/tiles
for f in panama_crops.npz gec_channel_windows.npz panama_overview_db.npy melbourne_overview_db.npy iowa_overview_db.npy s2_S2A_20240223.npz s2_S2B_20230723.npz; do
  [ -f results/v2r/figs/$f ] || curl -L -o results/v2r/figs/$f $REL/$f
done
for f in tile_+0.00.npy tile_+3.00.npy; do
  [ -f results/v2r/ship/tiles/$f ] || curl -L -o "results/v2r/ship/tiles/$f" "$REL/$f"
done
echo done
