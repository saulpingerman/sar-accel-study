#!/usr/bin/env bash
# Download the binary figure inputs from the GitHub release figure-data into results/fastsar/ (about 350 MB):
# downsampled overviews of the three collections, Sentinel-2 windows, geocoded windows of other Umbra passes, the
# refocused ship tiles, the image crops of the release records and the reference-against-vendor crops.
set -eu
cd "$(dirname "$0")"
D=https://github.com/saulpingerman/sar-accel-study/releases/download/figure-data
mkdir -p results/fastsar/figs results/fastsar/ship/tiles results/fastsar/sicd
for f in gec_channel_windows.npz panama_overview_db.npy melbourne_overview_db.npy iowa_overview_db.npy s2_S2A_20240223.npz s2_S2B_20230723.npz; do
  [ -f results/fastsar/figs/$f ] || curl -L -o results/fastsar/figs/$f $D/$f
done
for f in tile_+0.00.npy tile_+3.00.npy; do
  [ -f results/fastsar/ship/tiles/$f ] || curl -L -o "results/fastsar/ship/tiles/$f" "$D/$f"
done
[ -f results/fastsar/figs/panama_crops.npz ] || curl -L -o results/fastsar/figs/panama_crops.npz $D/panama_crops.npz
for f in sicd4_panama_crops.npz sicd_panama_raw_crops.npz; do
  [ -f results/fastsar/sicd/$f ] || curl -L -o results/fastsar/sicd/$f $D/$f
done
echo done
