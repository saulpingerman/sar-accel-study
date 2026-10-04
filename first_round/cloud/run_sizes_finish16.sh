#!/usr/bin/env bash
# On the large CPU instance: once the float64 reference at 16384 is written, stop the local emulation run
# (done on a second instance), fetch those images, and compare everything with the reference.
set -u
cd ~/sar; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; D=/data
until grep -q "bp/numba_fp64 ok" /tmp/form16384.log 2>/dev/null; do sleep 30; done
for p in $(pgrep -f "[r]un_sizes_big.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "[s]izes.py form --data /data/szfull.npz --n 16384"); do kill $p; done
until gcloud storage ls $B/sizes/done_emu16 >/dev/null 2>&1; do sleep 30; done
rm -f $D/out/sz16384_cpu/ffbp_*.npy
gcloud storage cp "$B/sizes/out/sz16384_cpuemu/ffbp_*.npy" $D/out/sz16384_cpu/ >/dev/null 2>&1
bash cloud/run_sizes_analyze.sh 16384 > /tmp/analyze16384.log 2>&1
echo DONE > /tmp/sizes_16384_done
gcloud storage cp /tmp/sizes_16384_done $B/sizes/done_16384 >/dev/null 2>&1
