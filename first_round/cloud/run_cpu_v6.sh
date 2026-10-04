#!/usr/bin/env bash
# Sixth CPU sequence: lattice references for the streamed plans at 1024 and 2048 and at 600 km,
# and the dynamic-range scene re-formed with a plan whose first level filters.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
for cfg in "1024 10e3 1024" "2048 10e3 2048" "4096 600e3 4096s"; do
  set -- $cfg
  $PY validate_large.py make --N $1 --r0 $2 --out /tmp/lat$3.npz > /tmp/lat$3_make.log 2>&1
  $PY validate_large.py ref --data /tmp/lat$3.npz --out /tmp/lat$3_ref.npz > /tmp/lat$3_ref.log 2>&1
  gcloud storage cp /tmp/lat$3.npz /tmp/lat$3_ref.npz /tmp/lat$3_ref_img.npy $B/lat/ >/dev/null 2>&1
done
echo DONE > /tmp/v6_lat_done
[ -f /tmp/hdr.npz ] || gcloud storage cp $B/q/hdr.npz /tmp/ >/dev/null 2>&1
$PY quality.py form --data /tmp/hdr.npz --out /tmp/hdr2_cpu.npz --n 1024 --spacing 0.17 --x64 --algos bp,ffbp --ffbp-levels "4:2,4:4,4:4@16" \
    --policies fp64,fp32 > /tmp/hdr2_form.log 2>&1
$PY ccd_hdr.py --data /tmp/hdr.npz --images /tmp/hdr2_cpu.npz --win 13 --out results/quality/hdr2_cpu.json > /tmp/hdr2_an.log 2>&1
gcloud storage cp results/quality/hdr2_cpu.json $B/results/v2/ >/dev/null 2>&1
echo DONE > /tmp/v6_done
