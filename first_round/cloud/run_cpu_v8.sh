#!/usr/bin/env bash
# Eighth CPU sequence: dynamic-range scene with six clutter bands, 0 to -50 dB, and noise 60 dB below the brightest.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
$PY quality.py make-data --hdr --strata=0,-10,-20,-30,-40,-50 --cnr-db 60 --K 1024 --geom air --out /tmp/hdr6.npz > /tmp/hdr6_make.log 2>&1
$PY quality.py form --data /tmp/hdr6.npz --out /tmp/hdr6_cpu.npz --n 1024 --spacing 0.17 --x64 --algos bp,ffbp --ffbp-levels "4:2,4:4,4:4@16" \
    --policies fp64,fp32 > /tmp/hdr6_form.log 2>&1
$PY ccd_hdr.py --data /tmp/hdr6.npz --images /tmp/hdr6_cpu.npz --win 13 --out results/quality/hdr6_cpu.json > /tmp/hdr6_an.log 2>&1
gcloud storage cp results/quality/hdr6_cpu.json $B/results/v2/ >/dev/null 2>&1
echo DONE > /tmp/v8_done
