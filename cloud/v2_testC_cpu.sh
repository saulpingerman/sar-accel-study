#!/usr/bin/env bash
# Slant-plane crop test on the CPU instance: float64 reference, factorized and polar format on a 1024 x 1024
# centre crop of the Panama native grid; the reference crop goes to the bucket for the accelerator tests.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
rm -rf /data/t; mkdir -p /data/t
$PY v2_form.py ref --data /data/panama.npz --outdir /data/t/b_ref --crop 1024,1024 > /tmp/tc_ref.log 2>&1
gcloud storage cp -r /data/t/b_ref $B/v2/t/ >/dev/null 2>&1
$PY v2_form.py ffbp --data /data/panama.npz --outdir /data/t/b_cpu --label cpu --crop 1024,1024 --policies fp32 --filters conv --trig direct --reps 1 > /tmp/tc_ffbp.log 2>&1
$PY v2_form.py pfa --data /data/panama.npz --outdir /data/t/b_cpu --label cpu --crop 1024,1024 --policies fp32 --reps 1 > /tmp/tc_pfa.log 2>&1
$PY v2_metrics.py --ref /data/t/b_ref --test /data/t/b_cpu --out /tmp/tc.json --scene testc > /tmp/tc_metrics.log 2>&1
echo C > /tmp/v2_testC_done
