#!/usr/bin/env bash
# Functional tests of the second-round kernels on the CPU instance.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; mkdir -p /data/t
# A: small decimated patch from the first round, every filter form, float64 and float32, both phase-ramp forms
[ -f /data/sz1024.npz ] || gcloud storage cp $B/sizes/sz1024.npz /data/ >/dev/null 2>&1
[ -f /data/t/a_ref/bp_numba_fp64.npy ] || $PY v2_form.py ref --data /data/sz1024.npz --outdir /data/t/a_ref --crop 768,1024 > /tmp/ta_ref.log 2>&1
rm -rf /data/t/a_cpu
$PY v2_form.py ffbp --data /data/sz1024.npz --outdir /data/t/a_cpu --label cpu --crop 768,1024 --x64 --policies fp64,fp32 --filters dense,conv,taps --reps 1 > /tmp/ta_cpu.log 2>&1
$PY v2_form.py ffbp --data /data/sz1024.npz --outdir /data/t/a_cpu --label cpu --crop 768,1024 --x64 --policies fp64,fp32 --filters dense --trig direct --reps 1 >> /tmp/ta_cpu.log 2>&1
$PY v2_metrics.py --ref /data/t/a_ref --test /data/t/a_cpu --out /tmp/ta.json --scene testa > /tmp/ta_metrics.log 2>&1
echo A > /tmp/v2_testA_done
