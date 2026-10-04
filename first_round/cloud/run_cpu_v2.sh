#!/usr/bin/env bash
# Second CPU sequence: re-score the harder change-detection scene with the mixed policy,
# then build the large-scene references. Waits for the measured-data job to finish first.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
until [ -f /tmp/umbra/real_done ]; do sleep 20; done
$PY quality.py form --data /tmp/hdr.npz --out /tmp/hdr_cpu.npz --n 768 --x64 --algos bp,ffbp --ffbp-levels "2:1,3:3,4:4@32" \
    --policies fp64,fp32,bf16_store,bf16_arith,bf16_acc,f16_arith > /tmp/hdr_form.log 2>&1
$PY ccd_hdr.py --data /tmp/hdr.npz --images /tmp/hdr_cpu.npz --out results/quality/hdr_cpu.json > /tmp/hdr_an.log 2>&1
gcloud storage cp results/quality/hdr_cpu.json $B/results/ >/dev/null 2>&1
for N in 4096 8192; do
  $PY validate_large.py make --N $N --out /tmp/lat$N.npz > /tmp/lat${N}_make.log 2>&1
  $PY validate_large.py ref --data /tmp/lat$N.npz --out /tmp/lat${N}_ref.npz > /tmp/lat${N}_ref.log 2>&1
  gcloud storage cp /tmp/lat$N.npz /tmp/lat${N}_ref.npz /tmp/lat${N}_ref_img.npy $B/lat/ >/dev/null 2>&1
  echo DONE > /tmp/lat${N}_done
done
