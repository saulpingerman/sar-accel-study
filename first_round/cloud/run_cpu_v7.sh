#!/usr/bin/env bash
# Seventh CPU sequence: backprojection precision rows for the re-formed dynamic-range scene.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
$PY quality.py form --data /tmp/hdr.npz --out /tmp/hdr2b_cpu.npz --n 1024 --spacing 0.17 --algos bp --policies bf16_store,bf16_arith,bf16_acc,f16_arith > /tmp/hdr2b_form.log 2>&1
$PY ccd_hdr.py --data /tmp/hdr.npz --ref /tmp/hdr2_cpu.npz --images /tmp/hdr2b_cpu.npz --win 13 --out results/quality/hdr2b_cpu.json > /tmp/hdr2b_an.log 2>&1
gcloud storage cp results/quality/hdr2b_cpu.json $B/results/v2/ >/dev/null 2>&1
echo DONE > /tmp/v7_done
