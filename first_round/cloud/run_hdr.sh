#!/usr/bin/env bash
# Harder change-detection scene on the CPU instance: simulate, form at every precision, score.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
$PY quality.py make-data --hdr --cnr-db 40 --K 1024 --geom air --out /tmp/hdr.npz > /tmp/hdr_make.log 2>&1
$PY quality.py form --data /tmp/hdr.npz --out /tmp/hdr_cpu.npz --n 768 --x64 --algos bp,ffbp --ffbp-levels "2:1,3:3,4:4@32" \
    --policies fp64,fp32,bf16_store,bf16_arith,bf16_acc,f16_arith > /tmp/hdr_form.log 2>&1
mkdir -p results/quality
$PY ccd_hdr.py --data /tmp/hdr.npz --images /tmp/hdr_cpu.npz --out results/quality/hdr_cpu.json > /tmp/hdr_an.log 2>&1
gcloud storage cp results/quality/hdr_cpu.json $B/results/ >/dev/null 2>&1
gcloud storage cp /tmp/hdr.npz $B/q/ >/dev/null 2>&1
echo DONE > /tmp/hdr_done
