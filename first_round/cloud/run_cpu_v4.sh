#!/usr/bin/env bash
# Score everything the accelerators formed against the CPU float64 references.
#   run_cpu_v4.sh "<labels>"     e.g. "tpu-v5e tpu-v6e gpu-l4"
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; mkdir -p results/quality results/real results/lat
for L in $1; do
  if gcloud storage cp $B/q/q768_$L.npz /tmp/ >/dev/null 2>&1; then
    $PY quality.py analyze --data /tmp/q_air.npz --ref /tmp/q_air_cpu.npz --test /tmp/q768_$L.npz --out results/quality/q_air_$L.json > /tmp/qan_$L.log 2>&1
  fi
  if gcloud storage cp $B/q/hdr_$L.npz /tmp/ >/dev/null 2>&1; then
    $PY ccd_hdr.py --data /tmp/hdr.npz --ref /tmp/hdr_cpu.npz --images /tmp/hdr_$L.npz --out results/quality/hdr_$L.json > /tmp/hdran_$L.log 2>&1
  fi
  if gcloud storage cp $B/q/hdr2_$L.npz /tmp/ >/dev/null 2>&1; then
    $PY ccd_hdr.py --data /tmp/hdr.npz --ref /tmp/hdr2_cpu.npz --images /tmp/hdr2_$L.npz --win 13 --out results/quality/hdr2_$L.json > /tmp/hdr2an_$L.log 2>&1
  fi
  if gcloud storage cp $B/real/lock_$L.npz /tmp/umbra/ >/dev/null 2>&1; then
    $PY realdata.py analyze --images /tmp/umbra/lock_$L.npz --ref /tmp/umbra/lock_cpu.npz --out results/real/lock_$L.json > /tmp/realan_$L.log 2>&1
  fi
done
for N in 1024 2048 4096 8192 4096s; do
  CH=/tmp/lat${N}_ref.npz
  for L in $1; do
    if gcloud storage cp $B/lat/lat${N}_$L.npz /tmp/ >/dev/null 2>&1; then CH=$CH,/tmp/lat${N}_$L.npz; fi
  done
  [ -f /tmp/lat${N}_ref.npz ] && $PY validate_large.py analyze --data /tmp/lat$N.npz --ref /tmp/lat${N}_ref.npz --chips $CH --out results/lat/lat$N.json > /tmp/latan_$N.log 2>&1
done
tar czf /tmp/v4_results.tgz results/quality results/real/*.json results/lat
gcloud storage cp /tmp/v4_results.tgz $B/results/v2/ >/dev/null 2>&1
echo DONE > /tmp/v4_done
