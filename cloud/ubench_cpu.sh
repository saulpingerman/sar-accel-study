#!/usr/bin/env bash
# Unit-rate microbenchmark on the CPU instance.   ubench_cpu.sh <dest>
set -u
DEST=${1:-cpp4}; cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; mkdir -p /tmp/v2
OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close $PY ubench_cpu.py --out /tmp/v2/ubench8.json > /tmp/v2/ubench8.log 2>&1
OMP_NUM_THREADS=16 OMP_PLACES=threads OMP_PROC_BIND=close $PY ubench_cpu.py --out /tmp/v2/ubench16.json > /tmp/v2/ubench16.log 2>&1
gcloud storage cp /tmp/v2/ubench8.json $B/v2/$DEST/ubench_cpu-c4d16_t8.json >/dev/null 2>&1
gcloud storage cp /tmp/v2/ubench16.json $B/v2/$DEST/ubench_cpu-c4d16_t16.json >/dev/null 2>&1
echo DONE > /tmp/v2/done_ubench_$DEST; gcloud storage cp /tmp/v2/done_ubench_$DEST $B/v2/logs/ >/dev/null 2>&1
