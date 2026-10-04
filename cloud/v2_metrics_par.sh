#!/usr/bin/env bash
# Full-image statistics for one scene, one process per device label in parallel.  v2_metrics_par.sh <scene> <labels>
set -u
s=$1; LABELS=$2
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
mkdir -p /data/out/$s
if [ ! -f /data/out/$s/ref/bp_numba_fp64.npy ]; then
  mkdir -p /data/out/$s/ref; gcloud storage cp $B/v2/out/$s/ref/bp_numba_fp64.npy /data/out/$s/ref/ >/dev/null 2>&1
fi
for L in $(echo $LABELS | tr ',' ' '); do
  rm -rf /data/out/$s/$L; gcloud storage cp -r $B/v2/out/$s/$L /data/out/$s/ >/dev/null 2>&1 &
done
wait
for L in $(echo $LABELS | tr ',' ' '); do
  $PY v2_metrics.py --ref /data/out/$s/ref --test /data/out/$s/$L --out /data/out/$s/metrics_$L.json --scene $s > /tmp/metrics_${s}_$L.log 2>&1 &
done
wait
for L in $(echo $LABELS | tr ',' ' '); do
  gcloud storage cp /data/out/$s/metrics_$L.json $B/v2/results/${s}_${L}_metrics.json >/dev/null 2>&1
  gcloud storage cp /tmp/metrics_${s}_$L.log $B/v2/logs/ >/dev/null 2>&1
done
echo M > /tmp/metrics_${s}_done; gcloud storage cp /tmp/metrics_${s}_done $B/v2/results/${s}_done >/dev/null 2>&1
