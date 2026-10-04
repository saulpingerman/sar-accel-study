#!/usr/bin/env bash
# Full-image statistics for one scene against its float64 reference, on the large CPU instance.
# v2_metrics_job.sh <scene> <label>[,<label>...]      (waits for each label's done marker)
set -u
s=$1; LABELS=$2
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
mkdir -p /data/out/$s/figs
if [ ! -f /tmp/ref_${s}_done ]; then      # the reference was made on another instance
  until gcloud storage ls $B/v2/out/$s/ref_done >/dev/null 2>&1; do sleep 60; done
  mkdir -p /data/out/$s/ref; gcloud storage cp $B/v2/out/$s/ref/bp_numba_fp64.npy /data/out/$s/ref/ >/dev/null 2>&1
fi
until [ -f /data/out/$s/ref/bp_numba_fp64.npy ]; do sleep 60; done
for M in $(echo ${MARKERS:-$LABELS} | tr ',' ' '); do
  until gcloud storage ls $B/v2/logs/done_$M >/dev/null 2>&1; do sleep 60; done
done
for L in $(echo $LABELS | tr ',' ' '); do
  rm -rf /data/out/$s/$L; gcloud storage cp -r $B/v2/out/$s/$L /data/out/$s/ >/dev/null 2>&1
done
T=$(for L in $(echo $LABELS | tr ',' ' '); do printf "/data/out/$s/$L,"; done | sed 's/,$//')
$PY v2_metrics.py --ref /data/out/$s/ref --test $T --out /data/out/$s/metrics.json --scene $s --figs /data/out/$s/figs --crops "${CROPS:-}" > /tmp/metrics_$s.log 2>&1
gcloud storage cp /data/out/$s/metrics.json $B/v2/results/${s}_metrics.json >/dev/null 2>&1
gcloud storage cp -r /data/out/$s/figs $B/v2/results/ >/dev/null 2>&1
gcloud storage cp /tmp/metrics_$s.log $B/v2/logs/ >/dev/null 2>&1
echo M > /tmp/metrics_${s}_done
