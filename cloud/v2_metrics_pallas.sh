#!/usr/bin/env bash
# Full-image statistics of the fused-kernel images (v2/pallas3/<scene>/<label>/) against the float64 references,
# one process per scene in parallel.   v2_metrics_pallas.sh <labels>
set -u
LABELS=$1; DESTS=${2:-pallas3}; OUT=${3:-pallas3}; cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
for s in panama melbourne iowa; do
  mkdir -p /data/out/$s/ref
  [ -f /data/out/$s/ref/bp_numba_fp64.npy ] || gcloud storage cp $B/v2/out/$s/ref/bp_numba_fp64.npy /data/out/$s/ref/ >/dev/null 2>&1 &
done
wait
for s in panama melbourne iowa; do
  for D in $(echo $DESTS | tr ',' ' '); do for L in $(echo $LABELS | tr ',' ' '); do
    gcloud storage ls $B/v2/$D/$s/$L/timing.json >/dev/null 2>&1 || continue
    rm -rf /data/$OUT/$s/${D}_$L; mkdir -p /data/$OUT/$s/${D}_$L
    # the device's float32 (six-pass) image of record supplies the family gain, as in the original scoring
    { gcloud storage cp -r "$B/v2/$D/$s/$L/*" /data/$OUT/$s/${D}_$L/ >/dev/null 2>&1; gcloud storage cp "$B/v2/out/$s/$L/ffbp_fp32.npy" /data/$OUT/$s/${D}_$L/ >/dev/null 2>&1; } &
  done; done
done
wait
for s in panama melbourne iowa; do
  T=$(ls -d /data/$OUT/$s/*/ | sed 's|/$||' | tr '\n' ',' | sed 's/,$//')
  CR=""; [ "$s" = panama ] && CR="${CROPS:-}"
  $PY v2_metrics.py --ref /data/out/$s/ref --test $T --out /data/$OUT/metrics_$s.json --scene $s ${CR:+--figs /data/$OUT/figs_$s --crops "$CR"} > /tmp/metrics_${OUT}_$s.log 2>&1 &
done
wait
for s in panama melbourne iowa; do
  gcloud storage cp /data/$OUT/metrics_$s.json $B/v2/results/${s}_${OUT}_metrics.json >/dev/null 2>&1
  gcloud storage cp /tmp/metrics_${OUT}_$s.log $B/v2/logs/ >/dev/null 2>&1
  [ -f /data/$OUT/figs_$s/${s}_crops.npz ] && gcloud storage cp /data/$OUT/figs_$s/${s}_crops.npz $B/v2/results/${s}_${OUT}_crops.npz >/dev/null 2>&1
done
echo DONE > /tmp/metrics_${OUT}_done; gcloud storage cp /tmp/metrics_${OUT}_done $B/v2/logs/done_metrics_${OUT} >/dev/null 2>&1
