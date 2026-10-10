#!/usr/bin/env bash
# Second-generation fused kernel on a TPU: block sweep on Panama, then the measured rows on the three collections.
#   pallas2_tpu.sh <label>
set -u
L=$1; cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
run() { $PY form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
mkdir -p /tmp/v2; s=panama; [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
for cfg in "256 8 8" "512 8 8" "256 16 8" "256 8 16" "128 8 8"; do
  set -- $cfg; pb=$1; nc=$2; ng=$3; tag=pb${pb}_nc${nc}_ng${ng}
  O=/tmp/v2/pal2/$s/$tag; rm -rf $O; mkdir -p $O
  run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast --filters pallas2 --trig direct --reps 2 --nosave --pallas-pb $pb --pallas-nc $nc --pallas-ng $ng > /tmp/v2/q_${s}_${L}_$tag.log 2>&1
  gcloud storage cp $O/timing.json $B/v2/pallas2/$s/$L/$tag.json >/dev/null 2>&1; gcloud storage cp /tmp/v2/q_${s}_${L}_$tag.log $B/v2/logs/ >/dev/null 2>&1
done
BEST=$($PY - <<'PY'
import json, glob
best=None
for f in glob.glob('/tmp/v2/pal2/panama/pb*/timing.json'):
    t=json.load(open(f)).get('ffbp/fp32_fast_pallas2_direct', {}).get('run_s')
    if t and (best is None or t < best[0]): best=(t, f.split('/')[-2])
print(best[1] if best else 'pb256_nc8_ng8')
PY
)
pb=$(echo $BEST | sed 's/pb\([0-9]*\)_nc\([0-9]*\)_ng\([0-9]*\)/\1/'); nc=$(echo $BEST | sed 's/pb\([0-9]*\)_nc\([0-9]*\)_ng\([0-9]*\)/\2/'); ng=$(echo $BEST | sed 's/pb\([0-9]*\)_nc\([0-9]*\)_ng\([0-9]*\)/\3/')
echo "best $BEST" > /tmp/v2/q_best_${L}.log; gcloud storage cp /tmp/v2/q_best_${L}.log $B/v2/logs/ >/dev/null 2>&1
for s in panama melbourne iowa; do
  [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/pal2/$s/best; rm -rf $O; mkdir -p $O
  { run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters pallas2 --trig direct --reps 2 --stream 4 --stream-seconds 60 --pallas-pb $pb --pallas-nc $nc --pallas-ng $ng
    run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters dense --trig direct --reps 2 --nosave
  } > /tmp/v2/q_${s}_${L}.log 2>&1
  gcloud storage cp $O/*.npy $O/timing.json $B/v2/pallas2/$s/$L/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/q_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
  rm -rf $O
done
echo DONE > /tmp/v2/done_${L}_pallas2; gcloud storage cp /tmp/v2/done_${L}_pallas2 $B/v2/logs/ >/dev/null 2>&1
