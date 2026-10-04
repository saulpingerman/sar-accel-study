#!/usr/bin/env bash
# Fused final stage on top of the fused levels: the measured rows on the three collections, plus the XLA-final
# ablation on Panama.   v2_pallas3_tpu.sh <label> <pb> <nc> <ng>
set -u
L=$1; pb=$2; nc=$3; ng=$4; DEST=${5:-pallas3}; LP=${6:-r}; cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
run() { $PY v2_form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
mkdir -p /tmp/v2
for s in panama melbourne iowa; do
  [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/$DEST/$s; rm -rf $O; mkdir -p $O
  { run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters pallas2 --trig direct --reps 2 --stream 4 --stream-seconds 60 --pallas-pb $pb --pallas-nc $nc --pallas-ng $ng
    [ $s = panama ] && run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters pallas2 --trig direct --reps 2 --nosave --pallas-pb $pb --pallas-nc $nc --pallas-ng $ng --no-pallas-final
    [ $s = panama ] && run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters pallas2 --trig direct --reps 2 --nosave --pallas-pb $pb --pallas-nc $nc --pallas-ng $ng --pallas-final 1
    [ $s = panama ] && [ "${DEST}" != pallas3 ] && run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters pallas2 --trig direct --reps 2 --nosave --pallas-pb $pb --pallas-nc $nc --pallas-ng $ng --pallas-final 2 --pallas-gen 2
    [ $s = panama ] && [ "${DEST}" != pallas3 ] && run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters pallas2 --trig direct --reps 2 --nosave --pallas-pb $pb --pallas-nc $nc --pallas-ng $ng --pallas-final 3
  } > /tmp/v2/${LP}_${s}_${L}.log 2>&1
  gcloud storage cp $O/*.npy $O/timing.json $B/v2/$DEST/$s/$L/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/${LP}_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
  rm -rf $O
done
echo DONE > /tmp/v2/done_${L}_${DEST}; gcloud storage cp /tmp/v2/done_${L}_${DEST} $B/v2/logs/ >/dev/null 2>&1
