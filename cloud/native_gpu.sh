#!/usr/bin/env bash
# Native-size images and timings on the L4 for one scene.   native_gpu.sh <scene>
set -u
s=$1; cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
mkdir -p /tmp/v2/out/$s; cd /tmp/v2
[ -f $s.npz ] || gcloud storage cp $B/v2/data/$s.npz . >/dev/null 2>&1
cd ~/sar; O=/tmp/v2/out/$s/gpu-l4
$PY form.py cuda --data /tmp/v2/$s.npz --outdir $O --label gpu-l4 --stores fp32,f16 --reps 1 --stream 3 --stream-seconds 60 > /tmp/v2/n_${s}_cuda.log 2>&1
$PY form.py ffbp --data /tmp/v2/$s.npz --outdir $O --label gpu-l4 --policies f16,fp32_fast,fp32 --filters dense --reps 1 > /tmp/v2/n_${s}_ffbp.log 2>&1
$PY form.py ffbp --data /tmp/v2/$s.npz --outdir $O --label gpu-l4 --policies f16 --filters conv --reps 1 >> /tmp/v2/n_${s}_ffbp.log 2>&1
$PY form.py ffbp --data /tmp/v2/$s.npz --outdir $O --label gpu-l4 --policies f16 --filters dense --trig direct --reps 1 --nosave >> /tmp/v2/n_${s}_ffbp.log 2>&1
gcloud storage cp -r $O $B/v2/out/$s/ >/dev/null 2>&1
gcloud storage cp /tmp/v2/n_${s}_*.log $B/v2/logs/ >/dev/null 2>&1
echo $s > /tmp/v2/native_${s}_done
