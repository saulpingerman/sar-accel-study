#!/usr/bin/env bash
# Finishing pass on the L4: pipelined loops for the selected configurations, saving the float16 convolution image.
set -u
L=gpu-l4; cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
~/venv/bin/python -c "import scipy" 2>/dev/null || ~/.local/bin/uv pip install -q --python ~/venv/bin/python scipy
run() { $PY form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
for s in panama melbourne iowa; do
  mkdir -p /tmp/v2; [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/out/$s/$L; rm -rf $O; mkdir -p $O; gcloud storage cp $B/v2/out/$s/$L/timing.json $O/ >/dev/null 2>&1
  { run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies f16 --filters conv --reps 1 --stream 4 --stream-seconds 60
    run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32 --filters conv --reps 1 --nosave --stream 4 --stream-seconds 60
    run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32 --resample taps --reps 1 --nosave --stream 6 --stream-seconds 40
    run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32 --resample taps --reps 1 --nosave --correct --stream 6 --stream-seconds 40
    run cuda --data /tmp/v2/$s.npz --outdir $O --label $L --stores f16 --reps 0 --stream 3 --stream-seconds 60
    run cuda --data /tmp/v2/$s.npz --outdir $O --label $L --stores fp32 --reps 0 --stream 2 --stream-seconds 60
  } > /tmp/v2/f_${s}_${L}.log 2>&1
  gcloud storage cp $O/ffbp_f16_conv.npy $O/timing.json $B/v2/out/$s/$L/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/f_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
  rm -rf $O
done
echo DONE > /tmp/v2/done_${L}_finish; gcloud storage cp /tmp/v2/done_${L}_finish $B/v2/logs/ >/dev/null 2>&1
