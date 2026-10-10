#!/usr/bin/env bash
# Second-round stages on an accelerator.  run_accel.sh <label> <tpu|gpu> <stages>
# stages: test, profile, native:<scene>[:<scene>...]
set -u
L=$1; KIND=$2; STAGES=$3
cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
~/venv/bin/python -c "import scipy" 2>/dev/null || ~/.local/bin/uv pip install -q --python ~/venv/bin/python scipy
mkdir -p /tmp/v2/out
has() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
if [ "$KIND" = tpu ]; then POLS=fp32_fast,fp32_high,fp32,bf16_mm; FAST=fp32_fast; PFAPOLS=fp32,fp32_fast; else POLS=f16,fp32_fast,fp32; FAST=f16; PFAPOLS=fp32,f16; fi
run() { $PY form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
if has test; then
  [ -f /tmp/v2/panama.npz ] || gcloud storage cp $B/v2/data/panama.npz /tmp/v2/ >/dev/null 2>&1
  [ -d /tmp/v2/b_ref ] || gcloud storage cp -r $B/v2/t/b_ref /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/out/t_$L; rm -rf $O
  { run ffbp --data /tmp/v2/panama.npz --outdir $O --label $L --crop 1024,1024 --policies $POLS --filters dense,conv --reps 2
    run ffbp --data /tmp/v2/panama.npz --outdir $O --label $L --crop 1024,1024 --policies $FAST --filters dense --trig direct --reps 2
    run pfa --data /tmp/v2/panama.npz --outdir $O --label $L --crop 1024,1024 --policies $PFAPOLS --reps 2
    run pfa --data /tmp/v2/panama.npz --outdir $O --label $L --crop 1024,1024 --policies fp32 --resample taps --reps 2
    [ "$KIND" = gpu ] && run cuda --data /tmp/v2/panama.npz --outdir $O --label $L --crop 1024,1024 --stores fp32,f16 --reps 2
  } > /tmp/v2/t_${L}.log 2>&1
  $PY metrics.py --ref /tmp/v2/b_ref --test $O --out /tmp/v2/t_$L.json --scene testc > /tmp/v2/t_${L}_metrics.log 2>&1
  gcloud storage cp /tmp/v2/t_$L.json /tmp/v2/t_${L}*.log $B/v2/logs/ >/dev/null 2>&1
fi
if has profile; then
  [ -f /tmp/v2/panama.npz ] || gcloud storage cp $B/v2/data/panama.npz /tmp/v2/ >/dev/null 2>&1
  $PY stage_profile.py --data /tmp/v2/panama.npz --label $L --out /tmp/v2/profile_$L.json > /tmp/v2/profile_$L.log 2>&1
  gcloud storage cp /tmp/v2/profile_$L.json /tmp/v2/profile_$L.log $B/v2/logs/ >/dev/null 2>&1
fi
for s in $(echo "$STAGES" | tr ',' '\n' | grep '^native:' | cut -d: -f2- | tr ':' ' '); do
  [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/out/$s/$L; rm -rf $O; mkdir -p $O; gcloud storage cp $B/v2/out/$s/$L/timing.json $O/ >/dev/null 2>&1
  { run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies $POLS --filters dense --reps 1
    run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies $FAST --filters dense --trig direct --reps 1 --nosave --stream 4 --stream-seconds 60
    run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies $FAST --filters conv --reps 1 --nosave
    run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies $PFAPOLS --reps 1 --stream 6 --stream-seconds 45
    run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32 --resample taps --reps 1 --nosave
    if [ "$KIND" = gpu ]; then run cuda --data /tmp/v2/$s.npz --outdir $O --label $L --stores fp32,f16 --reps 0; fi
  } > /tmp/v2/n_${s}_${L}.log 2>&1
  gcloud storage cp -r $O $B/v2/out/$s/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/n_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
  rm -rf $O
done
for s in $(echo "$STAGES" | tr ',' '\n' | grep '^pstream:' | cut -d: -f2- | tr ':' ' '); do
  [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/out/$s/$L; rm -rf $O; mkdir -p $O; gcloud storage cp $B/v2/out/$s/$L/timing.json $O/ >/dev/null 2>&1
  run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32 --reps 1 --nosave --stream 8 --stream-seconds 40 --resample ${PFA_RESAMPLE:-dense} > /tmp/v2/ps_${s}_${L}.log 2>&1
  gcloud storage cp $O/timing.json $B/v2/out/$s/$L/timing.json >/dev/null 2>&1
  gcloud storage cp /tmp/v2/ps_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
done
for s in $(echo "$STAGES" | tr ',' '\n' | grep '^pfa:' | cut -d: -f2- | tr ':' ' '); do
  [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/out/$s/$L; rm -rf $O; mkdir -p $O; gcloud storage cp $B/v2/out/$s/$L/timing.json $O/ >/dev/null 2>&1
  { run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies $PFAPOLS --reps 1 --stream 6 --stream-seconds 45 --resample ${PFA_RESAMPLE:-dense}
    run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32 --resample ${PFA_RESAMPLE2:-taps} --reps 1 --nosave
    run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32 --resample ${PFA_RESAMPLE:-dense} --reps 1 --correct --stream 2 --stream-seconds 20
    run pfa --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32 --resample ${PFA_RESAMPLE2:-taps} --reps 1 --correct --nosave
  } > /tmp/v2/p_${s}_${L}.log 2>&1
  gcloud storage cp -r $O $B/v2/out/$s/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/p_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
  rm -rf $O
done
echo DONE > /tmp/v2/done_${L}${DONE_SUFFIX:-}; gcloud storage cp /tmp/v2/done_${L}${DONE_SUFFIX:-} $B/v2/logs/ >/dev/null 2>&1
