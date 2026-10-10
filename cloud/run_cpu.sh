#!/usr/bin/env bash
# Second-round CPU stages on the 16-vCPU instance.  run_cpu.sh <label> <stages>   stages: test, native:<scene>[:...]
set -u
L=$1; STAGES=$2
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
has() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
run() { $PY form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
if has test; then
  O=/data/t/c_$L; rm -rf $O
  { run ffbp --data /data/panama.npz --outdir $O --label $L --crop 1024,1024 --policies fp32 --filters conv,taps --trig direct --reps 1
    run ffbp --data /data/panama.npz --outdir $O --label $L --crop 1024,1024 --x64 --policies fp64 --filters conv --trig direct --reps 1
    run pfa --data /data/panama.npz --outdir $O --label $L --crop 1024,1024 --policies fp32 --reps 1
    run pfa --data /data/panama.npz --outdir $O --label $L --crop 1024,1024 --policies fp32 --resample taps --reps 1
  } > /tmp/v2/t_${L}.log 2>&1
  $PY metrics.py --ref /data/t/b_ref --test $O --out /tmp/v2/t_$L.json --scene testc > /tmp/v2/t_${L}_metrics.log 2>&1
  gcloud storage cp /tmp/v2/t_$L.json /tmp/v2/t_${L}*.log $B/v2/logs/ >/dev/null 2>&1
fi
for s in $(echo "$STAGES" | tr ',' '\n' | grep '^native:' | cut -d: -f2- | tr ':' ' '); do
  O=/data/out/$s/$L; rm -rf $O; mkdir -p $O
  { run ffbp --data /data/$s.npz --outdir $O --label $L --policies fp32 --filters conv --trig direct --reps 1
    run ffbp --data /data/$s.npz --outdir $O --label $L --x64 --policies fp64 --filters conv --trig direct --reps 1
    run pfa --data /data/$s.npz --outdir $O --label $L --policies fp32 --reps 1 --stream 4 --stream-seconds 60
    run pfa --data /data/$s.npz --outdir $O --label $L --policies fp32 --resample taps --reps 1 --nosave
    $PY cpu_stream.py --data /data/$s.npz --label $L --workers 1,2,4 --n 1 --filt conv --trig direct --out $O/cpu_stream.jsonl 2>&1 | grep -v Warning
  } > /tmp/v2/n_${s}_${L}.log 2>&1
  gcloud storage cp -r $O $B/v2/out/$s/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/n_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
done
echo DONE > /tmp/v2/done_$L; gcloud storage cp /tmp/v2/done_$L $B/v2/logs/ >/dev/null 2>&1
