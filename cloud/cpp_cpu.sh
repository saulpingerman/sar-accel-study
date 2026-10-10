#!/usr/bin/env bash
# Factorized backprojection with the C++/OpenMP kernels on the CPU instance, three scenes.
#   cpp_cpu.sh [DEST]      results to gs://.../v2/<DEST>/<scene>/cpu-c4d16/ (timing.json, image, logs)
# A thread-count comparison on Panama first (16 hardware threads against the 8 cores), then the three scenes
# with the better setting, each with the pipelined (sequential) loop.
set -u
L=cpu-c4d16; DEST=${1:-cpp}; cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
run() { $PY form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
mkdir -p /tmp/v2 /data
for s in panama melbourne iowa; do [ -f /data/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /data/ >/dev/null 2>&1; done
s=panama
for th in 16 8; do
  O=/tmp/v2/$DEST/$s/th$th; rm -rf $O; mkdir -p $O
  OMP_NUM_THREADS=$th OMP_PLACES=cores OMP_PROC_BIND=close run ffbpcpu --data /data/$s.npz --outdir $O --label $L --reps 1 --nosave > /tmp/v2/${DEST}_${s}_${L}_th$th.log 2>&1
  gcloud storage cp $O/timing.json $B/v2/$DEST/$s/$L/th$th.json >/dev/null 2>&1; gcloud storage cp /tmp/v2/${DEST}_${s}_${L}_th$th.log $B/v2/logs/ >/dev/null 2>&1
done
TH=$($PY - <<'PYEOF'
import json, glob
best = None
for f in glob.glob('/tmp/v2/*/panama/th*/timing.json'):
    t = json.load(open(f)).get('ffbp/fp32_cpp')
    if t and (best is None or t['run_s'] < best[0]):
        best = (t['run_s'], f.split('/th')[-1].split('/')[0])
print(best[1] if best else 16)
PYEOF
)
echo "threads $TH"
for s in panama melbourne iowa; do
  O=/data/out_$DEST/$s/$L; rm -rf $O; mkdir -p $O
  OMP_NUM_THREADS=$TH OMP_PLACES=cores OMP_PROC_BIND=close run ffbpcpu --data /data/$s.npz --outdir $O --label $L --reps 2 --stream 2 --stream-seconds 60 > /tmp/v2/${DEST}_${s}_${L}.log 2>&1
  gcloud storage cp $O/*.npy $O/timing.json $B/v2/$DEST/$s/$L/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/${DEST}_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
done
OMP_PLACES=cores OMP_PROC_BIND=close $PY ubench_cpu.py --out /tmp/v2/ubench_cpu-c4d16.json > /tmp/v2/ubench_${L}.log 2>&1
gcloud storage cp /tmp/v2/ubench_cpu-c4d16.json $B/v2/$DEST/ubench_cpu-c4d16.json >/dev/null 2>&1; gcloud storage cp /tmp/v2/ubench_${L}.log $B/v2/logs/ >/dev/null 2>&1
echo DONE > /tmp/v2/done_${L}_${DEST}; gcloud storage cp /tmp/v2/done_${L}_${DEST} $B/v2/logs/ >/dev/null 2>&1
