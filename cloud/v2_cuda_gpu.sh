#!/usr/bin/env bash
# CUDA factorized backprojection on the L4: group-size check on Panama, then the measured row (float32) pipelined on
# the three collections with images saved, plus the conv baselines in the same session.
set -u
L=gpu-l4; DEST=${1:-cuda}; cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
export PATH=/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda   # nvcc for the tensor-core kernel
which g++ >/dev/null 2>&1 || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q g++ >/dev/null 2>&1   # nvcc's host compiler (not on the image)
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
~/venv/bin/python -c "import cupy" 2>/dev/null || ~/.local/bin/uv pip install -q --python ~/venv/bin/python cupy-cuda12x
run() { $PY v2_form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
mkdir -p /tmp/v2; s=panama; [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
for ng in $([ "$DEST" = cuda ] && echo 4 8 || echo 8); do
  O=/tmp/v2/$DEST/$s/ng$ng; rm -rf $O; mkdir -p $O
  run ffbpcuda --data /tmp/v2/$s.npz --outdir $O --label $L --reps 2 --nosave --pallas-ng $ng > /tmp/v2/${DEST}_${s}_${L}_ng$ng.log 2>&1
  gcloud storage cp $O/timing.json $B/v2/$DEST/$s/$L/ng$ng.json >/dev/null 2>&1; gcloud storage cp /tmp/v2/${DEST}_${s}_${L}_ng$ng.log $B/v2/logs/ >/dev/null 2>&1
done
NG=$($PY - <<'PY'
import json, glob
best=None
for f in glob.glob('/tmp/v2/$DEST/panama/ng*/timing.json'):
    t=json.load(open(f)).get('ffbp/fp32_cuda', {}).get('run_s')
    if t and (best is None or t < best[0]): best=(t, f.split('/ng')[-1].split('/')[0])
print(best[1] if best else 8)
PY
)
# the group-size sweep is only needed once
[ "$DEST" = cuda ] || NG=${NG_FIXED:-8}
for s in panama melbourne iowa; do
  [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/$DEST/$s/best; rm -rf $O; mkdir -p $O
  { run ffbpcuda --data /tmp/v2/$s.npz --outdir $O --label $L --reps 2 --stream 4 --stream-seconds 60 --pallas-ng $NG --cuda-final ${FINALS:-fp32,f16tc}
    [ -n "${SKIPCONV:-}" ] || run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies f16,fp32 --filters conv --reps 2 --nosave
  } > /tmp/v2/${DEST}_${s}_${L}.log 2>&1
  gcloud storage cp $O/*.npy $O/timing.json $B/v2/$DEST/$s/$L/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/${DEST}_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
  rm -rf $O
done
echo DONE > /tmp/v2/done_${L}_${DEST}; gcloud storage cp /tmp/v2/done_${L}_${DEST} $B/v2/logs/ >/dev/null 2>&1
