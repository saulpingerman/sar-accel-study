#!/usr/bin/env bash
# Fused Pallas kernel against the XLA path on a TPU: device time and pipelined throughput, single- and three-pass,
# block sizes swept, Panama first then the other collections with the best block size.  v2_pallas_tpu.sh <label>
set -u
L=$1; cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
~/venv/bin/python -c "import scipy" 2>/dev/null || ~/.local/bin/uv pip install -q --python ~/venv/bin/python scipy
run() { $PY v2_form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
mkdir -p /tmp/v2; s=panama; [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
# 1. block-size sweep, single pass, device time only
for pb in 64 128 256; do
  O=/tmp/v2/pal/$s/pb$pb; rm -rf $O; mkdir -p $O
  run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast --filters pallas --trig direct --reps 2 --nosave --pallas-pb $pb > /tmp/v2/p_${s}_${L}_pb$pb.log 2>&1
  gcloud storage cp $O/timing.json $B/v2/pallas/$s/$L/pb$pb.json >/dev/null 2>&1; gcloud storage cp /tmp/v2/p_${s}_${L}_pb$pb.log $B/v2/logs/ >/dev/null 2>&1
done
BEST=$($PY - <<'PY'
import json, glob
best=None
for f in glob.glob('/tmp/v2/pal/panama/pb*/timing.json'):
    t=json.load(open(f)).get('ffbp/fp32_fast_pallas_direct', {}).get('run_s')
    if t and (best is None or t < best[0]): best=(t, f.split('/pb')[-1].split('/')[0])
print(best[1] if best else 128)
PY
)
echo "best pb $BEST" >> /tmp/v2/p_${s}_${L}_pb$BEST.log
# 2. the measured rows: single- and three-pass with the fused kernel, pipelined, images saved; the XLA direct-ramp rows again in the same session
for s in panama melbourne iowa; do
  [ -f /tmp/v2/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /tmp/v2/ >/dev/null 2>&1
  O=/tmp/v2/pal/$s/best; rm -rf $O; mkdir -p $O
  { run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters pallas --trig direct --reps 2 --stream 4 --stream-seconds 60 --pallas-pb $BEST
    run ffbp --data /tmp/v2/$s.npz --outdir $O --label $L --policies fp32_fast,fp32_high --filters dense --trig direct --reps 2 --nosave
  } > /tmp/v2/p_${s}_${L}.log 2>&1
  gcloud storage cp $O/*.npy $O/timing.json $B/v2/pallas/$s/$L/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/p_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
  rm -rf $O
done
echo DONE > /tmp/v2/done_${L}_pallas; gcloud storage cp /tmp/v2/done_${L}_pallas $B/v2/logs/ >/dev/null 2>&1
