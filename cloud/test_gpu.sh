#!/usr/bin/env bash
# Tests of the second-round kernels on the L4: a 1024 x 1024 crop of the native Panama grid from the full phase history.
set -u
cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
mkdir -p /tmp/v2; cd /tmp/v2
[ -f panama.npz ] || gcloud storage cp $B/v2/data/panama.npz . >/dev/null 2>&1
[ -d b_ref ] || gcloud storage cp -r $B/v2/t/b_ref . >/dev/null 2>&1
cd ~/sar; ~/venv/bin/python -c "import scipy" 2>/dev/null || ~/.local/bin/uv pip install -q --python ~/venv/bin/python scipy
rm -rf /tmp/v2/b_l4
$PY form.py cuda --data /tmp/v2/panama.npz --outdir /tmp/v2/b_l4 --label gpu-l4 --crop 1024,1024 --stores fp32,f16 --reps 2 > /tmp/v2/tb_cuda.log 2>&1
$PY form.py ffbp --data /tmp/v2/panama.npz --outdir /tmp/v2/b_l4 --label gpu-l4 --crop 1024,1024 --policies f16,fp32_fast,fp32 --filters conv,taps,dense --reps 2 > /tmp/v2/tb_ffbp.log 2>&1
$PY form.py ffbp --data /tmp/v2/panama.npz --outdir /tmp/v2/b_l4 --label gpu-l4 --crop 1024,1024 --policies f16 --filters conv --trig direct --reps 2 >> /tmp/v2/tb_ffbp.log 2>&1
$PY metrics.py --ref /tmp/v2/b_ref --test /tmp/v2/b_l4 --out /tmp/v2/tb.json --scene testb > /tmp/v2/tb_metrics.log 2>&1
echo B > /tmp/v2/testB_done
