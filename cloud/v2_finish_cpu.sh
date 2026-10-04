#!/usr/bin/env bash
# Finishing pass on a CPU instance: polar-format edge-sensitivity tests on Panama, then statistics of the new device images.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
export PATH=$HOME/.local/bin:$PATH
sudo mkdir -p /data && sudo chown $USER /data
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d ~/venv ] || uv venv -q --python 3.12 ~/venv
uv pip install -q --python ~/venv/bin/python jax numpy scipy ml_dtypes 2>&1 | tail -1
mkdir -p /data/out/panama/ref /data/t
gcloud storage cp $B/v2/data/panama.npz /data/ >/dev/null 2>&1
gcloud storage cp $B/v2/out/panama/ref/bp_numba_fp64.npy /data/out/panama/ref/ >/dev/null 2>&1
run() { $PY v2_form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
O=/data/t/edge; rm -rf $O; mkdir -p $O
for tp in 4 8; do
  run pfa --data /data/panama.npz --outdir $O --label edge --policies fp32 --resample taps --reps 0 --correct --taps $tp > /tmp/edge_$tp.log 2>&1
  mv $O/pfa_fp32_taps_corr.npy $O/pfa_fp32_taps_corr_t$tp.npy
done
run pfa --data /data/panama.npz --outdir $O --label edge --policies fp32 --resample taps --reps 0 --correct --oversample 3 > /tmp/edge_ov3.log 2>&1
mv $O/pfa_fp32_taps_corr.npy $O/pfa_fp32_taps_corr_ov3.npy
run pfa --data /data/panama.npz --outdir $O --label edge --policies fp32 --resample taps --reps 0 --correct > /tmp/edge_base.log 2>&1
$PY v2_metrics.py --ref /data/out/panama/ref --test $O --out /data/t/edge_metrics.json --scene panama > /tmp/edge_metrics.log 2>&1
gcloud storage cp /data/t/edge_metrics.json $B/v2/results/panama_edge_metrics.json >/dev/null 2>&1
gcloud storage cp /tmp/edge_*.log $B/v2/logs/ >/dev/null 2>&1
echo E > /tmp/edge_done
# statistics of the new images once the device passes have finished
for s in panama melbourne iowa; do
  until gcloud storage ls $B/v2/logs/done_gpu-l4_finish >/dev/null 2>&1 && gcloud storage ls $B/v2/logs/done_tpu-v5e_finish >/dev/null 2>&1 && gcloud storage ls $B/v2/logs/done_tpu-v6e_finish >/dev/null 2>&1; do sleep 60; done
  [ -f /data/out/$s/ref/bp_numba_fp64.npy ] || { mkdir -p /data/out/$s/ref; gcloud storage cp $B/v2/out/$s/ref/bp_numba_fp64.npy /data/out/$s/ref/ >/dev/null 2>&1; }
  N=/data/out/$s/new; rm -rf $N; mkdir -p $N/gpu-l4 $N/tpu-v5e $N/tpu-v6e
  gcloud storage cp $B/v2/out/$s/gpu-l4/ffbp_f16_conv.npy $B/v2/out/$s/gpu-l4/timing.json $N/gpu-l4/ >/dev/null 2>&1
  gcloud storage cp $B/v2/out/$s/tpu-v5e/ffbp_fp32_high_direct.npy $B/v2/out/$s/tpu-v5e/timing.json $N/tpu-v5e/ >/dev/null 2>&1
  gcloud storage cp $B/v2/out/$s/tpu-v6e/ffbp_fp32_high_direct.npy $B/v2/out/$s/tpu-v6e/timing.json $N/tpu-v6e/ >/dev/null 2>&1
  for L in gpu-l4 tpu-v5e tpu-v6e; do
    ls $N/$L/*.npy >/dev/null 2>&1 || continue
    $PY v2_metrics.py --ref /data/out/$s/ref --test $N/$L --out $N/metrics_$L.json --scene $s > /tmp/new_${s}_$L.log 2>&1
    gcloud storage cp $N/metrics_$L.json $B/v2/results/${s}_${L}_new_metrics.json >/dev/null 2>&1
  done
done
echo DONE > /tmp/finish_cpu_done; gcloud storage cp /tmp/finish_cpu_done $B/v2/logs/done_cpu_finish >/dev/null 2>&1
