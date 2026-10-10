#!/usr/bin/env bash
# Float64 exact-backprojection references on the native grids (large CPU instance).
set -u
sudo mkdir -p /data && sudo chown $USER /data
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
[ -d ~/venv ] || uv venv -q --python 3.12 ~/venv
uv pip install -q --python ~/venv/bin/python jax numpy scipy ml_dtypes numba matplotlib 2>&1 | tail -1
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
for s in panama melbourne iowa; do
  until gcloud storage ls $B/v2/data/$s.npz >/dev/null 2>&1; do sleep 30; done
  sleep 20
  gcloud storage cp $B/v2/data/$s.npz /data/ >/dev/null 2>&1
  REF_OVERSAMPLE=${REF_OVERSAMPLE:-8} $PY form.py ref --data /data/$s.npz --outdir /data/out/$s/ref > /tmp/ref_$s.log 2>&1
  gcloud storage cp /data/out/$s/ref/* $B/v2/out/$s/ref/ >/dev/null 2>&1
  echo $s > /tmp/ref_${s}_done; gcloud storage cp /tmp/ref_${s}_done $B/v2/out/$s/ref_done >/dev/null 2>&1
done
echo DONE > /tmp/v2_ref_done; gcloud storage cp /tmp/v2_ref_done $B/v2/ref_all_done >/dev/null 2>&1
