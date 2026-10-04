#!/usr/bin/env bash
# Runs on an accelerator instance: Python environment plus code from the bucket.
#   accel_setup.sh <tpu|gpu> [minutes until forced shutdown]
set -euo pipefail
kind=$1
sudo shutdown -h +"${2:-60}" >/dev/null 2>&1 || true
B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
[ -d ~/venv ] || uv venv -q --python 3.12 ~/venv
if [ "$kind" = tpu ]; then
  uv pip install -q --python ~/venv/bin/python "jax[tpu]" numpy scipy ml_dtypes \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
else
  uv pip install -q --python ~/venv/bin/python "jax[cuda12]" numpy scipy ml_dtypes cupy-cuda12x
fi
mkdir -p ~/sar/results && cd ~/sar
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
~/venv/bin/python -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())"
