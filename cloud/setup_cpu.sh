#!/usr/bin/env bash
# CPU instance for the second round: environment and the three Umbra collections on their native grids.
set -u
sudo mkdir -p /data && sudo chown $USER /data
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
[ -d ~/venv ] || uv venv -q --python 3.12 ~/venv
uv pip install -q --python ~/venv/bin/python jax numpy scipy ml_dtypes numba sarpy matplotlib 2>&1 | tail -1
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
U="https://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks"
get() {  # name, url-encoded key prefix (without _CPHD.cphd)
  [ -f /data/$1.npz ] && return
  curl -s -o /data/$1.cphd "$U/$2_CPHD.cphd"; curl -s -o /data/$1.sicd "$U/$2_SICD.nitf"
  $PY prep.py --cphd /data/$1.cphd --sicd /data/$1.sicd --out /data/$1.npz --name $1 > /data/$1.info 2>&1
  rm -f /data/$1.sicd
  gcloud storage cp /data/$1.npz /data/$1.info $B/v2/data/ >/dev/null 2>&1
}
get panama "Panama%20Canal%2C%20Panama/2e00bde3-c6d1-4230-aca0-11cd51c9c998/2023-07-18-02-30-32_UMBRA-04/2023-07-18-02-30-32_UMBRA-04"
echo PANAMA > /tmp/v2_panama_done
get melbourne "Melbourne%2C%20Australia/02a3d6c7-d9e1-4a50-be13-5b2112a3ec1e/2023-02-08-11-51-12_UMBRA-04/2023-02-08-11-51-12_UMBRA-04"
get iowa "Johnston%2C%20Iowa%20Farm/05fd0504-7fec-4e88-97f8-e55fa3d68b09/2023-10-20-15-53-21_UMBRA-04/2023-10-20-15-53-21_UMBRA-04"
echo DONE > /tmp/v2_setup_done
