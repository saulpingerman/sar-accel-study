#!/usr/bin/env bash
# Moving-target refocus sweep for the Panama ship on a CPU instance.
set -u
sudo mkdir -p /data && sudo chown $USER /data
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
[ -d ~/venv ] || uv venv -q --python 3.12 ~/venv
uv pip install -q --python ~/venv/bin/python jax numpy scipy ml_dtypes numba 2>&1 | tail -1
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
gcloud storage cp $B/v2/data/panama.npz /data/ >/dev/null 2>&1
# coarse sweep at the measured bearing, then the best speed refined, then two other bearings for sensitivity
$PY ship_refocus.py --data /data/panama.npz --out /data/ship --speeds=-8:8:0.5 --bearing 158 --heading 0.7968,-0.5947 > /tmp/ship_158.log 2>&1
gcloud storage cp /tmp/ship_158.log /data/ship/sweep_158.json $B/v2/ship/ >/dev/null 2>&1
BEST=$($PY -c "import json; r=json.load(open('/data/ship/sweep_158.json'))['rows']; print(max(r,key=lambda x:x['sharpness'])['speed'])")
LO=$($PY -c "print($BEST-1.0)"); HI=$($PY -c "print($BEST+1.0)")
$PY ship_refocus.py --data /data/panama.npz --out /data/ship/fine --speeds=$LO:$HI:0.1 --bearing 158 --heading 0.7968,-0.5947 > /tmp/ship_fine.log 2>&1
BEST2=$($PY -c "import json; r=json.load(open('/data/ship/fine/sweep_158.json'))['rows']; print(max(r,key=lambda x:x['sharpness'])['speed'])")
$PY ship_refocus.py --data /data/panama.npz --out /data/ship/best --speeds=$BEST2:$BEST2:1 --bearing 158 --heading 0.7968,-0.5947 --save $BEST2 > /tmp/ship_best.log 2>&1
for bh in "148 0.7098,-0.6610" "168 0.8595,-0.5104"; do set -- $bh
  $PY ship_refocus.py --data /data/panama.npz --out /data/ship/b$1 --speeds=-8:8:0.5 --bearing $1 --heading $2 > /tmp/ship_$1.log 2>&1
done
gcloud storage cp -r /data/ship $B/v2/ >/dev/null 2>&1
echo DONE > /tmp/ship_done; gcloud storage cp /tmp/ship_done $B/v2/logs/done_ship >/dev/null 2>&1
