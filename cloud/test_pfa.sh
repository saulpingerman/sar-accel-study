#!/usr/bin/env bash
# Polar format on the 1024 x 1024 crop: orientation check against the float64 reference, on the CPU test instance.
set -u
cd ~/sar; PY=~/venv/bin/python
rm -rf /data/t/b_pfa
for o in "1,-1"; do
  $PY form.py pfa --data /data/panama.npz --outdir /data/t/b_pfa --label cpu --crop 1024,1024 --policies fp32 --reps 1 --orient=$o > /tmp/pfa_$o.log 2>&1
  $PY metrics.py --ref /data/t/b_ref --test /data/t/b_pfa --out /tmp/pfa_$o.json --scene testb 2>&1 | grep pfa | sed "s/^/orient $o: /" | cut -c1-230
done
echo P > /tmp/v2_pfa_done
