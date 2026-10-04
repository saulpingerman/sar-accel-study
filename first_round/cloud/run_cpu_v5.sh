#!/usr/bin/env bash
# Fifth CPU sequence: multi-process throughput, polar format accuracy over wide scenes, change-detection rescoring.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; mkdir -p results/v2 results/lat results/quality
O=results/v2/stream_cpu-c4d16.jsonl; rm -f $O
for A in ffbp pfaczt pfa_numba; do
  $PY stream_cpu.py --label cpu-c4d16 --algo $A --N 4096 --workers 1,2,4,8 --n 3 --out $O >> /tmp/stream_cpu.log 2>&1
done
$PY stream_cpu.py --label cpu-c4d16 --algo ffbp --N 2048 --workers 1,4,8 --n 4 --out $O >> /tmp/stream_cpu.log 2>&1
$PY stream_cpu.py --label cpu-c4d16 --algo bp_numba --N 4096 --workers 1 --n 2 --out $O >> /tmp/stream_cpu.log 2>&1
gcloud storage cp $O $B/results/v2/ >/dev/null 2>&1
for cfg in "4096 600e3 space" "2048 10e3 air" "4096 10e3 air" "8192 600e3 space"; do
  set -- $cfg
  $PY pfa_lattice.py --N $1 --r0 $2 --out results/lat/pfa_$1_$3.json 2>&1 | grep -v Warn > /tmp/pfa_lat_$1_$3.log
done
for g in air space; do
  $PY quality.py analyze --data /tmp/q_$g.npz --ref /tmp/q_${g}_cpu.npz --test /tmp/q_${g}_cpu.npz --out results/quality/q_${g}_cpu.json > /tmp/qan_$g.log 2>&1
done
tar czf /tmp/v5_results.tgz results/lat/pfa_*.json results/quality/q_air_cpu.json results/quality/q_space_cpu.json $O
gcloud storage cp /tmp/v5_results.tgz $B/results/v2/ >/dev/null 2>&1
echo DONE > /tmp/v5_done
