#!/usr/bin/env bash
# Full GPU sequence; run under nohup on the instance.  run_gpu.sh <label>
set -u
L=$1; cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false
B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
$PY probe.py results/probe_$L.jsonl $L > results/probe_$L.txt 2>&1
$PY bench_cupy.py --label $L --sizes 1024,2048,4096,8192 --out results/${L}_cuda.jsonl > results/${L}_cuda.txt 2>&1
$PY bench.py --label $L --x64 --sizes 1024,2048,4096 --out results/${L}_jax.jsonl --reps 3 > results/${L}_jax.txt 2>&1
gcloud storage cp $B/s512/s512.npz /tmp/s512.npz >/dev/null 2>&1
$PY quality.py form --data /tmp/s512.npz --out /tmp/s512_$L.npz --n 384 --x64 --ffbp-levels 4:2,3:4 --ffbp-T 32 > results/${L}_qform.txt 2>&1
$PY bench_cupy.py --label $L --quality /tmp/s512.npz --images /tmp/s512_${L}_cuda.npz --n 384 >> results/${L}_qform.txt 2>&1
gcloud storage cp /tmp/s512_$L.npz /tmp/s512_${L}_cuda.npz $B/s512/ >/dev/null 2>&1
gcloud storage cp results/* $B/results/ >/dev/null 2>&1
echo DONE > results/${L}_done
