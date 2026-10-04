#!/usr/bin/env bash
# Shorter GPU sequence for the T4: primitives, CUDA kernel, and the best JAX configurations.
set -u
L=$1; cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false
B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
$PY probe.py results/probe_$L.jsonl $L > results/probe_$L.txt 2>&1
$PY bench_cupy.py --label $L --sizes 1024,2048,4096 --stores fp32,f16,bf16 --out results/${L}_cuda.jsonl > results/${L}_cuda.txt 2>&1
$PY bench.py --label $L --sizes 1024,2048,4096 --algos ffbp,pfa,pfaczt,pfamm,rc,matmul --ffbp-configs 32:3,16:3 \
    --ffbp-policies fp32,bf16_mm,f16_mm,f16 --policies fp32,f16_arith --out results/${L}_main.jsonl --reps 3 > results/${L}_main.txt 2>&1
gcloud storage cp results/* $B/results/ >/dev/null 2>&1
echo DONE > results/${L}_done
