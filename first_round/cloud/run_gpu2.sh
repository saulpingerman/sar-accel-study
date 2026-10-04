#!/usr/bin/env bash
# GPU timing with float64 disabled, CUDA kernel at more sizes, and the full-size quality images.
set -u
L=$1; cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false
B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
rm -f results/${L}_done2
$PY bench.py --label $L --sizes 1024,2048,4096,8192 --algos ffbp,pfa,pfaczt,pfamm,rc,matmul --ffbp-configs 64:2,32:2,32:3,16:3 \
    --ffbp-policies fp32,bf16_mm,bf16,f16_mm,f16 --policies fp32,bf16_arith,f16_arith --out results/${L}_main.jsonl --reps 3 > results/${L}_main.txt 2>&1
$PY bench.py --label $L --sizes 1024,2048,4096 --algos bp --policies fp32,bf16_arith,f16_arith --out results/${L}_main.jsonl --reps 2 >> results/${L}_main.txt 2>&1
$PY bench_cupy.py --label $L --sizes 6144,8192 --stores fp32,f16 --out results/${L}_cuda.jsonl >> results/${L}_cuda.txt 2>&1
gcloud storage cp $B/q/q_air.npz /tmp/q_air.npz >/dev/null 2>&1
$PY quality.py form --data /tmp/q_air.npz --out /tmp/q_air_$L.npz --n 768 --ffbp-levels 4:2,6:8 --ffbp-T 32 \
    --policies fp32,fp32_naive,bf16_store,bf16_arith,bf16_acc,f16_arith,f16_acc > results/${L}_qform2.txt 2>&1
$PY bench_cupy.py --label $L --quality /tmp/q_air.npz --images /tmp/q_air_${L}_cuda.npz --n 768 >> results/${L}_qform2.txt 2>&1
gcloud storage cp /tmp/q_air_$L.npz /tmp/q_air_${L}_cuda.npz $B/q/ >/dev/null 2>&1
gcloud storage cp results/* $B/results/ >/dev/null 2>&1
echo DONE > results/${L}_done2
