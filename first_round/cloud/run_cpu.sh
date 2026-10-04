#!/usr/bin/env bash
# CPU timing sequence; run under nohup on the instance.  run_cpu.sh <label>
# float32 and 16-bit work is timed with float64 disabled; float64 gets its own pass.
set -u
L=$1; cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
rm -f results/${L}_main.jsonl results/${L}_x64.jsonl results/${L}_done
$PY probe.py results/probe_$L.jsonl $L > results/probe_$L.txt 2>&1
$PY bench.py --label $L --numba --sizes 1024,2048,4096 --algos ffbp,pfa,pfaczt,pfamm,rc,matmul --ffbp-configs 64:2,32:2,32:3,16:3 \
    --ffbp-policies fp32,bf16_mm --policies fp32 --tmax 400 --out results/${L}_main.jsonl --reps 2 > results/${L}_main.txt 2>&1
$PY bench.py --label $L --sizes 1024,2048 --algos bp --policies fp32,bf16_arith --tmax 400 --out results/${L}_main.jsonl --reps 1 >> results/${L}_main.txt 2>&1
$PY bench.py --label $L --x64 --numba --sizes 1024,2048 --algos ffbp,pfa --ffbp-policies fp64 --policies fp64 --tmax 400 \
    --out results/${L}_x64.jsonl --reps 1 > results/${L}_x64.txt 2>&1
gcloud storage cp results/* $B/results/ >/dev/null 2>&1
echo DONE > results/${L}_done
