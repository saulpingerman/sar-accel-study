#!/usr/bin/env bash
# Full TPU sequence; run under nohup on the instance.  run_tpu.sh <label>
set -u
L=$1; cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
$PY probe.py results/probe_$L.jsonl $L > results/probe_$L.txt 2>&1
$PY bench.py --label $L --sizes 1024,2048,4096,8192 --algos ffbp,pfaczt,rc,matmul --ffbp-configs 64:2,32:2,32:3,16:3 \
    --ffbp-policies fp32,fp32_fast,bf16_mm,bf16 --out results/${L}_main.jsonl --reps 3 > results/${L}_main.txt 2>&1
$PY bench.py --label $L --sizes 1024,2048,4096 --algos pfa,pfamm --policies fp32,bf16_arith --tmax 30 \
    --out results/${L}_lookup.jsonl --reps 2 > results/${L}_lookup.txt 2>&1
$PY bench.py --label $L --sizes 1024 --algos bp --policies fp32,bf16_arith --out results/${L}_lookup.jsonl --reps 1 >> results/${L}_lookup.txt 2>&1
gcloud storage cp $B/q/q_air.npz /tmp/q_air.npz >/dev/null 2>&1
$PY quality.py form --data /tmp/q_air.npz --out /tmp/q_air_$L.npz --n 768 --ffbp-levels 4:2,6:8 --ffbp-T 32 \
    --policies fp32,bf16_arith,bf16_acc > results/${L}_qform.txt 2>&1
gcloud storage cp /tmp/q_air_$L.npz $B/q/ >/dev/null 2>&1
gcloud storage cp results/* $B/results/ >/dev/null 2>&1
echo DONE > results/${L}_done
