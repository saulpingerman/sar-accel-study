#!/usr/bin/env bash
# Re-time the factorized algorithm with on-device geometry.  run_ffbp2.sh <label> <sizes> <policies>
set -u
L=$1; cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false
B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
rm -f results/${L}_ffbp2.jsonl results/${L}_done3
$PY bench.py --label $L --sizes $2 --algos ffbp --ffbp-configs 32:3,16:3,32:2 --ffbp-policies $3 \
    --out results/${L}_ffbp2.jsonl --reps 3 > results/${L}_ffbp2.txt 2>&1
gcloud storage cp results/${L}_ffbp2.jsonl $B/results/ >/dev/null 2>&1
echo DONE > results/${L}_done3
