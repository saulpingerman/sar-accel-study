#!/usr/bin/env bash
# TPU7x: primitives, device times, sustained throughput at four sizes, and the size-series images on the measured data.
set -u
cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false
B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; R=results/v2; mkdir -p $R results/raw /tmp/sz/out
L=tpu-7x; ID=i20
up() { gcloud storage cp "$@" $B/results/v2/ >/dev/null 2>&1; }
# 1. primitives and matrix-product accuracy
$PY probe.py results/raw/probe_$L.jsonl $L > results/raw/probe_$L.txt 2>&1
$PY probe2.py > results/raw/probe2_$L.txt 2>&1
gcloud storage cp results/raw/probe_$L.jsonl results/raw/probe_$L.txt results/raw/probe2_$L.txt $B/results/raw7x/ >/dev/null 2>&1
# 2. device times, standard program, then the largest size in both programs
O=$R/bench_${L}_$ID.jsonl
$PY bench.py --label $L --sizes 1024,2048,4096,8192 --algos ffbp --ffbp-configs 32:3 --ffbp-policies fp32_fast,fp32_high,fp32 --out $O --reps 3 > $R/bench_${L}_$ID.txt 2>&1
up $O
$PY bench.py --label $L --sizes 16384 --algos ffbp --ffbp-configs 32:3 --ffbp-policies fp32_fast,fp32_high --out $O --reps 2 >> $R/bench_${L}_$ID.txt 2>&1
$PY bench.py --label $L --sizes 16384 --algos ffbp --ffbp-configs 32:3 --ffbp-groups 64 --ffbp-policies fp32_fast,fp32_high --out $O --reps 2 >> $R/bench_${L}_$ID.txt 2>&1
up $O $R/bench_${L}_$ID.txt
# 3. sustained throughput
SO=$R/stream_${L}_$ID.jsonl
$PY stream.py --label $L --instance $ID --sizes 1024,2048 --batches 1 --policies fp32_fast --images 48 --min-seconds 12 --out $SO > $R/stream_${L}_$ID.txt 2>&1
$PY stream.py --label $L --instance $ID --sizes 4096 --batches 1,2 --policies fp32_fast --images 32 --min-seconds 15 --out $SO >> $R/stream_${L}_$ID.txt 2>&1
$PY stream.py --label $L --instance $ID --sizes 4096 --batches 1 --policies fp32_high --products complex64 --images 24 --min-seconds 15 --out $SO >> $R/stream_${L}_$ID.txt 2>&1
$PY stream.py --label $L --instance $ID --sizes 8192 --batches 1 --policies fp32_fast,fp32_high --products complex64 --images 12 --min-seconds 20 --out $SO >> $R/stream_${L}_$ID.txt 2>&1
up $SO
SB=$R/stream_${L}_${ID}big2.jsonl
$PY stream.py --label $L --instance $ID --sizes 16384 --batches 1 --policies fp32_fast --groups 64 --nhist 1 --depth 2 --images 6 --min-seconds 60 --out $SB >> $R/stream_${L}_$ID.txt 2>&1
up $SB
SS=$R/stream_${L}_${ID}std16.jsonl
$PY stream.py --label $L --instance $ID --sizes 16384 --batches 1 --policies fp32_fast --nhist 1 --depth 2 --images 6 --min-seconds 60 --out $SS >> $R/stream_${L}_$ID.txt 2>&1
up $SS $R/stream_${L}_$ID.txt
# 4. size-series images on the measured data
gcloud storage cp $B/sizes/sz1024.npz $B/sizes/sz4096.npz $B/sizes/szfull.npz /tmp/sz/ >/dev/null 2>&1
for cfg in "1024 sz1024 0" "4096 sz4096 0" "8192 szfull 0" "16384 szfull 64"; do
  set -- $cfg
  $PY sizes.py form --data /tmp/sz/$2.npz --n $1 --outdir /tmp/sz/out/sz$1_$L --label $L --ffbp-policies fp32_fast,fp32_high,fp32,bf16_mm --groups $3 --reps 1 > $R/sizes_$1_$L.txt 2>&1
  gcloud storage cp -r /tmp/sz/out/sz$1_$L $B/sizes/out/ >/dev/null 2>&1
  up $R/sizes_$1_$L.txt
  rm -rf /tmp/sz/out/sz$1_$L
done
echo DONE > $R/done_${L}_$ID; up $R/done_${L}_$ID
