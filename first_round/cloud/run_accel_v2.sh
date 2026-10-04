#!/usr/bin/env bash
# Second-round accelerator sequence.  run_accel_v2.sh <label> <tpu|gpu> <instance-id> <stages>
# stages: comma list of big, sizes, band, e2e, stream, stream2, pfa, q768, mixed, e1, real
set -u
L=$1; KIND=$2; ID=$3; STAGES=$4
cd ~/sar; PY=~/venv/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false
B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; R=results/v2; mkdir -p $R
gcloud storage cp $B/src.tgz . >/dev/null 2>&1 && tar xzf src.tgz
has() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
if [ "$KIND" = tpu ]; then P16=fp32_fast; MIX=fp32_fast,bf16_mm_l1f32,fp32; else P16=f16; MIX=f16_mm,f16_mm_l1f32,f16,fp32; fi

if has e2e; then
  if [ "$KIND" = gpu ]; then CU="--cuda fp32,f16"; CU8="--cuda fp32"; else CU=""; CU8=""; fi
  $PY e2e.py --label $L --instance $ID --N 4096 --count 200 --policies $P16,fp32 $CU --transfer-probe --out $R/e2e_${L}_$ID.jsonl > $R/e2e_${L}_$ID.txt 2>&1
  $PY e2e.py --label $L --instance $ID --N 8192 --count 30 --skip 5 --policies $P16 $CU8 --out $R/e2e_${L}_$ID.jsonl >> $R/e2e_${L}_$ID.txt 2>&1
  gcloud storage cp $R/e2e_${L}_$ID.jsonl $B/results/v2/ >/dev/null 2>&1
fi
if has stream; then
  if [ "$KIND" = gpu ]; then CU="--cuda fp32"; else CU=""; fi
  O=$R/stream_${L}_$ID.jsonl
  $PY stream.py --label $L --instance $ID --sizes 1024,2048 --batches 1,2,4,8 --policies $P16 --images 48 --min-seconds 12 $CU --out $O > $R/stream_${L}_$ID.txt 2>&1
  $PY stream.py --label $L --instance $ID --sizes 4096 --batches 1,2 --policies $P16 --images 32 --min-seconds 15 $CU --out $O >> $R/stream_${L}_$ID.txt 2>&1
  $PY stream.py --label $L --instance $ID --sizes 4096 --batches 1 --policies fp32 --products complex64 --images 24 --min-seconds 15 --out $O >> $R/stream_${L}_$ID.txt 2>&1
  $PY stream.py --label $L --instance $ID --sizes 8192 --batches 1 --policies $P16 --images 12 --min-seconds 20 $CU --out $O >> $R/stream_${L}_$ID.txt 2>&1
  gcloud storage cp $O $B/results/v2/ >/dev/null 2>&1
fi
if has pfa; then
  if [ "$KIND" = tpu ]; then PF=czt; else PF=czt,sinc; fi
  O=$R/stream_${L}_${ID}pfa.jsonl
  $PY stream.py --label $L --instance $ID --sizes 1024,2048,4096,8192 --batches 1 --policies "" --pfa $PF --images 48 --min-seconds 10 --out $O > $R/pfastream_${L}_$ID.txt 2>&1
  $PY e2e.py --label $L --instance $ID --N 4096 --count 0 --policies "" --transfer-probe --out $R/e2e_${L}_${ID}pfa.jsonl >> $R/pfastream_${L}_$ID.txt 2>&1
  gcloud storage cp $O $R/e2e_${L}_${ID}pfa.jsonl $B/results/v2/ >/dev/null 2>&1
fi
if has band; then
  if [ "$KIND" = tpu ]; then BP=fp32_fast,fp32_high,fp32; BL=0,64,256,1024; else BP=f16,fp32_fast,fp32; BL=0,32,64,256; fi
  $PY bench.py --label $L --sizes 4096,8192 --algos ffbp --ffbp-configs 32:3 --ffbp-blocks $BL --ffbp-policies $BP --out $R/band_${L}_$ID.jsonl --reps 3 > $R/band_${L}_$ID.txt 2>&1
  gcloud storage cp $R/band_${L}_$ID.jsonl $B/results/v2/ >/dev/null 2>&1
fi
if has mixed; then
  $PY bench.py --label $L --sizes 4096,8192 --algos ffbp --ffbp-configs 32:3 --ffbp-policies $MIX --out $R/mixed_$L.jsonl --reps 3 > $R/mixed_$L.txt 2>&1
  gcloud storage cp $R/mixed_$L.jsonl $B/results/v2/ >/dev/null 2>&1
fi
if has q768; then
  gcloud storage cp $B/q/q_air.npz $B/q/hdr.npz /tmp/ >/dev/null 2>&1
  if [ "$KIND" = gpu ]; then AL=bp,pfa,pfaczt,ffbp; else AL=pfaczt,ffbp; fi
  $PY quality.py form --data /tmp/q_air.npz --out /tmp/q768_$L.npz --n 768 --algos $AL --ffbp-levels "2:1,3:3,4:4@32" \
      --policies fp32,bf16_store,bf16_arith > $R/q768_$L.txt 2>&1
  $PY quality.py form --data /tmp/hdr.npz --out /tmp/hdr_$L.npz --n 768 --algos ffbp --ffbp-levels "2:1,3:3,4:4@32" --policies fp32 > $R/hdr_$L.txt 2>&1
  gcloud storage cp /tmp/q768_$L.npz /tmp/hdr_$L.npz $B/q/ >/dev/null 2>&1
fi
if has stream2; then
  if [ "$KIND" = gpu ]; then P2=f16,fp32_fast; CU="--cuda fp32"; else P2=fp32_fast,fp32_high; CU=""; fi
  O=$R/stream_${L}_${ID}s2.jsonl
  $PY stream.py --label $L --instance $ID --sizes 4096 --batches 1 --policies $P2 --images 32 --min-seconds 60 $CU --out $O > $R/stream2_${L}_$ID.txt 2>&1
  $PY stream.py --label $L --instance $ID --sizes 8192 --batches 1 --policies $P2 --images 12 --min-seconds 60 $CU --out $O >> $R/stream2_${L}_$ID.txt 2>&1
  gcloud storage cp $O $B/results/v2/ >/dev/null 2>&1
fi
if has big; then
  if [ "$KIND" = tpu ]; then BP=fp32_fast,fp32_high; else BP=f16,fp32_fast; fi
  O=$R/big_${L}_$ID.jsonl
  $PY bench.py --label $L --sizes 8192 --algos ffbp --ffbp-configs 32:3 --ffbp-groups 16 --ffbp-policies $BP --out $O --reps 2 > $R/big_${L}_$ID.txt 2>&1
  $PY bench.py --label $L --sizes 16384 --algos ffbp --ffbp-configs 32:3 --ffbp-groups 16 --ffbp-policies $BP --out $O --reps 2 >> $R/big_${L}_$ID.txt 2>&1
  gcloud storage cp $O $B/results/v2/ >/dev/null 2>&1
  $PY stream.py --label $L --instance $ID --sizes 16384 --batches 1 --policies $P16 --groups 16 --nhist 1 --depth 2 --images 4 --min-seconds 40 --out $R/stream_${L}_${ID}big.jsonl > $R/streambig_${L}_$ID.txt 2>&1
  gcloud storage cp $R/stream_${L}_${ID}big.jsonl $B/results/v2/ >/dev/null 2>&1
  if [ "$KIND" = gpu ]; then
    $PY bench_cupy.py --label $L --sizes 16384 --stores fp32,f16 --reps 1 --out $R/big_${L}_${ID}cuda.jsonl >> $R/big_${L}_$ID.txt 2>&1
    gcloud storage cp $R/big_${L}_${ID}cuda.jsonl $B/results/v2/ >/dev/null 2>&1
  fi
fi
if has big2; then
  if [ "$KIND" = tpu ]; then BP=fp32_fast,fp32_high; else BP=f16,fp32_fast; fi
  O=$R/big2_${L}_$ID.jsonl; SO=$R/stream_${L}_${ID}big2.jsonl
  $PY bench.py --label $L --sizes 8192 --algos ffbp --ffbp-configs 32:3 --ffbp-groups 64 --ffbp-policies $BP --out $O --reps 2 >> $R/big2_${L}_$ID.txt 2>&1
  $PY bench.py --label $L --sizes 16384 --algos ffbp --ffbp-configs 32:3 --ffbp-groups 64 --ffbp-policies $BP --out $O --reps 2 >> $R/big2_${L}_$ID.txt 2>&1
  gcloud storage cp $O $B/results/v2/ >/dev/null 2>&1
  for D in 2 1; do
    grep -q images_per_s $SO 2>/dev/null && break
    $PY stream.py --label $L --instance $ID --sizes 16384 --batches 1 --policies $P16 --groups 64 --nhist 1 --depth $D --images 4 --min-seconds 40 --out $SO >> $R/streambig2_${L}_$ID.txt 2>&1
  done
  gcloud storage cp $SO $B/results/v2/ >/dev/null 2>&1
fi
if has big4; then
  # standard and bounded-memory programs on one instance, then throughput at the largest size
  O=$R/big4_${L}_$ID.jsonl; SO=$R/stream_${L}_${ID}big2.jsonl
  $PY bench.py --label $L --sizes 8192 --algos ffbp --ffbp-configs 32:3 --ffbp-policies $P16 --out $O --reps 3 > $R/big4_${L}_$ID.txt 2>&1
  $PY bench.py --label $L --sizes 8192 --algos ffbp --ffbp-configs 32:3 --ffbp-groups 64 --ffbp-policies $P16 --out $O --reps 3 >> $R/big4_${L}_$ID.txt 2>&1
  $PY bench.py --label $L --sizes 16384 --algos ffbp --ffbp-configs 32:3 --ffbp-groups 64 --ffbp-policies $P16 --out $O --reps 2 >> $R/big4_${L}_$ID.txt 2>&1
  gcloud storage cp $O $B/results/v2/ >/dev/null 2>&1
  $PY stream.py --label $L --instance $ID --sizes 16384 --batches 1 --policies $P16 --groups 64 --nhist 1 --depth 2 --images 6 --min-seconds 60 --out $SO >> $R/streambig4_${L}_$ID.txt 2>&1
  gcloud storage cp $SO $B/results/v2/ >/dev/null 2>&1
fi
if has big3; then
  SO=$R/stream_${L}_${ID}big2.jsonl
  for D in 2 1; do
    grep -q images_per_s $SO 2>/dev/null && break
    $PY stream.py --label $L --instance $ID --sizes 16384 --batches 1 --policies $P16 --groups 64 --nhist 1 --depth $D --images 4 --min-seconds 40 --out $SO >> $R/streambig3_${L}_$ID.txt 2>&1
  done
  gcloud storage cp $SO $B/results/v2/ >/dev/null 2>&1
fi
if has sizes16; then
  mkdir -p /tmp/sz/out
  gcloud storage cp $B/sizes/szfull.npz /tmp/sz/ >/dev/null 2>&1
  # the CUDA kernel holds every range profile on the device and does not fit at this size
  if [ "$KIND" = tpu ]; then FP=fp32_fast,fp32_high,fp32,bf16_mm; CU=""; else FP=f16,f16_mm,fp32_fast,fp32; CU=""; fi
  $PY sizes.py form --data /tmp/sz/szfull.npz --n 16384 --outdir /tmp/sz/out/sz16384_$L --label $L --ffbp-policies $FP $CU --groups 64 --reps 1 > $R/sizes_16384_${L}_g64.txt 2>&1
  gcloud storage cp -r /tmp/sz/out/sz16384_$L $B/sizes/out/ >/dev/null 2>&1
  rm -rf /tmp/sz/out/sz16384_$L
  # the bounded-memory form at a size where the standard form was also run
  gcloud storage cp $B/sizes/sz4096.npz /tmp/sz/ >/dev/null 2>&1
  $PY sizes.py form --data /tmp/sz/sz4096.npz --n 4096 --outdir /tmp/sz/out/sz4096_${L}-bm --label ${L}-bm --ffbp-policies $FP --groups 64 --reps 1 > $R/sizes_4096_${L}_bm.txt 2>&1
  gcloud storage cp -r /tmp/sz/out/sz4096_${L}-bm $B/sizes/out/ >/dev/null 2>&1
fi
if has sizes; then
  mkdir -p /tmp/sz/out
  until gcloud storage ls $B/sizes/szfull.npz >/dev/null 2>&1; do sleep 30; done
  sleep 60
  gcloud storage cp $B/sizes/sz1024.npz $B/sizes/sz4096.npz $B/sizes/szfull.npz /tmp/sz/ >/dev/null 2>&1
  if [ "$KIND" = tpu ]; then FP=fp32_fast,fp32_high,fp32,bf16_mm; CU=""; else FP=f16,f16_mm,fp32_fast,fp32; CU="--cuda fp32"; fi
  for cfg in "1024 sz1024 0" "4096 sz4096 0" "8192 szfull 0" "16384 szfull 64"; do
    set -- $cfg
    $PY sizes.py form --data /tmp/sz/$2.npz --n $1 --outdir /tmp/sz/out/sz$1_$L --label $L --ffbp-policies $FP $CU --groups $3 --reps 1 > $R/sizes_$1_$L.txt 2>&1
    gcloud storage cp -r /tmp/sz/out/sz$1_$L $B/sizes/out/ >/dev/null 2>&1
    rm -rf /tmp/sz/out/sz$1_$L
  done
fi
if has hdr2; then
  gcloud storage cp $B/q/hdr.npz /tmp/ >/dev/null 2>&1
  $PY quality.py form --data /tmp/hdr.npz --out /tmp/hdr2_$L.npz --n 1024 --spacing 0.17 --algos ffbp --ffbp-levels "4:2,4:4,4:4@16" --policies fp32 > $R/hdr2_$L.txt 2>&1
  gcloud storage cp /tmp/hdr2_$L.npz $B/q/ >/dev/null 2>&1
fi
if has e1; then
  for N in ${E1_SIZES:-4096 8192}; do
    gcloud storage cp $B/lat/lat$N.npz $B/lat/lat${N}_ref_img.npy /tmp/ >/dev/null 2>&1
    if [ "$KIND" = gpu ]; then CU=--cuda; else CU=""; fi
    $PY validate_large.py form --data /tmp/lat$N.npz --ref-image /tmp/lat${N}_ref_img.npy --out /tmp/lat${N}_$L.npz $CU > $R/lat${N}_$L.txt 2>&1
    gcloud storage cp /tmp/lat${N}_$L.npz $B/lat/ >/dev/null 2>&1
    rm -f /tmp/lat$N.npz /tmp/lat${N}_ref_img.npy
  done
fi
if has real; then
  gcloud storage cp $B/real/lock1.npz $B/real/lock2.npz $B/real/shift.txt /tmp/ >/dev/null 2>&1
  SH=$(awk '{print $2}' /tmp/shift.txt)
  if [ "$KIND" = gpu ]; then BP=fp32,bf16_store,bf16_arith,bf16_acc; CU=--cuda; else BP=""; CU=""; fi
  $PY realdata.py form --data /tmp/lock1.npz,/tmp/lock2.npz --out /tmp/lock_$L.npz --shift="$SH" --policies "$BP" $CU \
      --ffbp-policies fp32,fp32_high,fp32_fast,bf16_mm,bf16,bf16_mm_l1f32,f16_mm,f16 > $R/real_$L.txt 2>&1
  gcloud storage cp /tmp/lock_$L.npz $B/real/ >/dev/null 2>&1
fi
gcloud storage cp $R/*.txt $B/results/v2/ >/dev/null 2>&1
echo DONE > $R/done_${L}_$ID
gcloud storage cp $R/done_${L}_$ID $B/results/v2/ >/dev/null 2>&1
