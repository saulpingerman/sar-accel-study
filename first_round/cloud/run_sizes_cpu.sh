#!/usr/bin/env bash
# Image-size series on the CPU instance: prepare the Umbra collection for four image sizes, time the
# CPU at the sizes not yet timed, then float64 references and emulated precision settings at each size.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; D=/data; mkdir -p $D/out results/sizes results/v2
URL="https://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks/Panama%20Canal%2C%20Panama/2e00bde3-c6d1-4230-aca0-11cd51c9c998/2023-07-18-02-30-32_UMBRA-04/2023-07-18-02-30-32_UMBRA-04_CPHD.cphd"
[ -f $D/p1.cphd ] || curl -s -o $D/p1.cphd "$URL"
$PY realdata.py prep --cphd $D/p1.cphd --out $D/sz1024.npz --center=-480,560 --dec 8 --patch 256 > /tmp/prep1024.log 2>&1
$PY realdata.py prep --cphd $D/p1.cphd --out $D/sz4096.npz --center=-480,560 --dec 2 --patch 1024 > /tmp/prep4096.log 2>&1
$PY realdata.py prep --cphd $D/p1.cphd --out $D/szfull.npz --center=0,0 --dec 1 --patch 4096 > /tmp/prepfull.log 2>&1
gcloud storage cp $D/sz1024.npz $D/sz4096.npz $B/sizes/ >/dev/null 2>&1
gcloud storage cp $D/szfull.npz $B/sizes/ >/dev/null 2>&1
echo DONE > /tmp/sizes_prep_done
# CPU throughput at the sizes not yet timed (machine otherwise idle here)
O=results/v2/stream_cpu-c4d16_sizes.jsonl; rm -f $O
$PY stream_cpu.py --label cpu-c4d16 --algo ffbp --N 1024 --workers 1,8 --n 6 --out $O >> /tmp/stream_cpu_sizes.log 2>&1
$PY stream_cpu.py --label cpu-c4d16 --algo ffbp --N 8192 --workers 1,8 --n 2 --out $O >> /tmp/stream_cpu_sizes.log 2>&1
$PY stream_cpu.py --label cpu-c4d16 --algo ffbp --N 16384 --workers 1 --n 1 --groups 64 --out $O >> /tmp/stream_cpu_sizes.log 2>&1
gcloud storage cp $O $B/results/v2/ >/dev/null 2>&1
echo DONE > /tmp/sizes_cputime_done
# float64 reference and emulated settings, smallest first
ALL=fp64,fp32,f16,bf16_mm,bf16_mm_l1f32,bf16,f8_mm,f4_mm
$PY sizes.py form --data $D/sz1024.npz --n 1024 --outdir $D/out/sz1024_cpu --label cpu --ref --x64 --reps 0 \
    --bp-policies fp32,fp32_naive,bf16_store,bf16_arith,bf16_acc,bf16_all --ffbp-policies $ALL > /tmp/form1024.log 2>&1
$PY sizes.py form --data $D/sz4096.npz --n 4096 --outdir $D/out/sz4096_cpu --label cpu --ref --x64 --reps 0 --ffbp-policies $ALL > /tmp/form4096.log 2>&1
echo DONE > /tmp/sizes_small_done
$PY sizes.py form --data $D/szfull.npz --n 8192 --outdir $D/out/sz8192_cpu --label cpu --ref --x64 --reps 0 \
    --ffbp-policies fp32,f16,bf16_mm,bf16_mm_l1f32,bf16,f8_mm > /tmp/form8192.log 2>&1
echo DONE > /tmp/sizes_8192_done
$PY sizes.py form --data $D/szfull.npz --n 16384 --outdir $D/out/sz16384_cpu --label cpu --ref --x64 --reps 0 --groups 64 \
    --ffbp-policies fp32,f16,bf16_mm,bf16,f8_mm > /tmp/form16384.log 2>&1
echo DONE > /tmp/sizes_16384_done
