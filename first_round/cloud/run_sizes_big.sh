#!/usr/bin/env bash
# The two full-collection sizes of the image-size series on a large CPU instance: float64 reference and
# emulated precision settings at 8192 and 16384, each followed by its comparison with the device images.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; D=/data; mkdir -p $D/out results/sizes
[ -f $D/szfull.npz ] || gcloud storage cp $B/sizes/szfull.npz $D/ >/dev/null 2>&1
$PY sizes.py form --data $D/szfull.npz --n 8192 --outdir $D/out/sz8192_cpu --label cpu --ref --x64 --reps 0 \
    --ffbp-policies fp32,f16,bf16_mm,bf16_mm_l1f32,bf16,f8_mm > /tmp/form8192.log 2>&1
bash cloud/run_sizes_analyze.sh 8192 > /tmp/analyze8192.log 2>&1
echo DONE > /tmp/sizes_8192_done
$PY sizes.py form --data $D/szfull.npz --n 16384 --outdir $D/out/sz16384_cpu --label cpu --ref --x64 --reps 0 --groups 64 \
    --ffbp-policies fp32,f16,bf16_mm,bf16,f8_mm > /tmp/form16384.log 2>&1
bash cloud/run_sizes_analyze.sh 16384 > /tmp/analyze16384.log 2>&1
echo DONE > /tmp/sizes_16384_done
gcloud storage cp /tmp/sizes_16384_done $B/sizes/done_16384 >/dev/null 2>&1
