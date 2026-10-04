#!/usr/bin/env bash
# Emulated precision settings at a side of 16384 (no reference), uploaded one image at a time.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; D=/data; mkdir -p $D/out
[ -f $D/szfull.npz ] || gcloud storage cp $B/sizes/szfull.npz $D/ >/dev/null 2>&1
for pol in fp32 bf16_mm f16 bf16 f8_mm; do
  $PY sizes.py form --data $D/szfull.npz --n 16384 --outdir $D/out/sz16384_cpu --label cpu --x64 --reps 0 --groups 64 --ffbp-policies $pol >> /tmp/emu16.log 2>&1
  gcloud storage cp $D/out/sz16384_cpu/ffbp_$pol.npy $D/out/sz16384_cpu/timing.json $B/sizes/out/sz16384_cpuemu/ >/dev/null 2>&1
done
echo DONE > /tmp/emu16_done; gcloud storage cp /tmp/emu16_done $B/sizes/done_emu16 >/dev/null 2>&1
