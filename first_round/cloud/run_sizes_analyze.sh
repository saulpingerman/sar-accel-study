#!/usr/bin/env bash
# Compare every image of one size (CPU emulation and the devices' uploads) with the float64 reference.
#   run_sizes_analyze.sh <N>
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; D=/data; N=$1
T=$D/out/sz${N}_cpu
for L in tpu-v5e tpu-v6e gpu-l4 tpu-v5e-bm tpu-v6e-bm gpu-l4-bm; do
  if gcloud storage ls $B/sizes/out/sz${N}_$L/timing.json >/dev/null 2>&1; then
    mkdir -p $D/out/sz${N}_$L; gcloud storage cp "$B/sizes/out/sz${N}_$L/*" $D/out/sz${N}_$L/ >/dev/null 2>&1
    T=$T,$D/out/sz${N}_$L
  fi
done
$PY sizes.py analyze --ref $D/out/sz${N}_cpu --test $T --win 9 --out results/sizes/sz$N.json --figs results/sizes/figs \
    --fig-tags ffbp/bf16_mm,ffbp/fp32_fast,ffbp/f16,ffbp/f8_mm,ffbp/fp32 > /tmp/an$N.log 2>&1
tar czf /tmp/sizes_results.tgz results/sizes
gcloud storage cp /tmp/sizes_results.tgz results/sizes/sz$N.json $B/sizes/ >/dev/null 2>&1
cat /tmp/an$N.log | cut -c1-220
