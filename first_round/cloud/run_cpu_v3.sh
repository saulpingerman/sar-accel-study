#!/usr/bin/env bash
# Third CPU sequence: regenerate the simulated-scene quality results and figures with the current code.
# Runs after the measured-data job.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
until [ -f /tmp/umbra/real_done ]; do sleep 20; done
mkdir -p results/quality
gcloud storage cp $B/q/q_air.npz $B/q/q_space.npz /tmp/ >/dev/null 2>&1
for g in air space; do
  $PY quality.py form --data /tmp/q_$g.npz --out /tmp/q_${g}_cpu.npz --n 768 --x64 --oracle --numba \
      --ffbp-levels "2:1,3:3,4:4@32;4:2,6:8@32;4:2,4:4,3:3@16" > /tmp/qform_$g.log 2>&1
  $PY quality.py analyze --data /tmp/q_$g.npz --ref /tmp/q_${g}_cpu.npz --test /tmp/q_${g}_cpu.npz --out results/quality/q_${g}_cpu.json > /tmp/qan_$g.log 2>&1
  $PY figs.py --images /tmp/q_${g}_cpu.npz --out results/figs_$g > /tmp/figs_$g.log 2>&1
done
$PY ccd_hdr.py --data /tmp/hdr.npz --images /tmp/hdr_cpu.npz --out results/quality/hdr_cpu.json > /tmp/hdr_an.log 2>&1
$PY figs.py --images /tmp/hdr_cpu.npz --out results/figs_hdr --range-db 60 > /tmp/figs_hdr.log 2>&1
tar czf results/figs_sim.tgz -C results figs_air figs_space figs_hdr
gcloud storage cp results/quality/q_air_cpu.json results/quality/q_space_cpu.json results/quality/hdr_cpu.json results/figs_sim.tgz $B/results/v2/ >/dev/null 2>&1
gcloud storage cp /tmp/q_air_cpu.npz /tmp/hdr_cpu.npz $B/q/ >/dev/null 2>&1
echo DONE > /tmp/v3_done
