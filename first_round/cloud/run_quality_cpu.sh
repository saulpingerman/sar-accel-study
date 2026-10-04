#!/usr/bin/env bash
# Float64 references and every precision policy for both geometries, on the CPU instance.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
rm -f /tmp/quality_done
for g in air space; do
  $PY quality.py form --data /tmp/q_$g.npz --out /tmp/q_${g}_cpu.npz --n 768 --x64 --oracle --numba \
      --ffbp-levels "4:2,6:8@32;4:2,3:4@64;4:2,4:4,3:3@16;2:1,3:3,4:4@32" > /tmp/qform_$g.log 2>&1
  $PY quality.py analyze --data /tmp/q_$g.npz --ref /tmp/q_${g}_cpu.npz --test /tmp/q_${g}_cpu.npz --out results/q_${g}_cpu.json > /tmp/qan_$g.log 2>&1
  $PY figs.py --images /tmp/q_${g}_cpu.npz --out results/figs_$g > /tmp/figs_$g.log 2>&1
done
gcloud storage cp /tmp/q_air_cpu.npz $B/q/ >/dev/null 2>&1
tar czf results/figs.tgz -C results figs_air figs_space
gcloud storage cp results/q_*_cpu.json results/figs.tgz $B/results/ >/dev/null 2>&1
echo DONE > /tmp/quality_done
