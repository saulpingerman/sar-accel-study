#!/usr/bin/env bash
# B: a 1024 x 1024 crop of the native grid from the full Panama phase history, on the CPU.
set -u
cd ~/sar; PY=~/venv/bin/python; mkdir -p /data/t
[ -f /data/t/b_ref/bp_numba_fp64.npy ] || $PY form.py ref --data /data/panama.npz --outdir /data/t/b_ref --crop 1024,1024 > /tmp/tb_ref.log 2>&1
rm -rf /data/t/b_cpu
$PY form.py ffbp --data /data/panama.npz --outdir /data/t/b_cpu --label cpu --crop 1024,1024 --policies fp32 --filters conv,taps --reps 1 > /tmp/tb_cpu.log 2>&1
$PY form.py ffbp --data /data/panama.npz --outdir /data/t/b_cpu --label cpu --crop 1024,1024 --policies fp32 --filters conv --trig direct --reps 1 >> /tmp/tb_cpu.log 2>&1
$PY metrics.py --ref /data/t/b_ref --test /data/t/b_cpu --out /tmp/tb.json --scene testb --figs /data/t/figs > /tmp/tb_metrics.log 2>&1
echo B > /tmp/v2_testB_done
