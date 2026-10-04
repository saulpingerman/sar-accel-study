#!/usr/bin/env bash
# Measured-data test on the CPU instance: prepare the lock-complex patch from both passes,
# register, form at every precision, and score against float64 backprojection.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; U=/tmp/umbra; CEN="-480,560"
rm -f $U/real_done
i=1; for f in $(ls $U/*.cphd); do $PY realdata.py prep --cphd $f --out $U/lock$i.npz --center="$CEN" > $U/prep$i.log 2>&1; i=$((i+1)); done
$PY realdata.py form --data $U/lock1.npz,$U/lock2.npz --out $U/lock_ref0.npz --policies "" --ffbp-T 0 --numba --x64 > $U/form0.log 2>&1
$PY realdata.py register --images $U/lock_ref0.npz > $U/register.json 2> $U/register.log
SH=$($PY -c "import json;d=json.load(open('$U/register.json'));print('%.4f,%.4f'%tuple(d['shift_m']))")
echo "shift $SH" > $U/shift.txt
$PY realdata.py form --data $U/lock1.npz,$U/lock2.npz --out $U/lock_cpu.npz --shift="$SH" --numba --x64 > $U/form1.log 2>&1
$PY realdata.py register --images $U/lock_cpu.npz > $U/register_after.json 2>> $U/register.log
mkdir -p results/real
$PY realdata.py analyze --images $U/lock_cpu.npz --out results/real/lock_cpu.json --figs results/real/figs > $U/analyze.log 2>&1
gcloud storage cp $U/lock1.npz $U/lock2.npz $B/real/ >/dev/null 2>&1
tar czf results/real/figs.tgz -C results/real figs
gcloud storage cp results/real/lock_cpu.json results/real/figs.tgz $U/shift.txt $U/register.json $U/register_after.json $U/lock_cpu.npz $B/real/ >/dev/null 2>&1
echo DONE > $U/real_done
