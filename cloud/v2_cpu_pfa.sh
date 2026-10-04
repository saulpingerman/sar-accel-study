#!/usr/bin/env bash
# Corrected polar format on the CPU instance for the three scenes (after the main CPU program).
set -u
L=cpu-c4d16; cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
run() { $PY v2_form.py "$@" 2>&1 | grep -v "^WARNING\|Warning" ; }
for s in panama melbourne iowa; do
  O=/data/out/$s/$L; mkdir -p $O
  { run pfa --data /data/$s.npz --outdir $O --label $L --policies fp32 --resample taps --reps 1 --correct
    run pfa --data /data/$s.npz --outdir $O --label $L --policies fp32 --reps 1 --correct --nosave
  } > /tmp/v2/pc_${s}_${L}.log 2>&1
  gcloud storage cp $O/pfa_fp32_taps_corr.npy $O/timing.json $B/v2/out/$s/$L/ >/dev/null 2>&1
  gcloud storage cp /tmp/v2/pc_${s}_${L}.log $B/v2/logs/ >/dev/null 2>&1
done
echo DONE > /tmp/v2/done_${L}_pfa; gcloud storage cp /tmp/v2/done_${L}_pfa $B/v2/logs/ >/dev/null 2>&1
