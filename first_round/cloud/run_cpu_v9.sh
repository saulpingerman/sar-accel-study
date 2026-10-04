#!/usr/bin/env bash
# Same-image comparison: every precision setting against the float64 image of the same collection.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}; mkdir -p results/quality results/real results/figs_same /tmp/umbra
gcloud storage cp $B/q/hdr.npz $B/q/q_air_cpu.npz $B/q/hdr2_tpu-v5e.npz $B/q/hdr2_tpu-v6e.npz $B/q/hdr2_gpu-l4.npz /tmp/ >/dev/null 2>&1
gcloud storage cp $B/real/lock_cpu.npz $B/real/lock_tpu-v5e.npz $B/real/lock_tpu-v6e.npz $B/real/lock_gpu-l4.npz /tmp/umbra/ >/dev/null 2>&1
# measured data
$PY same_image.py --images /tmp/umbra/lock_cpu.npz --ref-tag bp/numba_fp64 --win 9 --out results/real/same_lock_cpu.json \
    --figs results/figs_same/real --prefix real --zoom 560,1040,320 --range-db 45 > /tmp/same_real.log 2>&1
for L in tpu-v5e tpu-v6e gpu-l4; do
  $PY same_image.py --images /tmp/umbra/lock_$L.npz --ref /tmp/umbra/lock_cpu.npz --ref-tag bp/numba_fp64 --win 9 --out results/real/same_lock_$L.json > /tmp/same_real_$L.log 2>&1
done
echo DONE > /tmp/v9_real_done
# uniform scene (images already formed)
$PY same_image.py --images /tmp/q_air_cpu.npz --win 11 --out results/quality/same_q_air_cpu.json > /tmp/same_air.log 2>&1
# six-band dynamic-range scene
$PY quality.py make-data --hdr --strata=0,-10,-20,-30,-40,-50 --cnr-db 60 --K 1024 --geom air --out /tmp/hdr6.npz > /tmp/hdr6_make.log 2>&1
$PY quality.py form --data /tmp/hdr6.npz --out /tmp/hdr6_cpu.npz --n 1024 --spacing 0.17 --x64 --algos bp,ffbp --ffbp-levels "4:2,4:4,4:4@16" \
    --policies fp64,fp32,bf16_store,bf16_arith,bf16_acc,f16_arith > /tmp/hdr6_form.log 2>&1
$PY same_image.py --images /tmp/hdr6_cpu.npz --data /tmp/hdr6.npz --win 13 --out results/quality/same_hdr6_cpu.json \
    --figs results/figs_same/hdr6 --prefix hdr6 --range-db 70 > /tmp/same_hdr6.log 2>&1
# four-band scene, for agreement between emulation and hardware
$PY quality.py form --data /tmp/hdr.npz --out /tmp/hdr2_cpu.npz --n 1024 --spacing 0.17 --x64 --algos bp,ffbp --ffbp-levels "4:2,4:4,4:4@16" --policies fp64,fp32 > /tmp/hdr2_form.log 2>&1
$PY same_image.py --images /tmp/hdr2_cpu.npz --data /tmp/hdr.npz --win 13 --out results/quality/same_hdr2_cpu.json > /tmp/same_hdr2.log 2>&1
for L in tpu-v5e tpu-v6e gpu-l4; do
  $PY same_image.py --images /tmp/hdr2_$L.npz --ref /tmp/hdr2_cpu.npz --data /tmp/hdr.npz --win 13 --out results/quality/same_hdr2_$L.json > /tmp/same_hdr2_$L.log 2>&1
done
# sheets for the comparison page
T=ref,bp_fp32,bp_bf16_store,bp_bf16_arith,bp_bf16_acc,bp_bf16_acc_seq,bp_fp32_naive,bp_bf16_all,ffbp_fp32,ffbp_f16,ffbp_bf16_mm_l1f32,ffbp_bf16_mm,ffbp_bf16,ffbp_f8_mm,ffbp_f4_mm
rm -rf results/sheets_same
$PY sprites.py --dir results/figs_same/real --prefix real --views img,err,coh,loss --tags $T --name real --out results/sheets_same > /tmp/sprites.log 2>&1
$PY sprites.py --dir results/figs_same/hdr6 --prefix hdr6 --views img,err,coh,loss --tags $T --name hdr6 --out results/sheets_same >> /tmp/sprites.log 2>&1
tar czf /tmp/v9_results.tgz results/quality/same_*.json results/real/same_*.json results/figs_same results/sheets_same
gcloud storage cp /tmp/v9_results.tgz $B/results/v2/ >/dev/null 2>&1
echo DONE > /tmp/v9_done
