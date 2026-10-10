#!/usr/bin/env bash
# The float64 reference against the vendor's SICD image, three collections.   sicd_compare.sh
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
U="https://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks"
declare -A KEY
KEY[panama]="Panama%20Canal%2C%20Panama/2e00bde3-c6d1-4230-aca0-11cd51c9c998/2023-07-18-02-30-32_UMBRA-04/2023-07-18-02-30-32_UMBRA-04"
KEY[melbourne]="Melbourne%2C%20Australia/02a3d6c7-d9e1-4a50-be13-5b2112a3ec1e/2023-02-08-11-51-12_UMBRA-04/2023-02-08-11-51-12_UMBRA-04"
KEY[iowa]="Johnston%2C%20Iowa%20Farm/05fd0504-7fec-4e88-97f8-e55fa3d68b09/2023-10-20-15-53-21_UMBRA-04/2023-10-20-15-53-21_UMBRA-04"
declare -A CROPS
CROPS[panama]="locks:3400,6700,512,512;port:9500,1150,512,512;ships:6400,2650,512,512;corner:11400,8000,512,512"
CROPS[melbourne]=""
CROPS[iowa]=""
mkdir -p /data/sicd /data/out
for s in panama melbourne iowa; do
  [ -f /data/sicd/$s.sicd ] || curl -s -o /data/sicd/$s.sicd "$U/${KEY[$s]}_SICD.nitf" &
  [ -f /data/out/$s/ref/bp_numba_fp64.npy ] || { mkdir -p /data/out/$s/ref; gcloud storage cp $B/v2/out/$s/ref/bp_numba_fp64.npy /data/out/$s/ref/ >/dev/null 2>&1; } &
  [ -f /data/$s.npz ] || gcloud storage cp $B/v2/data/$s.npz /data/ >/dev/null 2>&1 &
done
wait
for s in panama melbourne iowa; do
  $PY compare_sicd.py --sicd /data/sicd/$s.sicd --ref /data/out/$s/ref/bp_numba_fp64.npy --npz /data/$s.npz --out /data/sicd_$s.json \
      --crops "${CROPS[$s]}" --crops-out /data/sicd_${s}_crops.npz > /tmp/sicd_$s.log 2>&1
  gcloud storage cp /data/sicd_$s.json $B/v2/results/sicd4_$s.json >/dev/null 2>&1
  [ -f /data/sicd_${s}_crops.npz ] && gcloud storage cp /data/sicd_${s}_crops.npz $B/v2/results/sicd4_${s}_crops.npz >/dev/null 2>&1
  gcloud storage cp /tmp/sicd_$s.log $B/v2/logs/ >/dev/null 2>&1
done
echo DONE > /tmp/done_sicd4; gcloud storage cp /tmp/done_sicd4 $B/v2/logs/done_sicd4 >/dev/null 2>&1
