#!/usr/bin/env bash
# Re-prepare the three collections on the vendor's slant-plane grids and refresh the bucket copies.
set -u
cd ~/sar; PY=~/venv/bin/python; B=gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}
U="https://umbra-open-data-catalog.s3.amazonaws.com/sar-data/tasks"
[ -f /data/iowa.sicd ] || curl -s -o /data/iowa.sicd "$U/Johnston%2C%20Iowa%20Farm/05fd0504-7fec-4e88-97f8-e55fa3d68b09/2023-10-20-15-53-21_UMBRA-04/2023-10-20-15-53-21_UMBRA-04_SICD.nitf"
for s in ${SCENES:-panama melbourne iowa}; do
  $PY v2_prep.py --cphd /data/$s.cphd --sicd /data/$s.sicd --out /data/$s.npz --name $s > /data/$s.info 2>&1 || { echo "$s FAILED"; tail -3 /data/$s.info; continue; }
  gcloud storage cp /data/$s.npz /data/$s.info $B/v2/data/ >/dev/null 2>&1
  echo "$s prepared"
done
[ -n "${SCENES:-}" ] || gcloud storage rm -r $B/v2/out $B/v2/t >/dev/null 2>&1
echo R > /tmp/v2_reprep_done
