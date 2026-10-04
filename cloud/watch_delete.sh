#!/usr/bin/env bash
# Wait for an instance's done marker to appear in the bucket, then delete the instance.
#   watch_delete.sh <tpu|gpu> <name> <zone> <marker>
P=${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}; kind=$1; name=$2; zone=$3; marker=$4
until gcloud storage ls gs://${GCS_BUCKET:?set GCS_BUCKET to your bucket name}/results/v2/$marker >/dev/null 2>&1; do sleep 30; done
if [ "$kind" = tpu ]; then gcloud compute tpus tpu-vm delete $name --zone $zone --project $P --quiet 2>&1 | tail -1
else gcloud compute instances delete $name --zone $zone --project $P --quiet 2>&1 | tail -1; fi
echo "$name deleted $(date -u +%T) after $marker"
