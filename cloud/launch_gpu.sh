#!/usr/bin/env bash
# Create an L4 instance in the first zone with capacity, set it up, start a v2 sequence.
#   launch_gpu.sh <name> <machine-type> <label> <id> <stages> [minutes]
set -u
P=${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}; S=${SCRATCH:-/tmp/sar-accel}; mkdir -p $S
name=$1; mt=$2; label=$3; id=$4; stages=$5; mins=${6:-80}
cd "$(dirname "$0")/.."
Z=""
for z in us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c us-east5-a us-south1-a; do
  if gcloud compute instances create $name --project $P --zone $z --machine-type $mt --accelerator type=nvidia-l4,count=1 --maintenance-policy TERMINATE \
      --image-family common-cu129-ubuntu-2404-nvidia-580 --image-project deeplearning-platform-release --boot-disk-size 100GB --boot-disk-type pd-balanced \
      --max-run-duration ${MAXRUN:-100m} --instance-termination-action DELETE --scopes cloud-platform --labels study=sar > $S/create_$name.log 2>&1; then Z=$z; break; fi
done
[ -n "$Z" ] || { echo "$name: no zone had capacity"; exit 1; }
echo $Z > $S/zone_$name; echo "$name created in $Z $(date -u +%T)"
for i in $(seq 1 30); do gcloud compute ssh $name --project $P --zone $Z --quiet --command 'true' -- -o ConnectTimeout=15 >/dev/null 2>&1 && break; done
gcloud compute scp --project $P --zone $Z --quiet cloud/accel_setup.sh cloud/run_accel_v2.sh $name:~/ > /dev/null 2>&1
gcloud compute ssh $name --project $P --zone $Z --quiet --command "bash ~/accel_setup.sh gpu $mins 2>&1 | tail -1; nohup bash ~/run_accel_v2.sh $label gpu $id $stages > ~/runv2.log 2>&1 < /dev/null & disown; echo started-$name" 2>&1 | grep -E "jax|started"
