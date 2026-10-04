#!/usr/bin/env bash
# Create a TPU VM, set it up, start a v2 sequence, and return.  launch_tpu.sh <name> <zone> <accel-type> <runtime> <label> <id> <stages> [minutes]
set -u
P=${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}; S=${SCRATCH:-/tmp/sar-accel}
name=$1; zone=$2; acc=$3; ver=$4; label=$5; stages=$6; mins=${7:-60}
cd "$(dirname "$0")/.."
gcloud compute tpus tpu-vm create $name --zone $zone --accelerator-type $acc --version $ver --project $P > $S/create_$name.log 2>&1 || { echo "$name create FAILED: $(grep -o '"message": "[^"]*' $S/create_$name.log | head -1 | cut -c1-160)"; exit 1; }
echo "$name created $(date -u +%T)"
for i in 1 2 3 4 5; do timeout 120 gcloud compute tpus tpu-vm scp --project $P --zone $zone --quiet $S/src.tgz cloud/accel_setup.sh cloud/run_accel_v2.sh cloud/v2_run_accel.sh cloud/v2_finish_tpu.sh cloud/v2_pallas_tpu.sh $name:~/ > $S/scp_$name.log 2>&1 && break; done
timeout 500 gcloud compute tpus tpu-vm ssh $name --project $P --zone $zone --quiet --command "mkdir -p ~/sar && tar xzf ~/src.tgz -C ~/sar; bash ~/accel_setup.sh tpu $mins 2>&1 | tail -1; ${ENVS:-} nohup bash ${RUNSCRIPT:-~/v2_run_accel.sh} $label tpu $stages > ~/runv2.log 2>&1 < /dev/null & disown; echo started-$name" 2>&1 | grep -E "jax|started"
