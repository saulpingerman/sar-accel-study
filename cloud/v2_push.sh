#!/usr/bin/env bash
# Copy the working tree to an instance and start a command there in the background.
#   cloud/v2_push.sh <instance> <zone> <log> '<remote command>'
P=${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}; inst=$1; zone=$2; log=$3; cmd=$4
cd "$(dirname "$0")/.."
tar czf /tmp/sar-src.tgz --exclude=.git --exclude='*.npz' --exclude=results --exclude=__pycache__ --exclude='*.html' --exclude=report .
gcloud compute scp --project $P --zone "$zone" --quiet /tmp/sar-src.tgz "$inst":~/sar-src.tgz >/dev/null 2>&1
printf '%s\n' "$cmd" > /tmp/v2_remote_cmd.sh
gcloud compute scp --project $P --zone "$zone" --quiet /tmp/v2_remote_cmd.sh "$inst":~/v2_remote_cmd.sh >/dev/null 2>&1
timeout 60 gcloud compute ssh "$inst" --project $P --zone "$zone" --quiet --command "mkdir -p ~/sar && tar xzf ~/sar-src.tgz -C ~/sar && cd ~/sar && (setsid nohup bash ~/v2_remote_cmd.sh > $log 2>&1 < /dev/null &); sleep 1; echo started" < /dev/null 2>&1 | tail -1
