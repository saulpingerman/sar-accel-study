#!/usr/bin/env bash
# Copy the working tree to an instance and optionally run a command there.
#   cloud/push.sh <instance> <zone> ['remote command']
set -euo pipefail
P=${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}
inst=$1; zone=$2; cmd=${3:-}
cd "$(dirname "$0")/.."
tar czf /tmp/sar-src.tgz --exclude=.git --exclude='*.npz' --exclude=results --exclude=__pycache__ .
gcloud compute scp --project $P --zone "$zone" --quiet /tmp/sar-src.tgz "$inst":~/sar-src.tgz >/dev/null
gcloud compute ssh "$inst" --project $P --zone "$zone" --quiet --command \
  "mkdir -p ~/sar && tar xzf ~/sar-src.tgz -C ~/sar && cd ~/sar && mkdir -p results && { $cmd ; }"
