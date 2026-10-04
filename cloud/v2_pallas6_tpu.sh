#!/usr/bin/env bash
# Fresh-instance entry for the third-generation build: v2_pallas6_tpu.sh <label> ...
cfg="256 8 16"
bash ~/sar/cloud/v2_pallas3_tpu.sh $1 $cfg pallas6 r6
