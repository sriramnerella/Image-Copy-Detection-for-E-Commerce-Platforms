#!/usr/bin/env bash
set -euo pipefail
cd /home/saketh/Codebase
launch_one() {
  local ds="$1"
  local gpu="$2"
  local log="/home/saketh/json/final_named_artifacts_20260601/${ds}/phase3_cpp_centroid_branchbound/${ds}_allq_per_attack_20260603.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup /home/saketh/miniforge3/envs/codebase-gpu/bin/python phase3_extract_per_attack_allq.py --dataset "$ds" > "$log" 2>&1 &
  echo "${ds}:$!"
}
launch_one cifar 1
launch_one flickr 3
launch_one ucid 1
launch_one amazon 3
