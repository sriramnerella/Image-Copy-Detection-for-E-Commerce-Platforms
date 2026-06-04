#!/usr/bin/env bash
set -u
pkill -u saketh -f phase3_centroid_branchbound_tuned.py || true
pkill -u saketh -f phase3_centroid_branchbound_autotune.py || true
sleep 1
cd /home/saketh/Codebase
for item in cifar:1 flickr:2 ucid:3 amazon:2; do
  ds="${item%%:*}"
  gpu="${item##*:}"
  log="/home/saketh/json/final_named_artifacts_20260601/${ds}/phase3_centroid_branchbound_autotune/${ds}_centroid_branchbound_autotune_20260603.log"
  mkdir -p "$(dirname "$log")"
  echo "[AutoBBLaunch] $(date) dataset=${ds} gpu=${gpu}" >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" /home/saketh/miniforge3/envs/codebase-gpu/bin/python \
    /home/saketh/Codebase/phase3_centroid_branchbound_autotune.py --dataset "$ds" >> "$log" 2>&1 &
  echo "${ds}:$!"
done
