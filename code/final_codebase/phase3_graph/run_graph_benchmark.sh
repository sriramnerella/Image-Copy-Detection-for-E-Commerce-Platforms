#!/usr/bin/env bash
set -u
pkill -u saketh -f phase3_cpp_centroid_bb_benchmark.py || true
pkill -u saketh -f phase3_centroid_branchbound_autotune.py || true
sleep 1
cd /home/saketh/Codebase
g++ -O3 -march=native -shared -std=c++17 -fPIC /home/saketh/Codebase/centroid_bb_kernel.cpp -o /home/saketh/Codebase/libcentroid_bb_kernel.so
for item in cifar:1 flickr:2 ucid:3 amazon:2; do
  ds="${item%%:*}"
  gpu="${item##*:}"
  log="/home/saketh/json/final_named_artifacts_20260601/${ds}/phase3_cpp_centroid_branchbound/${ds}_cpp_centroid_bb_20260603.log"
  mkdir -p "$(dirname "$log")"
  echo "[CppBBLaunch] $(date) dataset=${ds} gpu=${gpu}" >> "$log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" /home/saketh/miniforge3/envs/codebase-gpu/bin/python \
    /home/saketh/Codebase/phase3_cpp_centroid_bb_benchmark.py --dataset "$ds" >> "$log" 2>&1 &
  echo "${ds}:$!"
done
