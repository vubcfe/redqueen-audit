#!/usr/bin/env bash
# Revision R1 search runs on the lab server, in the environment of the paper's
# real-dataset runs (py311env, torch 2.11, 4 BLAS threads), which reproduces
# the stored records bit for bit. Workers split every config by --shard and
# skip finished records, so re-running resumes.
#   usage: scripts/revision/launch_r1.sh "0 0 1"   (one worker per listed GPU)
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/logs
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=~/miniconda3/envs/py311env/bin/python
GPUS=(${1:-0 0 1})
NW=${#GPUS[@]}
CONFIGS=""
for e in fixedpar gen20 lam0.1 lam0.5 lam1; do
  for ds in eurosat so2sat ucmerced breastmnist cifar10; do
    CONFIGS="$CONFIGS configs/revision/${e}_${ds}.yaml"
  done
done
for i in $(seq 0 $((NW - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid nohup bash -c "
    for c in $CONFIGS; do
      $PY -W ignore -u -m redqueen suite --config \$c --skip-existing --shard $i/$NW
    done; echo R1_WORKER_DONE" > results/logs/r1_worker$i.log 2>&1 < /dev/null &
  echo "worker $i on gpu ${GPUS[$i]}: pid $!"
done
