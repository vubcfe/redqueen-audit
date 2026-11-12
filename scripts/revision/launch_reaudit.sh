#!/usr/bin/env bash
# Revision R1: MLP-auditor re-audit of all 375 stored quantum runs, in the
# paper's environment. usage: scripts/revision/launch_reaudit.sh "1 1"
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/logs
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=~/miniconda3/envs/py311env/bin/python
GPUS=(${1:-1 1})
NW=${#GPUS[@]}
for i in $(seq 0 $((NW - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid nohup bash -c "
    for ds in ucmerced breastmnist eurosat so2sat cifar10; do
      $PY -W ignore -u scripts/revision/reaudit_mlp.py \$ds --shard $i/$NW
    done; echo REAUDIT_WORKER_DONE" > results/logs/reaudit_worker$i.log 2>&1 < /dev/null &
  echo "reaudit worker $i on gpu ${GPUS[$i]}: pid $!"
done
