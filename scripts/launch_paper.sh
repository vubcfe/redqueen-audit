#!/usr/bin/env bash
# Launch the EuroSAT study and qml-benchmarks on a 2-GPU machine, detached from the shell.
#   EuroSAT: 4 shards, two per GPU, env py311env (torch 2.11).
#   qml-benchmarks: 2 shards, one per GPU, env redqueen (the paper's pinned env).
# Re-running resumes: finished records are skipped (--skip-existing).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY_EURO=${PY_EURO:-python}     # main-study environment (requirements.txt)
PY_BENCH=${PY_BENCH:-python}   # qml-benchmarks environment (requirements-qmlbench.txt)

start() {  # gpu log cmd...
  local gpu=$1 log=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$@" > "results/logs/$log" 2>&1 < /dev/null &
  echo "$!" >> results/logs/pids.txt
  echo "started pid $! on gpu $gpu -> results/logs/$log"
}

: > results/logs/pids.txt
for i in 0 1 2 3; do
  start $((i / 2)) "paper_shard$i.log" "$PY_EURO" -W ignore -u -m redqueen suite \
    --config configs/paper.yaml --skip-existing --shard "$i/4"
done
for i in 0 1; do
  start "$i" "qmlbench_shard$i.log" "$PY_BENCH" -W ignore -u -m redqueen bench \
    --config configs/qmlbench_paper.yaml --skip-existing --shard "$i/2"
done
