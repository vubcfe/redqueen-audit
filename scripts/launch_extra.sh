#!/usr/bin/env bash
# Main study on the four additional datasets (docs/DESIGN.md section 10).
# Workers (GPUS, default two per GPU on two GPUs) each walk all four configs
# with --shard i/n. Re-running resumes, but
# refuses to start while workers of an earlier launch are still alive.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PY:-python}   # interpreter of the main-study environment (requirements.txt)
CONFIGS=${CONFIGS:-"configs/paper_so2sat.yaml configs/paper_ucmerced.yaml configs/paper_breastmnist.yaml configs/paper_cifar10.yaml"}
for c in $CONFIGS; do   # every config's features must exist before workers start
  f=$(grep '^features:' "$c" | awk '{print $2}')
  [ -f "$f" ] || { echo "missing features $f for $c" >&2; exit 1; }
done

if [ -f results/logs/pids_extra.txt ]; then
  for p in $(cat results/logs/pids_extra.txt); do
    if grep -qs "redqueen" "/proc/$p/cmdline"; then
      echo "worker pid $p from an earlier launch is still running; not starting" >&2; exit 1
    fi
  done
fi

# GPUS: one entry per worker (default two workers per GPU on both GPUs)
read -r -a GPU_OF <<< "${GPUS:-0 0 1 1}"
NW=${#GPU_OF[@]}
: > results/logs/pids_extra.txt
for i in $(seq 0 $((NW - 1))); do
  # a suite that dies without a Python traceback (segfault) is logged as
  # SUITE_FAILED so the watcher sees it; the worker then moves on
  CUDA_VISIBLE_DEVICES=${GPU_OF[$i]} setsid nohup bash -c "
    for c in $CONFIGS; do
      $PY -W ignore -u -m redqueen suite --config \$c --skip-existing --shard $i/$NW \
        || echo \"SUITE_FAILED \$c rc=\$?\"
    done; echo WORKER_DONE" >> "results/logs/extra_worker$i.log" 2>&1 < /dev/null &
  echo "$!" >> results/logs/pids_extra.txt
  echo "started worker $i/$NW pid $! on gpu ${GPU_OF[$i]}"
done
