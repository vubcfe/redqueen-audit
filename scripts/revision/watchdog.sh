#!/usr/bin/env bash
# Revision R1 watchdog, run from cron every 10 minutes and at boot:
#   */10 * * * * ~/vubc/redqueen_r1/scripts/revision/watchdog.sh
#   @reboot sleep 180; ~/vubc/redqueen_r1/scripts/revision/watchdog.sh
# For each job group (search runs, MLP re-audit): if the group is incomplete
# and no worker is alive, relaunch it (finished records are skipped); if
# workers are alive but no log has moved for STALE_MIN minutes, kill and
# relaunch. At most MAX_RELAUNCH relaunches per group, then it only logs.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
exec 9>/tmp/redqueen_r1_watchdog.lock
flock -n 9 || exit 0
LOG=results/logs/watchdog.log
STALE_MIN=60
MAX_RELAUNCH=8
mkdir -p results/logs
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

count() { find $1 -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }  # $1: dirs (glob)

group() {  # name expected count_dirs proc_pattern log_glob launcher gpus wrapper_tag
  local name=$1 expected=$2 dir=$3 pat=$4 logs=$5 launcher=$6 gpus=$7 tag=$8
  local done; done=$(count "$dir")
  local n_proc; n_proc=$(pgrep -f -- "$pat" | wc -l)
  local cnt_file="results/logs/watchdog_${name}.relaunches"
  local relaunches; relaunches=$(cat "$cnt_file" 2>/dev/null || echo 0)
  if [ "$done" -ge "$expected" ]; then
    [ -f "results/logs/watchdog_${name}.complete" ] || { say "$name complete ($done/$expected)"; touch "results/logs/watchdog_${name}.complete"; }
    return
  fi
  if [ "$n_proc" -gt 0 ]; then
    local newest; newest=$(ls -t $logs 2>/dev/null | head -1)
    if [ -n "$newest" ] && [ -n "$(find "$newest" -mmin +$STALE_MIN)" ]; then
      say "$name stale: no log update for ${STALE_MIN} min ($done/$expected); killing workers"
      pkill -f -- "$tag"; pkill -f -- "$pat"; sleep 5
      pkill -9 -f -- "$tag"; pkill -9 -f -- "$pat"
      n_proc=0
    else
      return
    fi
  fi
  if [ "$relaunches" -ge "$MAX_RELAUNCH" ]; then
    say "$name incomplete ($done/$expected), no workers, relaunch cap reached; manual check needed"
    return
  fi
  echo $((relaunches + 1)) > "$cnt_file"
  for f in $logs; do [ -f "$f" ] && cp "$f" "$f.prev$relaunches"; done
  say "$name relaunch #$((relaunches + 1)) ($done/$expected done)"
  "$launcher" "$gpus" >> "$LOG" 2>&1
}

# search: fixedpar 75 + gen20 75 + lambda 3 x 25 = 225 records
group search 225 "results/revision/fixedpar_* results/revision/gen20_* results/revision/lam*" "redqueen suite --config configs/revision" \
  "results/logs/r1_worker*.log" scripts/revision/launch_r1.sh "0 0 1" R1_WORKER_DONE
# MLP re-audit: 375 stored quantum runs
group reaudit 375 "results/revision/reaudit/*" "scripts/revision/reaudit_mlp.py" \
  "results/logs/reaudit_worker*.log" scripts/revision/launch_reaudit.sh "1 1" REAUDIT_WORKER_DONE
