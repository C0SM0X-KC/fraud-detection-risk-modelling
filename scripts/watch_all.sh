#!/usr/bin/env bash
# Supervisor: a status line per interval for every frds container, the Kaggle job and
# image builds, with STALL lines when something stops progressing.
#
#   scripts/watch_all.sh [interval_s=120] [log_stall_s=600] [kaggle_run_limit_s=3600]

interval=${1:-120}
log_stall=${2:-600}
kaggle_limit=${3:-3600}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1
set -a; [ -f .env ] && . ./.env; set +a
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

declare -A last_size last_change
kaggle_state="" kaggle_since=$(date +%s)

while true; do
  now=$(date +%s)
  parts=()

  # -- local containers ------------------------------------------------------
  for name in $(docker ps --filter "name=frds-run" --format '{{.Names}}'); do
    # The overnight runner sends each step's output to reports/overnight.log,
    # not to the container's stdout, so that is where progress shows.
    if [[ "$name" == *overnight* ]] && [ -f reports/overnight.log ]; then
      size=$(wc -c < reports/overnight.log)
    else
      size=$(docker logs "$name" 2>&1 | wc -c)
    fi
    if [ "${last_size[$name]}" != "$size" ]; then
      last_size[$name]=$size; last_change[$name]=$now
    fi
    idle=$(( now - ${last_change[$name]:-$now} ))
    cpu=$(docker stats --no-stream --format '{{.CPUPerc}}' "$name" 2>/dev/null | tr -d '%')
    line=$(docker logs --tail 300 "$name" 2>&1 | tr -d '\r' \
      | grep -E "\| (INFO|WARNING|ERROR) |Traceback|Error|passed|failed" | tail -1 \
      | sed -E 's/^[0-9:]+ \| [A-Z]+ +\| ([a-z_.]+) +\| /\1: /' | cut -c1-95)
    flag=""
    [ "$idle" -ge "$log_stall" ] && flag="STALL(log idle $((idle/60))m) "
    awk "BEGIN{exit !(${cpu:-0} < 15)}" && [ "$idle" -ge 180 ] && flag="${flag}STALL?(cpu ${cpu}%) "
    parts+=("local ${flag}cpu ${cpu:-?}%: ${line}")
  done
  [ ${#parts[@]} -eq 0 ] && parts+=("local: idle")

  # -- Kaggle ---------------------------------------------------------------
  ks=$(.venv/Scripts/kaggle kernels status venuslithium/frds-gpu-tiers 2>&1 | tail -1 \
       | sed -E 's/.*KernelWorkerStatus\.([A-Z_]+).*/\1/')
  if [ "$ks" != "$kaggle_state" ]; then kaggle_state=$ks; kaggle_since=$now; fi
  age=$(( (now - kaggle_since) / 60 ))
  kflag=""
  [ "$ks" = "QUEUED" ] && [ "$age" -ge 20 ] && kflag="STALL(queued ${age}m) "
  [ "$ks" = "RUNNING" ] && [ $((now - kaggle_since)) -ge "$kaggle_limit" ] && kflag="STALL(running ${age}m) "
  parts+=("kaggle ${kflag}${ks:-?} ${age}m")

  # -- image builds -----------------------------------------------------------
  if docker buildx history ls 2>/dev/null | awk 'NR>1' | grep -q "Running"; then
    parts+=("build: running")
  fi

  line="$(IFS=' | '; echo "${parts[*]}")"
  # Emit a line only when the set of jobs or Kaggle's state changes, or as a heartbeat.
  key="$(docker ps --filter name=frds-run --format '{{.Names}}' | sort | tr '\n' ' ') $ks"
  checks=$(( ${checks:-0} + 1 ))
  if [ "$key" != "${last_key:-}" ] || [[ "$line" == *STALL* ]] || [ $(( checks % ${HEARTBEAT:-15} )) -eq 0 ]; then
    printf '[%s] %s\n' "$(date +%H:%M)" "$line"
    last_key=$key
  fi
  sleep "$interval"
done
