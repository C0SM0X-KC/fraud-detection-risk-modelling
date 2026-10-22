#!/usr/bin/env bash
# Watch the running frds container: a status line per interval, STALL lines, and an exit line.
#
#   scripts/watch_job.sh [interval_s=120] [stall_after_s=600]

interval=${1:-120}
stall_after=${2:-600}
name=$(docker ps --filter "name=frds-run" --format '{{.Names}}' | head -1)
if [ -z "$name" ]; then
  echo "no frds job running"
  exit 0
fi

last_size=-1
last_change=$(date +%s)
started=$(date +%s)
while docker ps --format '{{.Names}}' | grep -qx "$name"; do
  size=$(docker logs "$name" 2>&1 | wc -c)
  now=$(date +%s)
  if [ "$size" != "$last_size" ]; then
    last_size=$size
    last_change=$now
  fi
  cpu=$(docker stats --no-stream --format '{{.CPUPerc}}' "$name" 2>/dev/null | tr -d '%')
  cpu=${cpu:-0}
  # The last pipeline log line, not library warnings that repeat between them.
  tail_line=$(docker logs --tail 200 "$name" 2>&1 | tr -d '\r' \
    | grep -E "\| (INFO|WARNING|ERROR) |Traceback|Error|passed|failed" | tail -1 | cut -c1-110)
  idle=$(( now - last_change ))
  elapsed=$(( (now - started) / 60 ))
  status="ok"
  if [ "$idle" -ge "$stall_after" ]; then
    status="STALL: log unchanged for $((idle / 60)) min"
  elif awk "BEGIN{exit !($cpu < 20)}"; then
    status="STALL?: CPU only ${cpu}%"
  fi
  echo "[$(date +%H:%M)] ${elapsed}m cpu ${cpu}% $status | ${tail_line}"
  sleep "$interval"
done
echo "[$(date +%H:%M)] $name exited"
