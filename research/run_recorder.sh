#!/bin/bash
# Keeps the websocket recorder alive (restarts it after any crash). Stop with: kill $(cat data/ws/supervisor.pid)
cd "$(dirname "$0")/.."
echo $$ > data/ws/supervisor.pid
while true; do
  python3 -u research/record_ws.py >> data/ws/stdout.log 2>&1
  echo "$(date -u +%FT%T) recorder exited, restarting in 5s" >> data/ws/recorder.log
  sleep 5
done
