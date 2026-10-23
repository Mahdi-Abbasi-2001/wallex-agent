#!/bin/bash
# Recorder status. Stop it with: kill $(cat data/ws/recorder.pid)
cd "$(dirname "$0")/.."
if kill -0 "$(cat data/ws/recorder.pid 2>/dev/null)" 2>/dev/null; then echo "recorder RUNNING (pid $(cat data/ws/recorder.pid))"; else echo "recorder NOT running -> nohup setsid python3 -u research/record_ws.py > data/ws/stdout.log 2>&1 &"; fi
du -sh data/ws | cut -f1 | xargs echo "disk used:"
for f in data/ws/trades_*.jsonl* data/ws/book_*.jsonl*; do [ -e "$f" ] && echo "$f  $(du -h "$f" | cut -f1)"; done
tail -n 3 data/ws/recorder.log
