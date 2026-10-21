#!/bin/bash
# Rule-set search: rules-only variants x 10 past 3-day windows (30 days), all CPU cores. Usage: bash scripts/windows10.sh
cd "$(dirname "$0")/.."
E0=1792605600; WINDOWS=$(for k in $(seq 0 9); do echo $((E0 - k*259200)); done)
echo "$WINDOWS" | xargs -P 10 -I{} sh -c 'PYTHONPATH=. python3 -m bot.replay --days 3 --rules --end {} --fetch-only --out /tmp/claude-1000/fetch_{} >/dev/null 2>&1'   # downloads in parallel
declare -A V=( [momentum]="" [momentum_strict]="--gate strict" [pullback]="--entry pullback" [pullback_strict]="--entry pullback --gate strict"
              [pullback_strict_2pos_dl1]="--entry pullback --gate strict --maxpos 2 --dlcap 0.01" [pullback_2pos_dl1]="--entry pullback --maxpos 2 --dlcap 0.01"
              [pullback_tp1.5_strict]="--entry pullback --gate strict --tp 1.5" )
JOBS=""; for v in "${!V[@]}"; do for e in $WINDOWS; do JOBS+="$v|$e|${V[$v]}"$'\n'; done; done
printf "%s" "$JOBS" | xargs -P "$(nproc)" -I{} sh -c 'IFS="|"; set -- $(echo "{}"); PYTHONPATH=. python3 -m bot.replay --days 3 --rules --end $2 $3 --out logs/w10/$1/$2 >/dev/null 2>&1'
python3 scripts/summarize.py logs/w10
