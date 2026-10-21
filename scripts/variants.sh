#!/bin/bash
# Rules-only variants x 4 past 3-day windows (cached data). Usage: bash scripts/variants.sh
cd "$(dirname "$0")/.."
WINDOWS="1791828000 1792087200 1792346400 1792605600"
declare -A V=( [base]="" [tp1]="--tp 1.0" [tp1.5]="--tp 1.5" [trail]="--trail --tp 3.0" [tp1.5_trail]="--tp 1.5 --trail"
              [pullback]="--entry pullback" [pullback_tp1]="--entry pullback --tp 1.0" [lead_1m]="--entry lead --tp 1.0 --decide 60" )
for v in "${!V[@]}"; do for e in $WINDOWS; do
  ( PYTHONPATH=. python3 -m bot.replay --days 3 --rules --end $e ${V[$v]} --out logs/var/$v/$e > /dev/null 2>&1 ) &
done; done; wait
python3 - <<'PY'
import json,glob,collections
R=collections.defaultdict(dict)
for f in glob.glob("logs/var/*/*/summary.json"):
    v,e=f.split("/")[2:4]; R[v][e]=json.load(open(f))
print(f"{'variant':14s} " + "  ".join(f"{e[-6:]:>16s}" for e in sorted(next(iter(R.values())))) + "   |   avg ret  worst dd  trades  win%")
for v,d in sorted(R.items(), key=lambda kv: -sum(x['return_pct'] for x in kv[1].values())):
    rs=[d[e] for e in sorted(d)]; tr=sum(x["trades"] for x in rs); w=sum(x["wins"] for x in rs)
    print(f"{v:14s} " + "  ".join(f"{x['return_pct']:+6.2f}% dd{x['max_drawdown_pct']:4.1f} n{x['trades']:<2d}" for x in rs) +
          f"   |  {sum(x['return_pct'] for x in rs)/len(rs):+6.2f}%   {max(x['max_drawdown_pct'] for x in rs):5.2f}%   {tr:4d}  {w/max(tr,1)*100:3.0f}%")
PY
