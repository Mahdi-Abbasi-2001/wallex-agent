"""Run rules-only variants over the 10 cached 3-day windows on all cores, then summarise. python scripts/run_variants.py"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
E0 = 1792605600; WINDOWS = [E0 - k * 259200 for k in range(10)]
V = {"momentum": "--entry momentum --gate normal --maxpos 3 --dlcap 0.02", "momentum_strict": "--entry momentum --gate strict --maxpos 3 --dlcap 0.02", "pullback": "--entry pullback --gate normal --maxpos 3 --dlcap 0.02", "pullback_strict": "--entry pullback --gate strict --maxpos 3 --dlcap 0.02",
     "pullback_strict_2pos_dl1": "--entry pullback --gate strict --maxpos 2 --dlcap 0.01", "pullback_2pos_dl1": "--entry pullback --gate normal --maxpos 2 --dlcap 0.01",
     "pullback_tp1.5_strict": "--entry pullback --gate strict --maxpos 3 --dlcap 0.02 --tp 1.5"}
only = sys.argv[1:] or list(V)
jobs = [(v, e) for v in only for e in WINDOWS if not os.path.exists(f"logs/w10/{v}/{e}/summary.json")]
def run(j):
    v, e = j
    r = subprocess.run([sys.executable, "-m", "bot.replay", "--days", "3", "--rules", "--end", str(e), *V[v].split(), "--out", f"logs/w10/{v}/{e}"],
                       env={**os.environ, "PYTHONPATH": "."}, capture_output=True, text=True)
    if r.returncode: print("FAILED", v, e, r.stderr[-300:], flush=True)
with ThreadPoolExecutor(os.cpu_count()) as ex: list(ex.map(run, jobs))
subprocess.run([sys.executable, "scripts/summarize.py", "logs/w10"])
