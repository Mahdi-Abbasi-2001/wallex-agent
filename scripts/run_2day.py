"""Rules-only variants on 15 past 2-day windows (30 days), matching the real challenge length. python scripts/run_2day.py"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
E0 = 1792605600; WINDOWS = [E0 - k * 172800 for k in range(15)]
V = {"base_33pct_tp2": "", "size50_tp2": "--posfrac 0.5", "base_33pct_tp1.5": "--tp 1.5", "size50_tp1.5": "--posfrac 0.5 --tp 1.5", "base_33pct_tp1": "--tp 1.0"}
env = {**os.environ, "PYTHONPATH": "."}
def fetch(e): subprocess.run([sys.executable, "-m", "bot.replay", "--days", "2", "--rules", "--end", str(e), "--fetch-only", "--out", f"/tmp/claude-1000/f2_{e}"], env=env, capture_output=True)
with ThreadPoolExecutor(8) as ex: list(ex.map(fetch, WINDOWS))
def run(j):
    v, e = j
    r = subprocess.run([sys.executable, "-m", "bot.replay", "--days", "2", "--rules", "--end", str(e), *V[v].split(), "--out", f"logs/d2/{v}/{e}"], env=env, capture_output=True, text=True)
    if r.returncode: print("FAILED", v, e, r.stderr[-300:], flush=True)
with ThreadPoolExecutor(os.cpu_count()) as ex: list(ex.map(run, [(v, e) for v in V for e in WINDOWS]))
subprocess.run([sys.executable, "scripts/summarize.py", "logs/d2"])
