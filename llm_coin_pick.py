"""Can an LLM pick coins? Anonymised 30-day stats for the 10 majors -> score each coin for the next 7 days. Evaluated on windows after 2025-07.
 python llm_coin_pick.py MODEL [max_windows]        cache: data/llm_pick_<model>.json"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import strat_search as S
from agent.llm import load_env
from contest.market import Market

load_env()
from openai import OpenAI

model = sys.argv[1]; maxw = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
mkt = Market(S.load_tf("1d")[1], 86400); WARM = 260; SPLIT = pd.Timestamp("2025-07-01").timestamp()
starts = [s for s in range(WARM, len(mkt) - 8, 2) if mkt.t[s] >= SPLIT][:maxw]; N = len(mkt.symbols); C, O, V = mkt.C, mkt.O, mkt.V
SYSTEM = ("You rank anonymised crypto assets by expected relative performance over the NEXT 7 DAYS, using only the data given. Coin labels are shuffled and meaningless. "
          'Reply with JSON only: {"scores": {"C1": <integer -100..100>, ...}} for every coin; higher = you expect it to outperform the others over the next 7 days.')


def prompt_for(s, rng):
    perm = rng.permutation(N); lab = {j: f"C{k+1}" for k, j in enumerate(perm)}; i = s - 1; lines = []
    ew = np.exp(np.cumsum(np.nanmean(np.vstack([np.zeros((1, N)), C[1:] / C[:-1] - 1])[i - 59:i + 1], axis=1)))    # equal-weight index over 60 days
    mk = f"Market (equal-weight index): 7d {((ew[-1]/ew[-8]-1)*100):+.1f}%, 30d {((ew[-1]/ew[-31]-1)*100):+.1f}%, {'above' if ew[-1] > ew[-50:].mean() else 'below'} its 50-day average"
    for j in sorted(perm, key=lambda x: int(lab[x][1:])):
        c = C[i - 30:i + 1, j]; rel = (c / c[-1] - 1) * 100; pts = ", ".join(f"{x:.0f}" for x in rel[::3]); r = lambda n: (C[i, j] / C[i - n, j] - 1) * 100
        vol = np.std(np.diff(C[i - 14:i + 1, j]) / C[i - 14:i, j]) * 100; hi = (C[i, j] / C[i - 29:i + 1, j].max() - 1) * 100; vr = V[i - 6:i + 1, j].mean() / (V[i - 29:i + 1, j].mean() + 1e-12)
        lines.append(f"{lab[j]}: closes vs latest (% every 3 days, oldest first): [{pts}] | returns 1d {r(1):+.1f}% 3d {r(3):+.1f}% 7d {r(7):+.1f}% 14d {r(14):+.1f}% 30d {r(30):+.1f}% | 14d volatility {vol:.1f}%/day | {hi:.0f}% from 30d high | 7d volume vs 30d avg x{vr:.2f}")
    return mk + "\n" + "\n".join(lines), lab


def ask(client, p):
    extra = dict(max_completion_tokens=2500) if not model.startswith("gpt-4") else dict(temperature=0, max_tokens=300)
    for a in range(4):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}], response_format={"type": "json_object"}, **extra)
            return json.loads(r.choices[0].message.content)["scores"], r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception as e:
            m = str(e)
            if "temperature" in m and "temperature" in extra: extra.pop("temperature"); continue
            if "max_tokens" in m and "max_tokens" in extra: extra["max_completion_tokens"] = extra.pop("max_tokens"); continue
            time.sleep(2 ** a)
    return None, 0, 0


cache_f = Path("data") / f"llm_pick_{model}.json"; cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
rng = np.random.default_rng(3); jobs = []
for s in starts: p, lab = prompt_for(s, rng); jobs.append((s, p, lab))
todo = [(s, p) for s, p, _ in jobs if str(s) not in cache]; client = OpenAI(); pt = ct = 0; t0 = time.time()
if todo:
    with ThreadPoolExecutor(8) as ex:
        for n, ((s, _), (sc, a, b)) in enumerate(zip(todo, ex.map(lambda x: ask(client, x[1]), todo)), 1):
            if sc is not None: cache[str(s)] = sc
            pt += a; ct += b
            if n % 25 == 0: cache_f.write_text(json.dumps(cache)); print(f"  {n}/{len(todo)} | {time.time()-t0:.0f}s", flush=True)
    cache_f.write_text(json.dumps(cache))
print(f"{model}: {len(jobs)} windows ({len(todo)} new calls), tokens in {pt} out {ct}")
# ---- evaluation ----
rows = []
for s, _, lab in jobs:
    sc = cache.get(str(s))
    if not sc: continue
    z = np.array([float(sc.get(lab[j], np.nan)) for j in range(N)]); ret = C[s + 6] / O[s] - 1
    if np.isnan(z).any(): continue
    mom14 = C[s - 1] / C[s - 15] - 1; mom30 = C[s - 1] / C[s - 31] - 1; mom7 = C[s - 1] / C[s - 8] - 1
    ic = lambda a: spearmanr(a, ret)[0]
    rows.append(dict(s=s, t=mkt.t[s], ic_llm=ic(z), ic_m7=ic(mom7), ic_m14=ic(mom14), ic_m30=ic(mom30), top=ret[np.argmax(z)], bot=ret[np.argmin(z)], avg=ret.mean(), mtop=ret[np.argmax(mom14)], mbot=ret[np.argmin(mom14)], best=ret.max()))
D = pd.DataFrame(rows); half = pd.Timestamp("2026-01-01").timestamp(); print(f"\nevaluated {len(D)} windows | {pd.to_datetime(D.t.min(), unit='s').date()} -> {pd.to_datetime(D.t.max(), unit='s').date()}")
def se(x): return x.std() / np.sqrt(len(x) / 3.5)      # overlapping windows (step 2d, 7d holds): ~3.5x fewer independent observations
for lab, d in (("ALL test windows", D), ("2025-07 to 2025-12", D[D.t < half]), ("2026 only (most likely after model cutoff)", D[D.t >= half])):
    print(f"\n[{lab}] n={len(d)}")
    print(f"  mean rank correlation (IC) with next-7d returns:  LLM {d.ic_llm.mean():+.3f} (se {se(d.ic_llm):.3f}) | momentum 7d {d.ic_m7.mean():+.3f} | 14d {d.ic_m14.mean():+.3f} | 30d {d.ic_m30.mean():+.3f}")
    print(f"  LLM top pick avg 7d return {d.top.mean()*100:+.2f}% vs average coin {d.avg.mean()*100:+.2f}% (excess {(d.top-d.avg).mean()*100:+.2f}%, se {se(d.top-d.avg)*100:.2f}) | LLM bottom pick {d.bot.mean()*100:+.2f}%")
    print(f"  momentum-14d top pick {d.mtop.mean()*100:+.2f}% | bottom {d.mbot.mean()*100:+.2f}% | hindsight best coin {d.best.mean()*100:+.1f}%")
