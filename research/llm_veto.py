"""Does an LLM score Donchian breakout entries better than chance? (4h, long-only, 8 USDT markets, ~3y)
 python llm_veto.py MODEL [N_SAMPLE|all]   results cached in data/llm_cache_<model>.json"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from agent.data import load_candles
from agent.llm import load_env
from agent.wallex_client import Wallex

load_env()
from openai import OpenAI

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT"]
N, M, COST, LOOK = 60, 30, 0.0018, 48
SYSTEM = ("You evaluate candidate LONG entries from a mechanical breakout system on an anonymized 4-hour crypto chart. "
          "The system buys when price closes above the highest high of the previous 60 bars and exits when price closes below the lowest low of the previous 30 bars. "
          "Each round trip costs 0.36% in fees and slippage. Prices are shown as percent difference from the current close (the entry price); volume is relative to its 48-bar average. "
          'Estimate how likely THIS trade is to end profitable after costs. Reply with JSON only: {"score": <integer 0-100>}, higher = more likely profitable.')


def candidates():
    w = Wallex(); rows = []
    for s in SYMS:
        d = load_candles(w, s, "240", 1095)
        o, h, l, c, v, t = (d[k].to_numpy() for k in ("o", "h", "l", "c", "v", "t"))
        hi = pd.Series(h).rolling(N).max().shift(1).to_numpy(); lo = pd.Series(l).rolling(M).min().shift(1).to_numpy()
        sma200 = pd.Series(c).rolling(200).mean().to_numpy(); atr = pd.Series(np.maximum(h - l, 0)).rolling(14).mean().to_numpy()
        pos, i0 = 0, 0
        for i in range(len(c)):
            if pos == 0 and c[i] > hi[i] and i >= 250:
                pos, i0 = 1, i
                # --- features available at entry bar close only ---
                sl = slice(i - LOOK + 1, i + 1); ref = c[i]; vr = v[sl] / (v[sl].mean() + 1e-12)
                bars = "\n".join(f"{(o[k]/ref-1)*100:6.2f} {(h[k]/ref-1)*100:6.2f} {(l[k]/ref-1)*100:6.2f} {(c[k]/ref-1)*100:6.2f} {vr[j]:5.2f}" for j, k in enumerate(range(sl.start, sl.stop)))
                summ = (f"breakout size above prior 60-bar high: {(c[i]/hi[i]-1)*100:.2f}%\nATR(14): {atr[i]/ref*100:.2f}% of price\n"
                        f"distance to 200-bar average: {(c[i]/sma200[i]-1)*100:.1f}%\nreturn over last 6/24/48 bars: {(c[i]/c[i-6]-1)*100:.1f}% / {(c[i]/c[i-24]-1)*100:.1f}% / {(c[i]/c[i-48]-1)*100:.1f}%\n"
                        f"initial stop distance (30-bar low): {(c[i]/lo[i]-1)*100:.1f}% below entry")
                prompt = f"{summ}\n\nLast {LOOK} bars, oldest first (open high low close vol_rel), % vs entry close:\n{bars}"
                rows.append(dict(sym=s, i=i, t=int(t[i]), prompt=prompt, above200=bool(c[i] > sma200[i]), entry=ref, exit_i=None))
            elif pos == 1 and c[i] < lo[i]:
                pos = 0; rows[-1]["exit_i"] = i; rows[-1]["net"] = (c[i] / rows[-1]["entry"]) * (1 - COST) / (1 + COST) - 1
        if pos == 1:
            rows[-1]["net"] = (c[-1] / rows[-1]["entry"]) * (1 - COST) / (1 + COST) - 1
    return pd.DataFrame(rows)


def ask(client, model, prompt, extra):
    for attempt in range(4):
        try:
            kw = dict(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], response_format={"type": "json_object"}, **extra)
            r = client.chat.completions.create(**kw)
            sc = float(json.loads(r.choices[0].message.content)["score"])
            return sc, r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception as e:
            msg = str(e)
            if "temperature" in msg and "temperature" in extra: extra.pop("temperature"); continue
            if "max_tokens" in msg and "max_tokens" in extra: extra["max_completion_tokens"] = extra.pop("max_tokens"); continue
            time.sleep(2 ** attempt)
    return None, 0, 0


def score_all(df, model, n=None, seed=0):
    cache_f = Path("data") / f"llm_cache_{model}.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    sub = df if n is None else df.sample(n=min(n, len(df)), random_state=seed)
    client = OpenAI(); extra = dict(max_completion_tokens=2000) if not model.startswith("gpt-4") else dict(temperature=0, max_tokens=30)
    todo = [(f"{r.sym}:{r.i}", r.prompt) for r in sub.itertuples() if f"{r.sym}:{r.i}" not in cache]
    ptok = ctok = 0; t0 = time.time()
    def work(a):
        return a[0], ask(client, model, a[1], dict(extra))
    def save():  # atomic write so an interrupted run never leaves a corrupt cache
        tmp = cache_f.with_suffix(".tmp"); tmp.write_text(json.dumps(cache)); tmp.replace(cache_f)
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for k, (sc, p, c) in ex.map(work, todo):
            if sc is not None: cache[k] = sc
            ptok += p; ctok += c; done += 1
            if done % 25 == 0 or done == len(todo):
                save(); el = time.time() - t0; eta = el / done * (len(todo) - done)
                print(f"  progress {done}/{len(todo)} ({done/len(todo)*100:.0f}%) | {el:.0f}s elapsed, ~{eta:.0f}s left | tokens in={ptok} out={ctok}", flush=True)
    save()
    sub = sub.assign(score=[cache.get(f"{r.sym}:{r.i}") for r in sub.itertuples()]).dropna(subset=["score", "net"])
    print(f"{model}: scored {len(sub)} trades ({len(todo)} new calls) in {time.time()-t0:.0f}s | tokens in={ptok} out={ctok}")
    return sub


def evaluate(S, label):
    from scipy.stats import spearmanr
    print(f"\n--- {label}: {len(S)} trades ---")
    print(f"all trades:            mean net {S.net.mean()*100:6.2f}%  win {(S.net>0).mean()*100:3.0f}%")
    print(f"score spread: min {S.score.min():.0f} median {S.score.median():.0f} max {S.score.max():.0f} | distinct values {S.score.nunique()}")
    rho, p = spearmanr(S.score, S.net); print(f"Spearman(score, net return) = {rho:+.3f} (p={p:.3f})")
    q = pd.qcut(S.score.rank(method="first"), 3, labels=["low", "mid", "high"])
    print("by score tercile (mean net %, win %):", {k: (round(g.net.mean() * 100, 2), round((g.net > 0).mean() * 100)) for k, g in S.groupby(q, observed=True)})
    top = S[S.score >= S.score.median()]
    rng = np.random.default_rng(0); sims = [S.net.sample(len(top), random_state=int(x)).mean() for x in rng.integers(0, 10**6, 3000)]
    print(f"LLM keeps top half:    n={len(top)} mean net {top.net.mean()*100:6.2f}%  win {(top.net>0).mean()*100:3.0f}%  | vs random same-size subsets: beats {np.mean(top.net.mean() > np.array(sims))*100:.0f}% of them")
    f = S[S.above200]; print(f"simple trend filter (>SMA200): n={len(f)} mean net {f.net.mean()*100:6.2f}%  win {(f.net>0).mean()*100:3.0f}%")


if __name__ == "__main__":
    model = sys.argv[1]; n = None if len(sys.argv) < 3 or sys.argv[2] == "all" else int(sys.argv[2])
    df = candidates(); print(f"{len(df)} candidate entries; all-trades mean net {df.net.mean()*100:.2f}%")
    S = score_all(df, model, n); evaluate(S, model)
