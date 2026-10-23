import numpy as np
import pandas as pd

from .market import Market


def load_universe(name):
    import strat_search as S
    if name == "majors": return Market(S.load_tf("1d")[1], 86400)
    meta = pd.read_pickle("data/ALLD_meta.pkl"); bars = {}; last = max(int(pd.read_csv(f"data/ALLD_{c}USDT.csv").t.iloc[-1]) for c in meta["ok"])
    for c in meta["ok"]:
        d = pd.read_csv(f"data/ALLD_{c}USDT.csv")
        if int(d.t.iloc[-1]) < last - 3 * 86400: continue
        if pd.to_datetime(d.t.iloc[0], unit="s") <= pd.Timestamp("2023-09-01") and d.qv.tail(90).mean() >= 5e6:
            cal = np.arange(int(pd.Timestamp("2023-09-01").timestamp()), int(d.t.iloc[-1]) + 1, 86400); dd = d.set_index("t").reindex(cal)
            if dd.c.isna().sum() <= 3: bars[c + "USDT"] = dd.ffill().bfill().reset_index().rename(columns={"index": "t"})[["t", "o", "h", "l", "c", "v"]]
    return Market(bars, 86400)
