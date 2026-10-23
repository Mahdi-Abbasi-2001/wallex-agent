import numpy as np


def metrics(eq, start_cash, bpy):
    eq = np.asarray(eq, float); full = np.r_[start_cash, eq]; r = full[1:] / np.maximum(full[:-1], 1e-12) - 1
    dd = (1 - full / np.maximum.accumulate(full)).max(); sd = r.std()
    ret = full[-1] / start_cash - 1
    return dict(ret=ret, mdd=dd, sharpe=(r.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0, calmar=ret / max(dd, 0.01))
