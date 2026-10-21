"""Summarise replay folders: python scripts/summarize.py logs/w10"""
import collections, glob, json, sys
import numpy as np
root = sys.argv[1]; R = collections.defaultdict(dict); C = collections.defaultdict(lambda: [0, 0])
for f in glob.glob(f"{root}/*/*/summary.json"):
    v, e = f.split("/")[-3:-1]; R[v][e] = json.load(open(f))
    for l in open(f.replace("summary.json", "decisions.jsonl")):
        d = json.loads(l)
        if d["kind"] == "cycle": C[v][0] += 1; C[v][1] += bool(d["candidates"])
print(f"{'variant':26s} {'windows':>7s} {'avg ret':>8s} {'median':>7s} {'win.w':>5s} {'worst':>7s} {'worstDD':>7s} {'halts':>5s} {'trades':>6s} {'win%':>5s} {'cand%':>5s}")
for v, d in sorted(R.items(), key=lambda kv: -np.mean([x["return_pct"] for x in kv[1].values()])):
    r = np.array([x["return_pct"] for x in d.values()]); dd = [x["max_drawdown_pct"] for x in d.values()]
    tr = sum(x["trades"] for x in d.values()); w = sum(x["wins"] for x in d.values()); n, k = C[v]
    print(f"{v:26s} {len(r):7d} {r.mean():+7.2f}% {np.median(r):+6.2f}% {int((r > 0).sum()):2d}/{len(r):<2d} {r.min():+6.2f}% {max(dd):6.2f}% {sum(x >= 5 for x in dd):5d} {tr:6d} {w / max(tr, 1) * 100:4.0f}% {k / max(n, 1) * 100:4.0f}%")
