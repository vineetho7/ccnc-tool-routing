"""LLM-free retrieval curve: recall@k for dense + BM25 over per-query catalog subsets."""
import json, sys, numpy as np, time
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import *
from rank_bm25 import BM25Okapi
import re

cat, byname = load_catalog()
docs = json.load(open(f"{D}/tool_docs.json"))
E = np.load(f"{D}/tool_emb.npy")
idx_of = {t["tool"]: i for i, t in enumerate(cat)}
N_ALL = len(cat)

def tok(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w.lower() for w in re.findall(r"[A-Za-z0-9]+", s)]

tokd = [tok(d) for d in docs]
bm = BM25Okapi(tokd)

qs = load_queries()
QE = np.load(f"{D}/query_emb.npy")
SIZES = [25, 50, 100, 200, 400, 800, N_ALL]
KS = [1, 3, 5, 10]

rows = []
for qi, q in enumerate(qs):
    g = idx_of[q["gold_tool"]]
    sims = E @ QE[qi]
    bsc = np.array(bm.get_scores(tok(q["query"])), dtype=np.float32)
    for n in SIZES:
        sub = subset_indices(q["qid"], g, min(n, N_ALL), N_ALL)
        conf_present = sum(1 for i in sub if cat[i]["tool"] in set(byname[q["gold_tool"]]["confusers"]))
        for name, sc in (("dense", sims), ("bm25", bsc)):
            order = sorted(sub, key=lambda i: -sc[i])
            rank = order.index(g) + 1
            rows.append({"qid": q["qid"], "n": n, "retriever": name, "rank": rank,
                         "conf_present": conf_present, "confusable": conf_present > 0,
                         "global_confusable": q["confusable"], "severity": q["severity"],
                         **{f"hit@{k}": int(rank <= k) for k in KS}})
json.dump(rows, open(f"{R}/retrieval_curve.json", "w"))
import collections
print(f"{len(qs)} queries, {len(rows)} rows")
for name in ("dense", "bm25"):
    print(f"-- {name}")
    for n in SIZES:
        r = [x for x in rows if x["n"] == n and x["retriever"] == name]
        c = [x for x in r if x["confusable"]]
        d = [x for x in r if not x["confusable"]]
        f = lambda a, k: 100*np.mean([x[f"hit@{k}"] for x in a]) if a else float("nan")
        print(f"  n={n:5d} hit@1={f(r,1):5.1f} hit@5={f(r,5):5.1f} | conf({len(c):3d}) @5={f(c,5):5.1f}"
              f" | dist({len(d):3d}) @5={f(d,5):5.1f}")
