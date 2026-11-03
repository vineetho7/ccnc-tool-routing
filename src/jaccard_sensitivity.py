"""Does the confusable/distinct gap survive other Jaccard thresholds? (LLM-free)"""
import json, re, sys, itertools, collections, numpy as np
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, R, subset_indices

cat = json.load(open(f"{D}/catalog.json"))
idx_of = {t["tool"]: i for i, t in enumerate(cat)}
qs = json.load(open(f"{D}/queries_eval.json"))
N_ALL = len(cat)

def jac(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if A | B else 0.0

by_key = collections.defaultdict(list)
for i, t in enumerate(cat):
    by_key[(t["method"], t["verb"])].append(i)

def confusers_at(thr):
    conf = collections.defaultdict(set)
    for key, idxs in by_key.items():
        for i, j in itertools.combinations(idxs, 2):
            a, b = cat[i], cat[j]
            if a["api"] == b["api"]:
                continue
            if jac(a["entity_tokens"], b["entity_tokens"]) >= thr:
                conf[a["tool"]].add(b["tool"]); conf[b["tool"]].add(a["tool"])
    return conf

sel = {r["qid"]: r for r in json.load(open(f"{R}/sel_dense5_n1062_qwen3.5-latest.json"))}
out = []
for thr in (0.3, 0.5, 0.7, 0.9):
    conf = confusers_at(thr)
    n_conf_tools = sum(1 for t in cat if conf.get(t["tool"]))
    c, d = [], []
    for q in qs:
        g = idx_of[q["gold_tool"]]
        sub = set(subset_indices(q["qid"], g, N_ALL, N_ALL))
        present = any(idx_of[x] in sub for x in conf.get(q["gold_tool"], ()))
        (c if present else d).append(sel[q["qid"]]["correct"])
    row = {"thr": thr, "conf_tools_pct": round(100*n_conf_tools/len(cat), 1),
           "n_conf_q": len(c), "acc_conf": round(100*np.mean(c), 1) if c else None,
           "n_dist_q": len(d), "acc_dist": round(100*np.mean(d), 1) if d else None}
    row["gap"] = round(row["acc_dist"] - row["acc_conf"], 1) if c and d else None
    out.append(row)
json.dump(out, open(f"{R}/jaccard_sensitivity.json", "w"), indent=1)
print(f"{'J>=':>5} {'conf tools %':>13} {'n_conf':>7} {'acc_conf':>9} {'n_dist':>7} {'acc_dist':>9} {'gap':>6}")
for r in out:
    print(f"{r['thr']:>5} {r['conf_tools_pct']:>13} {r['n_conf_q']:>7} {r['acc_conf']:>9} "
          f"{r['n_dist_q']:>7} {r['acc_dist']:>9} {r['gap']:>6}")
