"""Controlled test: same query, same catalogue size, confusers absent vs forced in."""
import json, sys, random, re, numpy as np
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, R
from rank_bm25 import BM25Okapi

N = 200
cat = json.load(open(f"{D}/catalog.json"))
byname = {t["tool"]: t for t in cat}
docs = json.load(open(f"{D}/tool_docs.json"))
E = np.load(f"{D}/tool_emb.npy"); QE = np.load(f"{D}/query_emb.npy")
idx_of = {t["tool"]: i for i, t in enumerate(cat)}
qs = json.load(open(f"{D}/queries_eval.json"))

def tok(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w.lower() for w in re.findall(r"[A-Za-z0-9]+", s)]
bm = BM25Okapi([tok(d) for d in docs])

rows = []
for qi, q in enumerate(qs):
    gold = q["gold_tool"]
    confs = [idx_of[c] for c in byname[gold]["confusers"] if c in idx_of]
    if not confs:
        continue                      # only queries that HAVE confusers can be contrasted
    g = idx_of[gold]
    rng = random.Random(f"ctl:{q['qid']}")
    clean_pool = [i for i in range(len(cat)) if i != g and i not in set(confs)]
    rng.shuffle(clean_pool)
    k = min(len(confs), 4)
    base = clean_pool[: N - 1]                       # confusers absent
    inj = clean_pool[: N - 1 - k] + confs[:k]        # confusers present, same size
    sims = E @ QE[qi]; bsc = np.array(bm.get_scores(tok(q["query"])), dtype=np.float32)
    for cond, sub in (("absent", base), ("present", inj)):
        s = [g] + sub
        for name, sc in (("dense", sims), ("bm25", bsc)):
            order = sorted(s, key=lambda i: -sc[i])
            r = order.index(g) + 1
            rows.append({"qid": q["qid"], "cond": cond, "retriever": name, "rank": r,
                         "hit@1": int(r <= 1), "hit@5": int(r <= 5), "k_inj": k})
json.dump(rows, open(f"{R}/controlled_conf.json", "w"))
n = len({r["qid"] for r in rows})
print(f"paired queries: {n} (N={N}, up to 4 confusers injected)")
for ret in ("dense", "bm25"):
    for k in ("hit@1", "hit@5"):
        a = [r[k] for r in rows if r["retriever"] == ret and r["cond"] == "absent"]
        b = [r[k] for r in rows if r["retriever"] == ret and r["cond"] == "present"]
        d = np.array(b) - np.array(a)
        se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0
        print(f"{ret:6s} {k}: absent={100*np.mean(a):5.1f}  present={100*np.mean(b):5.1f} "
              f" delta={100*np.mean(d):+5.1f} +/- {100*1.96*se:.1f}")
