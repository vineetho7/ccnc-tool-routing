import json, sys, numpy as np
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D
from embed import embed
qs = json.load(open(f"{D}/queries_eval.json"))
E = embed([q["query"] for q in qs], "search_query: ")
np.save(f"{D}/query_emb.npy", E)
print("query emb", E.shape)
