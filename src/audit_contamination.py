"""Quantify the 'service catalog' filler leak introduced by the generation prompt."""
import json, re, sys, collections, numpy as np
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, R, load_catalog

cat, byname = load_catalog()
qs = json.load(open(f"{D}/queries_eval.json"))

CUE = re.compile(r"\bservice catalog(ue)?\b", re.I)
# APIs where "service catalog" is a legitimate description of the target
LEGIT_APIS = {"TMF633", "TMF909"}

flag = []
for q in qs:
    if not CUE.search(q["query"]):
        continue
    t = byname[q["gold_tool"]]
    title = t["api_title"].lower()
    legit = (t["api"] in LEGIT_APIS) or ("service catalog" in title)
    flag.append((q["qid"], legit, t["api"], t["api_title"][:42], q["query"][:88]))

n_cue = len(flag)
n_bad = sum(1 for f in flag if not f[1])
print(f"queries mentioning 'service catalog': {n_cue}/{len(qs)} ({100*n_cue/len(qs):.1f}%)")
print(f"  of those, gold tool is NOT a service-catalog API: {n_bad} ({100*n_bad/len(qs):.1f}% of all queries)")
print("\nexamples of misleading ones:")
for qid, legit, api, title, text in flag:
    if not legit:
        print(f"  {qid} [{api} {title}] {text}")
        if flag.index((qid, legit, api, title, text)) > 400: break

bad_ids = {f[0] for f in flag if not f[1]}
json.dump(sorted(bad_ids), open(f"{R}/contaminated_qids.json", "w"))

# does excluding them change the headline?
sel = json.load(open(f"{R}/sel_dense5_n1062_qwen3.5-latest.json"))
hyb = json.load(open(f"{R}/sel_hybrid_n1062_hybrid.json"))
def report(rows, name):
    keep = [r for r in rows if r["qid"] not in bad_ids]
    for label, rs in (("all", rows), ("clean", keep)):
        c = [r for r in rs if r["confusable"]]; d = [r for r in rs if not r["confusable"]]
        print(f"  {name:8s} {label:6s} n={len(rs):3d} acc={100*np.mean([r['correct'] for r in rs]):5.1f} "
              f"conf={100*np.mean([r['correct'] for r in c]):5.1f} dist={100*np.mean([r['correct'] for r in d]):5.1f} "
              f"gap={100*np.mean([r['correct'] for r in d])-100*np.mean([r['correct'] for r in c]):5.1f}")
print("\nheadline with and without the contaminated queries:")
report(sel, "dense5"); report(hyb, "hybrid")
