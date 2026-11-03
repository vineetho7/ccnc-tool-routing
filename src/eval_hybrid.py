"""Hybrid: union of dense top-5 and the tuned router's top-1, then the 8B model picks."""
import json, sys, re, time, numpy as np
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1062
cat, byname = load_catalog()
E = np.load(f"{D}/tool_emb.npy"); QE = np.load(f"{D}/query_emb.npy")
idx_of = {t["tool"]: i for i, t in enumerate(cat)}
N_ALL = len(cat)
router = {o["qid"]: o["pred"] for o in json.load(open(f"{R}/router_tuned.json"))}
qs = load_queries()
import os as _os
_lim = int(_os.environ.get("LIMIT", "0"))
if _lim:
    qs = qs[:_lim]
SYS = ("You are a tool router for telecom OSS/BSS management APIs. "
       "Pick exactly one tool from the candidate list that fulfils the operator's request. "
       "If no candidate fits, answer NONE. Answer with the tool name only, nothing else.")
out, t0 = [], time.time()
for qi, q in enumerate(qs):
    g = idx_of[q["gold_tool"]]
    sub = subset_indices(q["qid"], g, min(N, N_ALL), N_ALL)
    sims = E @ QE[qi]
    cand = sorted(sub, key=lambda i: -sims[i])[:5]
    rp = router.get(q["qid"])
    if rp in idx_of and idx_of[rp] in set(sub) and idx_of[rp] not in cand:
        cand = [idx_of[rp]] + cand
    conf = sum(1 for i in sub if cat[i]["tool"] in set(byname[q["gold_tool"]]["confusers"]))
    lines = "\n".join(schema_line(cat[i]) for i in cand)
    r = ollama_chat([{"role": "system", "content": SYS},
                     {"role": "user", "content": f"Candidate tools:\n{lines}\n\nOperator request: {q['query']}\n\nTool name:"}],
                    num_predict=32)
    pred_raw = r["text"].strip().split("\n")[0].strip().strip('`"\' .,')
    disp = {cat[i]["tool"]: cat[i]["tool"] for i in cand}
    pred = disp.get(pred_raw) or {k.lower(): v for k, v in disp.items()}.get(pred_raw.lower())
    ok = int(pred == q["gold_tool"])
    psev = byname[pred]["severity"] if pred in byname else 0
    out.append({"qid": q["qid"], "method": "hybrid", "n": N, "gold": q["gold_tool"],
                "pred_raw": pred_raw, "pred": pred, "correct": ok,
                "in_candidates": int(g in cand), "confusable": conf > 0,
                "n_conf_present": conf, "global_confusable": q["confusable"],
                "gold_sev": q["severity"], "pred_sev": psev,
                "sev_cost": 0.0 if ok else SEV_COST.get(psev, 1.0),
                "destructive_misfire": int((not ok) and q["severity"] == 0 and psev == 3),
                "p_tok": r["p_tok"], "g_tok": r["g_tok"], "wall": r["wall"]})
    if (qi+1) % 50 == 0:
        print(f"{qi+1}/{len(qs)} acc={100*np.mean([o['correct'] for o in out]):.1f}", flush=True)
json.dump(out, open(f"{R}/sel_hybrid_n{N}_hybrid.json", "w"), indent=1)
print("hybrid acc", 100*np.mean([o["correct"] for o in out]))
