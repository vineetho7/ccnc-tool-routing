"""End-to-end tool-selection eval. Usage: eval_select.py METHOD N [MODEL]"""
import json, sys, re, time, numpy as np
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import *
from rank_bm25 import BM25Okapi

METHOD = sys.argv[1]            # full | full_rename | dense5 | dense5_rename | bm25_5 | direct ; suffix _abst = gold removed
N = int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else BIG
TAG = f"{METHOD}_n{N}_{MODEL.replace(':','-')}"

cat, byname = load_catalog()
docs = json.load(open(f"{D}/tool_docs.json"))
E = np.load(f"{D}/tool_emb.npy")
QE = np.load(f"{D}/query_emb.npy")
idx_of = {t["tool"]: i for i, t in enumerate(cat)}
N_ALL = len(cat)

def tok(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w.lower() for w in re.findall(r"[A-Za-z0-9]+", s)]
bm = BM25Okapi([tok(d) for d in docs]) if "bm25" in METHOD else None

SYS = ("You are a tool router for telecom OSS/BSS management APIs. "
       "Pick exactly one tool from the candidate list that fulfils the operator's request. "
       "If no candidate fits, answer NONE. Answer with the tool name only, nothing else.")
SYS_DIRECT = ("You are a tool router for telecom OSS/BSS management APIs (TM Forum Open APIs). "
              "Given an operator request, answer with the single TM Forum operation that fulfils it, "
              "formatted as TMFxxx_operationId. Answer with the name only, nothing else.")

qs = load_queries()
import os as _os
_lim = int(_os.environ.get("LIMIT", "0"))
if _lim:
    qs = qs[:_lim]
ABST = METHOD.endswith("_abst")
if ABST:
    METHOD_BASE = METHOD[:-5]
else:
    METHOD_BASE = METHOD
rename = METHOD_BASE.endswith("_rename")
out, t0 = [], time.time()
for qi, q in enumerate(qs):
    g = idx_of[q["gold_tool"]]
    sub = subset_indices(q["qid"], g, min(N, N_ALL), N_ALL)
    if ABST:
        sub = [i for i in sub if i != g]
    gold_disp = canon_name(cat[g]) if rename else q["gold_tool"]
    conf_present = sum(1 for i in sub if cat[i]["tool"] in set(byname[q["gold_tool"]]["confusers"]))

    if METHOD_BASE == "direct":
        r = ollama_chat([{"role": "system", "content": SYS_DIRECT},
                         {"role": "user", "content": q["query"]}], model=MODEL, num_predict=32)
        cand = sub
    else:
        if METHOD_BASE.startswith("full"):
            cand = sub
        elif METHOD_BASE.startswith("dense"):
            sims = E @ QE[qi]
            cand = sorted(sub, key=lambda i: -sims[i])[:5]
        elif METHOD_BASE.startswith("bm25"):
            sc = np.array(bm.get_scores(tok(q["query"])), dtype=np.float32)
            cand = sorted(sub, key=lambda i: -sc[i])[:5]
        lines = "\n".join(schema_line(cat[i], rename) for i in cand)
        r = ollama_chat([{"role": "system", "content": SYS},
                         {"role": "user", "content": f"Candidate tools:\n{lines}\n\nOperator request: {q['query']}\n\nTool name:"}],
                        model=MODEL, num_predict=32, num_ctx=32768)

    pred_raw = r["text"].strip().split("\n")[0].strip().strip('`"\' .,')
    disp = {(canon_name(cat[i]) if rename else cat[i]["tool"]): cat[i]["tool"] for i in cand}
    pred_tool = disp.get(pred_raw)
    if pred_tool is None:                       # fuzzy: substring / case-insensitive match
        low = {k.lower(): v for k, v in disp.items()}
        pred_tool = low.get(pred_raw.lower())
    if pred_tool is None and METHOD_BASE == "direct":
        pred_tool = byname.get(pred_raw, {}).get("tool") or \
                    next((t["tool"] for t in cat if canon_name(t) == pred_raw.lower()), None)
    said_none = pred_raw.strip().upper().startswith("NONE")
    correct = int(said_none) if ABST else int(pred_tool == q["gold_tool"] and not said_none)
    psev = byname[pred_tool]["severity"] if pred_tool in byname else 0
    out.append({"qid": q["qid"], "method": METHOD, "n": N, "model": MODEL,
                "gold": q["gold_tool"], "pred_raw": pred_raw, "pred": pred_tool,
                "correct": correct, "in_candidates": int(g in cand),
                "confusable": conf_present > 0, "n_conf_present": conf_present,
                "global_confusable": q["confusable"],
                "gold_sev": q["severity"], "pred_sev": psev,
                "sev_cost": 0.0 if correct else SEV_COST.get(psev, 1.0),
                "destructive_misfire": int((not correct) and q["severity"] == 0 and psev == 3),
                "p_tok": r["p_tok"], "g_tok": r["g_tok"], "wall": r["wall"]})
    if (qi + 1) % 25 == 0:
        el = time.time() - t0
        acc = 100 * np.mean([o["correct"] for o in out])
        print(f"{qi+1}/{len(qs)} acc={acc:.1f} {el:.0f}s eta={el/(qi+1)*(len(qs)-qi-1):.0f}s", flush=True)

json.dump(out, open(f"{R}/sel_{TAG}.json", "w"), indent=1)
acc = 100*np.mean([o["correct"] for o in out])
cf = [o for o in out if o["confusable"]]; ds = [o for o in out if not o["confusable"]]
m = lambda a, k: 100*np.mean([x[k] for x in a]) if a else float("nan")
print(f"DONE {TAG}: acc={acc:.1f} conf({len(cf)})={m(cf,'correct'):.1f} dist({len(ds)})={m(ds,'correct'):.1f} "
      f"sevcost={np.mean([o['sev_cost'] for o in out]):.2f} destmis={m(out,'destructive_misfire'):.1f} "
      f"ptok={np.mean([o['p_tok'] for o in out]):.0f} wall={np.mean([o['wall'] for o in out]):.2f}s")
