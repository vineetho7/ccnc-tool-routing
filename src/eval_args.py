"""Stage-2: argument generation, conditioned on the (correct) tool. Separated from selection."""
import json, sys, re, time, numpy as np
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import *

MODEL = sys.argv[1] if len(sys.argv) > 1 else BIG
cat, byname = load_catalog()
qs = load_queries()

def norm(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower())

def schema_block(t):
    ps = [{"name": p.get("name"), "in": p.get("in"), "type": p.get("type"),
           "required": bool(p.get("required")), "desc": (p.get("desc") or "")[:80]}
          for p in t["params"] if p.get("name")][:12]
    return (f"Tool: {t['tool']}\n{t['method']} {t['path']}\nPurpose: {t['summary']}\n"
            f"Parameters: {json.dumps(ps, indent=0)}")

out, t0 = [], time.time()
for qi, q in enumerate(qs):
    t = byname[q["gold_tool"]]
    valid = {str(p.get("name")) for p in t["params"] if p.get("name")}
    gold_raw = {k: v for k, v in (q.get("gold_args") or {}).items() if v not in (None, "")}
    # the generator sometimes invents keys that are not parameters of the tool;
    # scoring against those measures generator noise, not the model
    gold = {k: v for k, v in gold_raw.items() if k in valid}
    if not gold:
        continue
    r = ollama_chat([{"role": "system", "content": "You fill tool call arguments. Output strict JSON only."},
                     {"role": "user", "content": f"{schema_block(t)}\n\nOperator request: {q['query']}\n\n"
                      f"Return JSON object mapping parameter name to value, for the parameters the request specifies. JSON only."}],
                    model=MODEL, num_predict=160)
    m = re.search(r"\{.*\}", r["text"], re.S)
    pred = {}
    if m:
        try:
            pred = json.loads(m.group(0))
        except Exception:
            pred = {}
    pred = {k: v for k, v in pred.items() if isinstance(k, str)}
    keys_g, keys_p = set(gold), set(pred)
    inter = keys_g & keys_p
    val_ok = sum(1 for k in inter if norm(pred[k]) == norm(gold[k]))
    out.append({"qid": q["qid"], "tool": q["gold_tool"], "n_gold": len(gold),
                "n_gold_raw": len(gold_raw), "dropped_keys": len(gold_raw) - len(gold),
                "key_p": len(inter)/len(keys_p) if keys_p else 0.0,
                "key_r": len(inter)/len(keys_g),
                "value_em": val_ok/len(keys_g),
                "exact_all": int(val_ok == len(keys_g) and keys_p == keys_g),
                "p_tok": r["p_tok"], "g_tok": r["g_tok"], "wall": r["wall"]})
    if (qi+1) % 50 == 0:
        print(f"{qi+1}/{len(qs)} {time.time()-t0:.0f}s", flush=True)

json.dump(out, open(f"{R}/args_{MODEL.replace(':','-')}.json", "w"), indent=1)
f = lambda k: 100*np.mean([o[k] for o in out])
print(f"ARGS n={len(out)} key_P={f('key_p'):.1f} key_R={f('key_r'):.1f} valueEM={f('value_em'):.1f} allExact={f('exact_all'):.1f}")
