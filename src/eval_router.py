"""Evaluate the SLM router (base or LoRA-tuned) in description-free mode."""
import json, sys, time, numpy as np
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, R, SEV_COST, load_catalog, load_queries, subset_indices
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ADAPTER = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "none" else None
TAG = sys.argv[2] if len(sys.argv) > 2 else ("router_tuned" if ADAPTER else "router_base")
BASE = "mlx-community/Qwen3-1.7B-bf16"

SYS = ("You are a tool router for telecom OSS/BSS management APIs (TM Forum Open APIs). "
       "Given an operator request, answer with the single TM Forum operation that fulfils it, "
       "formatted as TMFxxx_operationId. Answer with the name only, nothing else.")

model, tok = load(BASE, adapter_path=ADAPTER)
cat, byname = load_catalog()
idx_of = {t["tool"]: i for i, t in enumerate(cat)}
N_ALL = len(cat)
qs = load_queries()
SIZES = [25, 50, 100, 200, 400, 800, N_ALL]
sampler = make_sampler(temp=0.0)

out, t0 = [], time.time()
for qi, q in enumerate(qs):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q["query"]}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
    t1 = time.time()
    txt = generate(model, tok, prompt=prompt, max_tokens=24, sampler=sampler, verbose=False)
    wall = time.time() - t1
    pred_raw = txt.strip().split("\n")[0].strip().strip('`"\' .,')
    pred = pred_raw if pred_raw in byname else None
    ptok = len(tok.encode(prompt))
    rec = {"qid": q["qid"], "method": TAG, "gold": q["gold_tool"], "pred_raw": pred_raw,
           "pred": pred, "gold_sev": q["severity"], "global_confusable": q["confusable"],
           "p_tok": ptok, "g_tok": len(tok.encode(pred_raw)), "wall": wall}
    g = idx_of[q["gold_tool"]]
    for n in SIZES:
        sub = set(subset_indices(q["qid"], g, min(n, N_ALL), N_ALL))
        conf = sum(1 for i in sub if cat[i]["tool"] in set(byname[q["gold_tool"]]["confusers"]))
        ok = int(pred == q["gold_tool"])
        in_cat = pred is not None and idx_of.get(pred) in sub
        psev = byname[pred]["severity"] if (pred and in_cat) else 0
        rec[f"n{n}"] = {"correct": ok, "confusable": conf > 0, "n_conf_present": conf,
                        "pred_in_catalog": int(bool(in_cat)),
                        "sev_cost": 0.0 if ok else SEV_COST.get(psev, 1.0),
                        "destructive_misfire": int((not ok) and q["severity"] == 0 and psev == 3)}
    out.append(rec)
    if (qi + 1) % 25 == 0:
        el = time.time() - t0
        acc = 100*np.mean([o[f"n{N_ALL}"]["correct"] for o in out])
        print(f"{qi+1}/{len(qs)} acc={acc:.1f} {el:.0f}s eta={el/(qi+1)*(len(qs)-qi-1):.0f}s", flush=True)

json.dump(out, open(f"{R}/{TAG}.json", "w"), indent=1)
for n in SIZES:
    r = [o[f"n{n}"] for o in out]
    c = [x for x in r if x["confusable"]]; d = [x for x in r if not x["confusable"]]
    m = lambda a, k: 100*np.mean([x[k] for x in a]) if a else float("nan")
    print(f"n={n:5d} acc={m(r,'correct'):5.1f} conf({len(c):3d})={m(c,'correct'):5.1f} dist({len(d):3d})={m(d,'correct'):5.1f}")
print("mean prompt tok", np.mean([o["p_tok"] for o in out]), "wall", np.mean([o["wall"] for o in out]))
