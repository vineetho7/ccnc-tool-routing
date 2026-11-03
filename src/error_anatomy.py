"""What kind of wrong tool does the selector pick?"""
import json, re, sys, collections, glob
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import R

def verb(t):
    """First camelCase token of the operationId, lowercased; None if unparseable."""
    op = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", t.split("_", 1)[1])
    toks = re.findall(r"[A-Za-z]+", op)
    return toks[0].lower() if toks else None

out = {}
for f in sorted(glob.glob(f"{R}/sel_*.json")):
    d = json.load(open(f))
    if not d or d[0].get("method", "").endswith("_abst"):
        continue
    m, n = d[0]["method"], d[0]["n"]
    err = [o for o in d if not o["correct"] and o.get("pred")]
    if len(err) < 20:
        continue
    pair = collections.Counter((o["gold_sev"], o["pred_sev"]) for o in err)
    errs_total = [o for o in d if not o["correct"]]
    out[f"{m}_n{n}"] = {
        "queries": len(d), "errors_total": len(errs_total), "errors_named": len(err),
        "named_frac_pct": round(100*len(err)/max(1, len(errs_total)), 1),
        "same_verb_pct": round(100*sum(1 for o in err if verb(o["gold"]) is not None and verb(o["gold"]) == verb(o["pred"]))/len(err), 1),
        "same_sev_pct": round(100*sum(1 for o in err if o["gold_sev"] == o["pred_sev"])/len(err), 1),
        "same_api_pct": round(100*sum(1 for o in err if o["gold"].split("_")[0] == o["pred"].split("_")[0])/len(err), 1),
        "dest_for_dest": pair.get((3, 3), 0),
        "escalations": sum(v for (g, p), v in pair.items() if p > g),
        "deescalations": sum(v for (g, p), v in pair.items() if p < g),
    }
json.dump(out, open(f"{R}/error_anatomy.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:34s} err={v['errors_named']:3d} sameVerb={v['same_verb_pct']:5.1f} "
          f"sameSev={v['same_sev_pct']:5.1f} sameAPI={v['same_api_pct']:5.1f} "
          f"DEL->DEL={v['dest_for_dest']:3d} up={v['escalations']:2d} down={v['deescalations']:2d}")
