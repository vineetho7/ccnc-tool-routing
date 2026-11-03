"""Re-score severity post-hoc from stored predictions.

Fixes three scoring defects found in review:
  * unresolved predictions were charged read-severity, making 0%-accuracy
    baselines look safest; they now get severity from the verb they named,
    and rows are flagged by resolution rate;
  * the router's severity was gated on whether its (fixed) prediction fell
    inside the sampled subset, which made cost grow with N for free;
  * destructive misfire required a resolvable prediction, so it could not fire.
"""
import json, re, sys, glob, os, collections
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, R, SEV_COST, load_catalog

cat, byname = load_catalog()
VERB_SEV = {"list": 0, "retrieve": 0, "get": 0, "read": 0, "fetch": 0, "show": 0,
            "create": 2, "post": 2, "add": 2, "patch": 2, "update": 2, "modify": 2,
            "put": 2, "replace": 2,
            "delete": 3, "remove": 3, "destroy": 3, "purge": 3}

def claimed_severity(pred_raw):
    """Severity implied by the name the model emitted, even if it resolves to no tool."""
    s = re.sub(r"^TMF\d+[_-]?", "", str(pred_raw)).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()
    toks = re.findall(r"[a-z]+", s)
    for t in toks[:2]:
        if t in VERB_SEV:
            return VERB_SEV[t]
    return None

def rescore_rows(rows, router=False, n=None):
    out = []
    for o in rows:
        pred = o.get("pred")
        resolved = pred in byname
        if resolved:
            psev = byname[pred]["severity"]
        else:
            psev = claimed_severity(o.get("pred_raw", ""))
        correct = o["correct"] if not router else o["correct"]
        rec = dict(o)
        rec["resolved"] = int(resolved)
        rec["psev2"] = psev
        rec["sev_cost2"] = 0.0 if correct else (SEV_COST[psev] if psev is not None else None)
        rec["dest_call"] = int((not correct) and psev == 3)
        rec["escalation"] = int((not correct) and psev is not None
                                and psev > o.get("gold_sev", 0))
        out.append(rec)
    return out

def summarise(rows, label):
    n = len(rows)
    res = 100 * sum(r["resolved"] for r in rows) / n
    costed = [r["sev_cost2"] for r in rows if r["sev_cost2"] is not None]
    cov = 100 * len(costed) / n
    acc = 100 * sum(r["correct"] for r in rows) / n
    return {"method": label, "N": n, "acc": round(acc, 1),
            "resolved_pct": round(res, 1), "sev_cov_pct": round(cov, 1),
            "sev_cost": round(sum(costed) / len(costed), 2) if costed else None,
            "dest_call_pct": round(100 * sum(r["dest_call"] for r in rows) / n, 1),
            "escalation_pct": round(100 * sum(r["escalation"] for r in rows) / n, 1)}

NQ = len(json.load(open(f"{os.path.dirname(R)}/data/queries.json"))) if os.path.exists(f"{os.path.dirname(R)}/data/queries.json") else len(json.load(open("data/queries.json")))

summary = []
for f in sorted(glob.glob(f"{R}/sel_*.json")):
    rows = json.load(open(f))
    if not rows:
        continue
    m, n = rows[0]["method"], rows[0]["n"]
    assert len(rows) == NQ, f"{f} has {len(rows)} rows, expected {NQ}"   # smoke-run guard
    rs = rescore_rows(rows)
    s = summarise(rs, m); s["n"] = n
    summary.append(s)

for f in sorted(glob.glob(f"{R}/router_*.json")):
    base = os.path.basename(f)[:-5]
    if base.endswith("long"):
        continue                      # byte-identical duplicate of router_tuned
    data = json.load(open(f))
    assert len(data) == NQ, f"{f} has {len(data)} rows, expected {NQ}"
    rows = [{"correct": o[f"n{len(cat)}"]["correct"], "pred": o["pred"],
             "pred_raw": o["pred_raw"], "gold_sev": o["gold_sev"]} for o in data]
    rs = rescore_rows(rows, router=True)
    s = summarise(rs, base); s["n"] = "any"
    summary.append(s)

json.dump(summary, open(f"{R}/rescored.json", "w"), indent=1)
hdr = f"{'method':24s} {'n':>5s} {'acc':>6s} {'res%':>6s} {'sevcov%':>8s} {'c_sev':>6s} {'destcall%':>10s} {'escal%':>7s}"
print(hdr); print("-" * len(hdr))
for s in summary:
    c = f"{s['sev_cost']:.2f}" if s["sev_cost"] is not None else "  -- "
    print(f"{s['method']:24s} {str(s['n']):>5s} {s['acc']:6.1f} {s['resolved_pct']:6.1f} "
          f"{s['sev_cov_pct']:8.1f} {c:>6s} {s['dest_call_pct']:10.1f} {s['escalation_pct']:7.1f}")
