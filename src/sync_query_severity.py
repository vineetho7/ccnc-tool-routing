"""Re-sync each query's gold severity from the catalogue (PUT reclassified as state-changing)."""
import json, sys
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, load_catalog
cat, byname = load_catalog()
qs = json.load(open(f"{D}/queries_eval.json"))
changed = 0
for q in qs:
    s = byname[q["gold_tool"]]["severity"]
    if q.get("severity") != s:
        q["severity"] = s; changed += 1
json.dump(qs, open(f"{D}/queries_eval.json", "w"), indent=1)
print(f"synced severity on {changed}/{len(qs)} queries")
