"""Stratified sample of generated queries for manual inspection."""
import json, random, sys, csv
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, R, load_catalog

cat, byname = load_catalog()
qs = json.load(open(f"{D}/queries_eval.json"))
rng = random.Random(4242)
strata = {}
for q in qs:
    strata.setdefault((q["confusable"], q["severity"]), []).append(q)
per = 100 // len(strata)
sample = []
for k in sorted(strata, key=str):
    v = strata[k][:]; rng.shuffle(v); sample += v[:per]
rng.shuffle(sample)
sample = sample[:100]
with open(f"{R}/query_audit_sample.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qid", "query", "gold_tool", "gold_summary", "confusable", "severity",
                "plausible(y/n)", "uniquely_implies_gold(y/n)", "note"])
    for q in sample:
        t = byname[q["gold_tool"]]
        w.writerow([q["qid"], q["query"], q["gold_tool"], t["summary"][:120],
                    int(q["confusable"]), q["severity"], "", "", ""])
print("wrote", len(sample), "rows")
for q in sample:
    t = byname[q["gold_tool"]]
    print(f'{q["qid"]} | {q["query"]}\n    GOLD {q["gold_tool"]} :: {t["summary"][:95]}')
