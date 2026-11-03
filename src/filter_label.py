"""Filter catalog to core resource operations; label confusability (retriever-independent)."""
import json, re, collections, itertools, random

D = "/Users/vineeth/Documents/ccnc-tool-routing/data"
cat = json.load(open(f"{D}/catalog_full.json"))

def split_camel(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return [w.lower() for w in s.split() if w]

VERBS = {"list", "retrieve", "create", "patch", "update", "delete", "get", "post", "put", "cancel", "run", "start", "stop", "activate", "deactivate"}

keep = []
for t in cat:
    p = t["path"].lower()
    if "/listener/" in p or "/hub" in p:          # notification boilerplate
        continue
    if t["operation_id"].lower() in ("registerlistener", "unregisterlistener"):
        continue
    toks = split_camel(t["operation_id"])
    verb = toks[0] if toks and toks[0] in VERBS else "op"
    entity = [w for w in toks[1:]] or toks
    t["verb"] = verb
    t["entity_tokens"] = entity
    t["entity"] = "".join(w.capitalize() for w in entity)
    keep.append(t)

print("after filter:", len(keep), "APIs:", len({t['api'] for t in keep}))
print("verbs:", collections.Counter(t['verb'] for t in keep).most_common(10))

# ---- confusability: same method+verb, different API, entity-token Jaccard >= 0.5
def jac(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if A | B else 0.0

by_key = collections.defaultdict(list)
for i, t in enumerate(keep):
    by_key[(t["method"], t["verb"])].append(i)

for t in keep:
    t["confusers"] = []
for key, idxs in by_key.items():
    for i, j in itertools.combinations(idxs, 2):
        a, b = keep[i], keep[j]
        if a["api"] == b["api"]:
            continue
        s = jac(a["entity_tokens"], b["entity_tokens"])
        if s >= 0.5:
            a["confusers"].append(b["tool"]); b["confusers"].append(a["tool"])

for t in keep:
    t["n_confusers"] = len(t["confusers"])
    t["confusable"] = t["n_confusers"] > 0

n_c = sum(t["confusable"] for t in keep)
print(f"confusable: {n_c}/{len(keep)} ({100*n_c/len(keep):.1f}%)")
print("severity dist:", collections.Counter(t['severity'] for t in keep))
ex = [t for t in keep if t["n_confusers"] >= 3][:6]
for t in ex:
    print(" ", t["tool"], "->", t["confusers"][:4])

json.dump(keep, open(f"{D}/catalog.json", "w"), indent=1)
