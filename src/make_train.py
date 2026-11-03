"""Template-only router training data (no LLM): keeps eval queries out-of-distribution."""
import json, random, re, sys
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, canon_name

rng = random.Random(1337)
cat = json.load(open(f"{D}/catalog.json"))

VERB_PHRASE = {
    "list": ["list", "show me", "give me all", "fetch all", "pull up", "enumerate", "I need all", "what are the"],
    "retrieve": ["get", "retrieve", "show", "look up", "fetch", "open", "pull", "display"],
    "create": ["create", "raise", "open", "submit", "register", "add", "file", "set up"],
    "patch": ["update", "modify", "change", "amend", "edit", "adjust", "revise"],
    "update": ["update", "modify", "change", "amend", "edit"],
    "delete": ["delete", "remove", "purge", "drop", "cancel and remove", "tear down"],
    "op": ["run", "execute", "trigger", "invoke", "perform"],
    "get": ["get", "fetch", "read", "look up"],
}
TAIL = ["", " please", " for me", " right now", " in the OSS", " on the production system",
        " for the north-east region", " urgently", " before the maintenance window",
        " for today's audit", " so I can check the state"]
FRAME = [
    "{v} {ent}{p}{t}",
    "I need to {v} {ent}{p}{t}",
    "Can you {v} {ent}{p}{t}?",
    "{v} the {ent} entries{p}{t}",
    "As part of {api}, {v} {ent}{p}{t}",
    "Operator request: {v} {ent}{p}{t}",
    "We have to {v} {ent}{p}{t}",
    "{v} {ent} from the {api}{t}",
]

def words(e):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", e)
    return re.sub(r"[^A-Za-z0-9 ]", " ", s).strip().lower()

def api_phrase(t):
    s = re.sub(r"\s*API\s*$", "", t["api_title"]).strip()
    return s if s and len(s) < 60 else t["api"]

ex = []
PER = 10
for t in cat:
    ent = words(t["entity"]) or words(t["operation_id"])
    verbs = VERB_PHRASE.get(t["verb"], VERB_PHRASE["op"])
    reqp = [str(p.get("name")) for p in t["params"] if p.get("required") and p.get("name")][:3]
    for i in range(PER):
        v = rng.choice(verbs)
        frame = rng.choice(FRAME)
        p = ""
        if reqp and rng.random() < 0.55:
            pn = rng.choice(reqp)
            p = f" with {words(pn)} {rng.choice(['AB-7781','X9012','3345','site-204','NE-1182','ORD-55219'])}"
        q = frame.format(v=v, ent=ent, p=p, t=rng.choice(TAIL), api=api_phrase(t))
        q = q[0].upper() + q[1:]
        ex.append({"q": q, "tool": t["tool"], "api": t["api"]})
    # one spec-grounded paraphrase from the summary itself
    if t["summary"]:
        s = t["summary"].rstrip(". ")
        ex.append({"q": f"{s} ({api_phrase(t)})", "tool": t["tool"], "api": t["api"]})

rng.shuffle(ex)
SYS = ("You are a tool router for telecom OSS/BSS management APIs (TM Forum Open APIs). "
       "Given an operator request, answer with the single TM Forum operation that fulfils it, "
       "formatted as TMFxxx_operationId. Answer with the name only, nothing else.")
def rec(e):
    return {"messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": e["q"]},
                         {"role": "assistant", "content": e["tool"]}]}
nv = max(200, len(ex)//40)
import os
os.makedirs(f"{D}/lora", exist_ok=True)
for name, part in (("train", ex[nv:]), ("valid", ex[:nv])):
    with open(f"{D}/lora/{name}.jsonl", "w") as f:
        for e in part:
            f.write(json.dumps(rec(e)) + "\n")
print("train", len(ex)-nv, "valid", nv, "tools", len({e['tool'] for e in ex}))
print(ex[0]); print(ex[1])
