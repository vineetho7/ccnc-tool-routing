"""Generate realistic operator utterances for a stratified sample of catalog tools."""
import json, random, re, sys, time, urllib.request, collections

D = "/Users/vineeth/Documents/ccnc-tool-routing/data"
GEN_MODEL = "qwen3.5:latest"
N_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 420
SEED = 20261004

def ollama(messages, model=GEN_MODEL, num_predict=320, temperature=0.8):
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "think": False, "options": {"num_predict": num_predict,
                       "temperature": temperature, "num_ctx": 8192}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as f:
        return json.load(f)["message"]["content"]

cat = json.load(open(f"{D}/catalog.json"))
rng = random.Random(SEED)

# stratified sample: confusable x severity, spread across APIs
strata = collections.defaultdict(list)
for t in cat:
    strata[(t["confusable"], t["severity"])].append(t)
per = max(1, N_TARGET // len(strata))
sample = []
for k, v in sorted(strata.items(), key=lambda x: str(x[0])):
    rng.shuffle(v)
    # cap per-API to keep diversity
    seen_api = collections.Counter()
    picked = []
    for t in v:
        if seen_api[t["api"]] >= 4:
            continue
        picked.append(t); seen_api[t["api"]] += 1
        if len(picked) >= per:
            break
    sample += picked
rng.shuffle(sample)
print(f"sampled {len(sample)} tools over {len({t['api'] for t in sample})} APIs", flush=True)

SYS = ("You write realistic one-sentence requests that a telecom network/OSS operations engineer "
       "would type into an AI assistant. Output strict JSON only.")

def prompt_for(t):
    entity_phrase = " ".join(t["entity_tokens"]) or t["entity"]
    req = [p for p in t["params"] if p.get("required")][:4]
    opt = [p for p in t["params"] if not p.get("required")][:5]
    return f"""API: {t['api_title']} ({t['api']})
Operation: {t['method']} {t['path']}
Purpose: {t['summary']}
Details: {t['description'][:300]}
Required parameters: {json.dumps([{ 'name': p['name'], 'in': p['in'], 'type': p['type']} for p in req])}
Other parameters: {json.dumps([p['name'] for p in opt])}

Write ONE request an operator would make that this operation (and only this operation) answers.
Rules:
- Natural operator language. Do NOT use the words "{t['operation_id']}" or "{t['api']}".
- Refer to the object as "{entity_phrase}" (or a natural operator synonym of it). Sibling
  APIs manage similar-sounding objects, so naming a DIFFERENT domain's catalogue, inventory,
  order or ticket would point the request at the wrong system.
- Never write the word "API", the API's title, or any TMF number. Write it the way an
  engineer would say it out loud, not the way a specification labels it.
- Invent plausible concrete values for the required parameters and mention them in the sentence.
- 1 sentence, max 32 words.

Return JSON: {{"query": "<sentence>", "args": {{"<param>": "<value>"}}}}"""

DOMAIN_CUES = {
    "service catalog": ("TMF633", "TMF909"), "resource catalog": ("TMF634",),
    "product catalog": ("TMF620",), "entity catalog": ("TMF662",),
    "service inventory": ("TMF638",), "resource inventory": ("TMF639",),
    "product inventory": ("TMF637",), "service order": ("TMF641",),
    "product order": ("TMF622",), "service test": ("TMF653",),
    "trouble ticket": ("TMF621",), "shopping cart": ("TMF663",),
    "billing account": ("TMF666",),
}

def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())

def bad_cue(q, t):
    """Reject two opposite leaks.

    (a) the utterance names a domain owned by a DIFFERENT API (points elsewhere);
    (b) it names its OWN API/title/TMF code, which hands over the answer and would
        erase the sibling ambiguity the study is about.
    """
    low = q.lower()
    title = t["api_title"].lower()
    for cue, apis in DOMAIN_CUES.items():
        if cue in low and t["api"] not in apis and cue not in title:
            return True
    ql = f" {_norm(q)} "
    if " api " in ql or re.search(r"\btmf\s*\d", low):
        return True
    stop = {"management", "api", "and", "the", "of", "for"}
    toks = [w for w in _norm(t["api_title"]).split() if w not in stop]
    for i in range(len(toks) - 1):
        if f"{toks[i]} {toks[i+1]}" in ql:       # 2+ consecutive title words verbatim
            return True
    return False

def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return None

out, t0, bad = [], time.time(), 0
for i, t in enumerate(sample):
    try:
        raw = ollama([{"role": "system", "content": SYS},
                      {"role": "user", "content": prompt_for(t)}])
    except Exception as e:
        print("err", t["tool"], e, flush=True); bad += 1; continue
    j = parse_json(raw)
    if not j or not j.get("query"):
        bad += 1; continue
    q = re.sub(r"\s+", " ", str(j["query"])).strip()
    # reject leakage of the operationId
    if t["operation_id"].lower() in q.lower().replace(" ", ""):
        bad += 1; continue
    if len(q.split()) < 5:
        bad += 1; continue
    if bad_cue(q, t):
        bad += 1; continue
    out.append({"qid": f"q{len(out):04d}", "query": q,
                "gold_tool": t["tool"], "gold_api": t["api"],
                "gold_method": t["method"], "severity": t["severity"],
                "confusable": t["confusable"], "n_confusers": t["n_confusers"],
                "gold_args": j.get("args") if isinstance(j.get("args"), dict) else {}})
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"{i+1}/{len(sample)} ok={len(out)} bad={bad} {el:.0f}s "
              f"eta={el/(i+1)*(len(sample)-i-1):.0f}s", flush=True)

json.dump(out, open(f"{D}/queries.json", "w"), indent=1)
print("WROTE", len(out), "queries; bad:", bad, flush=True)
