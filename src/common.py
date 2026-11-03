import os
import json, re, time, urllib.request, numpy as np, random

D = "/Users/vineeth/Documents/ccnc-tool-routing/data"
R = os.environ.get("CCNC_RESULTS", "/Users/vineeth/Documents/ccnc-tool-routing/results")
BIG = "qwen3.5:latest"
SMALL = "qwen3:1.7b"
SEV_COST = {0: 1.0, 2: 3.0, 3: 10.0}

def load_catalog():
    cat = json.load(open(f"{D}/catalog.json"))
    return cat, {t["tool"]: t for t in cat}

def load_queries():
    return json.load(open(f"{D}/queries_eval.json"))

def ollama_chat(messages, model=BIG, num_predict=64, temperature=0.0, num_ctx=16384):
    body = json.dumps({"model": model, "messages": messages, "stream": False, "think": False,
                       "options": {"num_predict": num_predict, "temperature": temperature,
                                   "num_ctx": num_ctx, "top_p": 1.0}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as f:
        d = json.load(f)
    return {"text": d["message"]["content"], "p_tok": d.get("prompt_eval_count", 0),
            "g_tok": d.get("eval_count", 0), "wall": time.time() - t0}

def embed_query(q):
    body = json.dumps({"model": "nomic-embed-text:latest",
                       "prompt": "search_query: " + q}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/embeddings", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as f:
        v = np.array(json.load(f)["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

# ---------- per-query catalog subsets (gold + seeded distractors) ----------
def subset_indices(qid, gold_idx, n, n_all, base_seed=7):
    rng = random.Random(f"{base_seed}:{qid}")
    pool = list(range(n_all)); pool.remove(gold_idx)
    rng.shuffle(pool)
    idx = [gold_idx] + pool[: n - 1]
    rng.shuffle(idx)
    return idx

def schema_line(t, rename=False):
    name = canon_name(t) if rename else t["tool"]
    req = [str(p.get("name")) for p in t["params"] if p.get("required")][:4]
    s = f"- {name}: {t['summary'][:110]}"
    if req:
        s += f" [params: {', '.join(req)}]"
    return s

_CANON = None

def _domain_token(t):
    import re as _re
    ttl = _re.sub(r"\s*(API|Management)\s*", " ", t["api_title"])
    ws = [w.lower() for w in _re.findall(r"[A-Za-z]+", ttl)][:2]
    return "_".join(ws) if ws else t["api"].lower()

def _build_canon():
    """PA-Tool-style pretraining-aligned names.

    The standards code (TMFxxx) disambiguates sibling APIs, so dropping it collides
    (27.6% of tools). Cross-API collisions are resolved with a domain token from the
    API title; a residual 5 names still collide WITHIN one API, where that token is
    shared. Uniqueness is therefore not guaranteed -- which is itself the finding.
    """
    import json as _json, re as _re, collections as _c
    cat = _json.load(open(f"{D}/catalog.json"))
    base = {}
    for t in cat:
        s = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", t["operation_id"]).lower()
        s = _re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        parts = s.split("_")
        vm = {"retrieve": "get", "patch": "update"}
        if parts and parts[0] in vm:
            parts[0] = vm[parts[0]]
        base[t["tool"]] = "_".join(parts)
    counts = _c.Counter(base.values())
    out = {}
    for t in cat:
        b = base[t["tool"]]
        out[t["tool"]] = b if counts[b] == 1 else f"{_domain_token(t)}_{b}"
    # final safety: any residual duplicate keeps the standards code
    seen = _c.Counter(out.values())
    for t in cat:
        if seen[out[t["tool"]]] > 1:
            out[t["tool"]] = f"{t['api'].lower()}_{out[t['tool']]}"
    return out

def canon_name(t):
    global _CANON
    if _CANON is None:
        _CANON = _build_canon()
    return _CANON.get(t["tool"], t["tool"])

def canon_collision_stats():
    import collections as _c, json as _json
    cat = _json.load(open(f"{D}/catalog.json"))
    import re as _re
    base = _c.Counter()
    for t in cat:
        s = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", t["operation_id"]).lower()
        base[_re.sub(r"[^a-z0-9]+", "_", s).strip("_")] += 1
    dup = sum(v for k, v in base.items() if v > 1)
    return {"tools": len(cat), "colliding_tools": dup, "distinct_names": len(base)}
