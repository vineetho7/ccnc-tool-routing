"""Parse harvested TMF specs into a flat tool catalog (one tool per operation)."""
import json, os, re, glob, yaml, collections

D = "/Users/vineeth/Documents/ccnc-tool-routing/data"

SEV = {"get": 0, "head": 0, "options": 0, "post": 2, "put": 2, "patch": 2, "delete": 3}
# 0=read, 2=state-changing (POST/PUT/PATCH), 3=destructive (DELETE only)

def load(p):
    txt = open(p, encoding="utf-8", errors="replace").read()
    try:
        return json.loads(txt)
    except Exception:
        return yaml.safe_load(txt)

def clean(s):
    if not s:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:400]

def main():
    tools, skipped = [], 0
    for p in sorted(glob.glob(f"{D}/specs/*")):
        api = os.path.basename(p).split("__")[0]
        tmf = api.split("_")[0]
        try:
            spec = load(p)
        except Exception as e:
            print("parse fail", api, e); continue
        if not isinstance(spec, dict) or "paths" not in spec:
            skipped += 1; continue
        title = clean((spec.get("info") or {}).get("title") or api)
        for path, item in (spec["paths"] or {}).items():
            if not isinstance(item, dict):
                continue
            for method, op in item.items():
                if method.lower() not in SEV or not isinstance(op, dict):
                    continue
                opid = op.get("operationId") or ""
                if not opid:
                    continue
                desc = clean(op.get("summary")) or clean(op.get("description"))
                if not desc:
                    desc = f"{method.upper()} {path}"
                params = []
                for pr in (op.get("parameters") or []):
                    if not isinstance(pr, dict):
                        continue
                    params.append({"name": pr.get("name"), "in": pr.get("in"),
                                   "required": bool(pr.get("required")),
                                   "type": (pr.get("schema") or {}).get("type") or pr.get("type"),
                                   "desc": clean(pr.get("description"))[:120]})
                body_req = method.lower() in ("post", "put", "patch")
                tools.append({
                    "tool": f"{tmf}_{opid}",
                    "api": tmf, "api_title": title,
                    "operation_id": opid, "method": method.upper(), "path": path,
                    "description": clean(op.get("description")) or desc,
                    "summary": desc,
                    "params": params, "has_body": body_req,
                    "severity": SEV[method.lower()],
                })
    # dedup by tool name
    seen, uniq = set(), []
    for t in tools:
        if t["tool"] in seen:
            continue
        seen.add(t["tool"]); uniq.append(t)
    json.dump(uniq, open(f"{D}/catalog_full.json", "w"), indent=1)
    print("specs skipped(no paths):", skipped)
    print("tools:", len(uniq), "APIs:", len({t['api'] for t in uniq}))
    c = collections.Counter(t["method"] for t in uniq)
    print("methods:", dict(c))
    # most common operationId roots -> confusability signal
    roots = collections.Counter(re.sub(r"^(list|retrieve|create|patch|delete|update|get)", "", t["operation_id"]) for t in uniq)
    print("top entity roots:", roots.most_common(12))
main()
