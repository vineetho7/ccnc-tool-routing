"""Embed tool documents with nomic-embed-text (local, via ollama)."""
import json, sys, time, urllib.request, numpy as np

D = "/Users/vineeth/Documents/ccnc-tool-routing/data"
EM = "nomic-embed-text:latest"

def embed(texts, prefix):
    out = []
    for i, t in enumerate(texts):
        body = json.dumps({"model": EM, "prompt": prefix + t}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embeddings", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as f:
            out.append(json.load(f)["embedding"])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(texts)}", flush=True)
    a = np.array(out, dtype=np.float32)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

def tool_doc(t):
    ps = ", ".join(str(p.get("name") or "?") for p in t["params"][:8])
    return (f"{t['tool']} ({t['method']} {t['path']}). API: {t['api_title']}. "
            f"{t['summary']} {t['description'][:200]} Parameters: {ps}")

if __name__ == "__main__":
    cat = json.load(open(f"{D}/catalog.json"))
    docs = [tool_doc(t) for t in cat]
    json.dump(docs, open(f"{D}/tool_docs.json", "w"))
    t0 = time.time()
    E = embed(docs, "search_document: ")
    np.save(f"{D}/tool_emb.npy", E)
    print("tools embedded", E.shape, f"{time.time()-t0:.0f}s", flush=True)
