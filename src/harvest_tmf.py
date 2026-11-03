"""Harvest real TM Forum Open API operations into a tool catalog."""
import json, os, re, subprocess, sys, time

OUT = "/Users/vineeth/Documents/ccnc-tool-routing/data"
GH = "/opt/homebrew/bin/gh"

def gh_api(path):
    r = subprocess.run([GH, "api", path], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)

def list_repos():
    repos, page = [], 1
    while True:
        d = gh_api(f"/orgs/tmforum-apis/repos?per_page=100&page={page}")
        if not d:
            break
        repos += [(r["name"], r["default_branch"]) for r in d]
        if len(d) < 100:
            break
        page += 1
    return repos

def spec_files(repo, branch):
    t = gh_api(f"/repos/tmforum-apis/{repo}/git/trees/{branch}?recursive=1")
    if not t or "tree" not in t:
        return []
    hits = []
    for n in t["tree"]:
        p = n["path"]
        low = p.lower()
        if n["type"] != "blob":
            continue
        if low.endswith((".json", ".yaml", ".yml")) and ("swagger" in low or "openapi" in low or "api" in low):
            if n.get("size", 0) > 2000:
                hits.append((p, n["size"]))
    # prefer the largest swagger/openapi file (the full spec)
    hits.sort(key=lambda x: (("swagger" in x[0].lower() or "openapi" in x[0].lower()), x[1]), reverse=True)
    return hits[:1]

def fetch_raw(repo, branch, path):
    import urllib.parse, urllib.request
    url = f"https://raw.githubusercontent.com/tmforum-apis/{repo}/{branch}/{urllib.parse.quote(path)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as f:
            return f.read().decode("utf-8", "replace")
    except Exception as e:
        print("  raw fail", repo, path, e)
        return None

def main():
    os.makedirs(f"{OUT}/specs", exist_ok=True)
    repos = list_repos()
    print(f"{len(repos)} repos")
    manifest = []
    for name, branch in repos:
        if not re.match(r"^TMF\d{3}", name):
            continue
        fs = spec_files(name, branch)
        if not fs:
            print("  no spec:", name); continue
        path, size = fs[0]
        txt = fetch_raw(name, branch, path)
        if not txt:
            continue
        dest = f"{OUT}/specs/{name}__{os.path.basename(path)}"
        open(dest, "w").write(txt)
        manifest.append({"repo": name, "branch": branch, "path": path, "size": size, "local": dest})
        print(f"  ok {name} {os.path.basename(path)} {size}B")
    json.dump(manifest, open(f"{OUT}/spec_manifest.json", "w"), indent=1)
    print("saved", len(manifest))

main()
