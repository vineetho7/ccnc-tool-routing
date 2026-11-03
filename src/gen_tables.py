"""Emit the LaTeX results table from the re-scored results."""
import json, sys
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import R
P = "/Users/vineeth/Documents/ccnc-tool-routing/paper"

T = json.load(open(f"{R}/table_main.json"))      # accuracy + conf/dist split + tokens
S = json.load(open(f"{R}/rescored.json"))        # corrected severity + resolution
NAME = {"full": "Full schemas", "dense5": "Dense top-5", "bm25_5": "BM25 top-5",
        "dense5_rename": "\; + PA-Tool rename", "direct": "No schemas (8B)",
        "router_base": "Router 1.7B, untuned", "router_tuned300": "Router 1.7B, LoRA-300",
        "router_tuned": "Router 1.7B, LoRA-1500",
        "hybrid": "\\textbf{Hybrid} (router $\\cup$ dense-5)"}
ROWS = [("full", 200), ("bm25_5", 1062), ("dense5", 1062), ("dense5_rename", 1062),
        ("direct", 1062), ("router_base", "any"), ("router_tuned", "any"),
        ("hybrid", 1062)]

def acc_row(m, n):
    key_n = 1062 if n == "any" else n
    return next((r for r in T if r["method"] == m and r["n"] == key_n), None)

def sev_row(m, n):
    return next((r for r in S if r["method"] == m and str(r["n"]) == str(n)), None)

with open(f"{P}/tab_main.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\caption{Tool selection. C/D = confusable/distinct "
            "accuracy; res = outputs naming a real tool; $c_{\\mathrm{sev}}$ = mean "
            "severity-weighted error cost over all predictions, charging an unresolvable "
            "name by the operation class it requests; dcall = queries ending in an "
            "unrequested \\texttt{DELETE}.}\n"
            "\\label{tab:main}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3.2pt}\n"
            "\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
            "Method & $N$ & Acc & C & D & res & $c_{\\mathrm{sev}}$ & dcall \\\\\n\\midrule\n")
    for m, n in ROWS:
        a, s = acc_row(m, n), sev_row(m, n)
        if a is None or s is None:
            continue
        cs = f"{s['sev_cost']:.2f}" if s["sev_cost"] is not None else "--"
        nn = "--" if n == "any" else str(n)
        f.write(f"{NAME.get(m, m)} & {nn} & {a['acc']:.1f} & {a['acc_conf']:.1f} & "
                f"{a['acc_dist']:.1f} & {s['resolved_pct']:.0f} & {cs} & "
                f"{s['dest_call_pct']:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
print(open(f"{P}/tab_main.tex").read())
