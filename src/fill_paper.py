"""Substitute catalogue/stat placeholders into the paper sources."""
import json, re, sys, collections
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import D, R
P = "/Users/vineeth/Documents/ccnc-tool-routing/paper"

cat = json.load(open(f"{D}/catalog.json"))
raw = json.load(open(f"{D}/catalog_full.json"))
qs = json.load(open(f"{D}/queries_eval.json"))
sev = collections.Counter(t["severity"] for t in cat)
train_n = sum(1 for _ in open(f"{D}/lora/train.jsonl"))
conf = sum(t["confusable"] for t in cat)
sub = {
    "RAW_TOOLS": f"{len(raw):,}", "CAT_TOOLS": f"{len(cat):,}",
    "CAT_APIS": str(len({t['api'] for t in cat})),
    "CAT_READ": str(sev[0]), "CAT_WRITE": str(sev[2]), "CAT_DEL": str(sev[3]),
    "CAT_CONF": str(conf), "CAT_CONFPCT": f"{100*conf/len(cat):.1f}",
    "TRAIN_N": f"{train_n:,}",
}
# argument-stage numbers, computed from the scored file so the text cannot drift
import numpy as _np
_a = json.load(open(f"{R}/args_qwen3.5-latest.json"))
_f = lambda k: 100*float(_np.mean([o[k] for o in _a]))
# N reported in the paper is the row count of the headline result file, so the
# text can never claim a dataset size the experiments did not actually use.
_sel = json.load(open(f"{R}/sel_dense5_n1062_qwen3.5-latest.json"))
sub["NQ"] = str(len(_sel))
if len(_sel) != len(qs):
    print(f"WARNING: results have {len(_sel)} queries but data/queries_eval.json has "
          f"{len(qs)} -- paper will report {len(_sel)} (the evaluated set)")
sub.update({
    "ARGP": f"{_f('key_p'):.1f}", "ARGR": f"{_f('key_r'):.1f}",
    "ARGEM": f"{_f('value_em'):.1f}", "ARGEX": f"{_f('exact_all'):.1f}",
    "ARGN": str(len(_a)),
    "ARGCOMP": f"{54.2*_f('exact_all')/100:.0f}",
})
import glob, os
for f in glob.glob(f"{P}/sec_*.tex") + glob.glob(f"{P}/*CCNC*.tex"):
    s = open(f).read()
    for k, v in sub.items():
        esc = k.replace("_", r"\\_")          # LaTeX-escaped form, e.g. CAT\_TOOLS
        s = re.sub(rf"(?<![A-Za-z_]){esc}(?![A-Za-z_])", v, s)
        s = re.sub(rf"(?<![A-Za-z_]){k}(?![A-Za-z_])", v, s)
    open(f, "w").write(s)
print("filled:", sub)
