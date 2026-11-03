"""Aggregate all result files into tables + figures."""
import json, glob, os, sys, numpy as np, collections
sys.path.insert(0, "/Users/vineeth/Documents/ccnc-tool-routing/src")
from common import R, D
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42   # IEEE/EDAS reject Type 3 fonts
matplotlib.rcParams["ps.fonttype"] = 42

FIG = "/Users/vineeth/Documents/ccnc-tool-routing/figs"
os.makedirs(FIG, exist_ok=True)
pct = lambda a, k: 100*float(np.mean([x[k] for x in a])) if a else float("nan")

# ---------------- selection results ----------------
sel = []
for f in glob.glob(f"{R}/sel_*.json"):
    sel += json.load(open(f))
rows = {}
for o in sel:
    key = (o["method"], o["n"])
    rows.setdefault(key, []).append(o)

table = []
for (m, n), a in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1])):
    c = [x for x in a if x["confusable"]]; d = [x for x in a if not x["confusable"]]
    table.append({"method": m, "n": n, "N": len(a),
                  "acc": pct(a, "correct"), "acc_conf": pct(c, "correct"), "n_conf": len(c),
                  "acc_dist": pct(d, "correct"), "n_dist": len(d),
                  "recall_cand": pct(a, "in_candidates"),
                  "sev_cost": float(np.mean([x["sev_cost"] for x in a])),
                  "dest_mis": pct(a, "destructive_misfire"),
                  "p_tok": float(np.mean([x["p_tok"] for x in a])),
                  "g_tok": float(np.mean([x["g_tok"] for x in a])),
                  "wall": float(np.mean([x["wall"] for x in a]))})

# ---------------- router results ----------------
SIZES = [25, 50, 100, 200, 400, 800, 1062]
for f in glob.glob(f"{R}/router_*.json"):
    data = json.load(open(f))
    m = os.path.basename(f)[:-5]
    for n in SIZES:
        a = [o[f"n{n}"] for o in data]
        c = [x for x in a if x["confusable"]]; d = [x for x in a if not x["confusable"]]
        table.append({"method": m, "n": n, "N": len(a),
                      "acc": pct(a, "correct"), "acc_conf": pct(c, "correct"), "n_conf": len(c),
                      "acc_dist": pct(d, "correct"), "n_dist": len(d),
                      "recall_cand": pct(a, "pred_in_catalog"),
                      "sev_cost": float(np.mean([x["sev_cost"] for x in a])),
                      "dest_mis": pct(a, "destructive_misfire"),
                      "p_tok": float(np.mean([o["p_tok"] for o in data])),
                      "g_tok": float(np.mean([o["g_tok"] for o in data])),
                      "wall": float(np.mean([o["wall"] for o in data]))})

json.dump(table, open(f"{R}/table_main.json", "w"), indent=1)
hdr = f"{'method':22s} {'n':>5s} {'N':>4s} {'acc':>6s} {'conf':>6s} {'dist':>6s} {'rec':>6s} {'sev':>5s} {'dmis':>5s} {'ptok':>7s} {'s/q':>6s}"
print(hdr); print("-"*len(hdr))
for r in table:
    print(f"{r['method']:22s} {r['n']:5d} {r['N']:4d} {r['acc']:6.1f} {r['acc_conf']:6.1f} "
          f"{r['acc_dist']:6.1f} {r['recall_cand']:6.1f} {r['sev_cost']:5.2f} {r['dest_mis']:5.1f} "
          f"{r['p_tok']:7.0f} {r['wall']:6.2f}")

# ---------------- retrieval curve ----------------
rc_file = f"{R}/retrieval_curve.json"
if os.path.exists(rc_file):
    rc = json.load(open(rc_file))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for ri, ret in enumerate(["dense", "bm25"]):
        for grp, lbl, st in ((True, "confusable", "-o"), (False, "distinct", "--s")):
            xs, ys5, ys1 = [], [], []
            for n in SIZES:
                a = [x for x in rc if x["n"] == n and x["retriever"] == ret and x["confusable"] == grp]
                if not a: continue
                xs.append(n); ys5.append(pct(a, "hit@5")); ys1.append(pct(a, "hit@1"))
            ax[ri].plot(xs, ys5, st, label=f"{lbl} hit@5", ms=4)
            ax[ri].plot(xs, ys1, st, alpha=.45, label=f"{lbl} hit@1", ms=4)
        ax[ri].set_xscale("log"); ax[ri].set_xlabel("catalog size (tools)")
        ax[ri].set_title(ret.upper()); ax[ri].grid(alpha=.3); ax[ri].set_ylim(0, 102)
    ax[0].set_ylabel("recall (%)"); ax[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{FIG}/retrieval_curve.pdf"); plt.savefig(f"{FIG}/retrieval_curve.png", dpi=160)
    print("fig: retrieval_curve")

# ---------------- main accuracy figure ----------------
def series(m):
    a = [r for r in table if r["method"] == m]
    a.sort(key=lambda r: r["n"])
    return [r["n"] for r in a], [r["acc"] for r in a], [r["acc_conf"] for r in a], [r["acc_dist"] for r in a]

plt.figure(figsize=(5, 3.4))
style = {"full": ("-o", "tab:red"), "dense5": ("-s", "tab:blue"),
         "bm25_5": ("-^", "tab:green"), "dense5_rename": ("-v", "tab:purple"),
         "router_tuned": ("-D", "tab:orange"), "router_base": ("--x", "gray"),
         "direct": (":+", "black"), "hybrid": ("-*", "tab:brown"),
         }
for m, (st, c) in style.items():
    x, y, *_ = series(m)
    if x: plt.plot(x, y, st, color=c, label=m, ms=4)
plt.xscale("log"); plt.xlabel("catalog size (tools)"); plt.ylabel("top-1 tool accuracy (%)")
plt.grid(alpha=.3); plt.legend(fontsize=6); plt.tight_layout()
plt.savefig(f"{FIG}/accuracy_vs_size.pdf"); plt.savefig(f"{FIG}/accuracy_vs_size.png", dpi=160)

# ---------------- confusable vs distinct bars ----------------
ms = [m for m in ["full", "dense5", "bm25_5", "dense5_rename", "router_base", "router_tuned"]
      if any(r["method"] == m for r in table)]
NBIG = 1062
plt.figure(figsize=(5.4, 3.2))
w = 0.38
for i, m in enumerate(ms):
    cand = [r for r in table if r["method"] == m]
    r = max(cand, key=lambda z: z["n"])
    plt.bar(i - w/2, r["acc_dist"], w, color="tab:blue", label="distinct" if i == 0 else None)
    plt.bar(i + w/2, r["acc_conf"], w, color="tab:red", label="confusable" if i == 0 else None)
    plt.text(i, 2, f"n={r['n']}", ha="center", fontsize=6)
plt.xticks(range(len(ms)), ms, rotation=20, fontsize=7)
plt.ylabel("top-1 accuracy (%)"); plt.legend(fontsize=7); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG}/conf_vs_dist.pdf"); plt.savefig(f"{FIG}/conf_vs_dist.png", dpi=160)
print("figs written to", FIG)

# ---------------- controlled confusability contrast ----------------
cc_file = f"{R}/controlled_conf.json"
if os.path.exists(cc_file):
    cc = json.load(open(cc_file))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    labels, w = [], 0.35
    for i, ret in enumerate(["dense", "bm25"]):
        for j, k in enumerate(["hit@1", "hit@5"]):
            a = [r[k] for r in cc if r["retriever"] == ret and r["cond"] == "absent"]
            b = [r[k] for r in cc if r["retriever"] == ret and r["cond"] == "present"]
            d = np.array(b) - np.array(a)
            ci = 100*1.96*d.std(ddof=1)/np.sqrt(len(d))
            x = i*2 + j
            ax.bar(x - w/2, 100*np.mean(a), w, color="tab:blue")
            ax.bar(x + w/2, 100*np.mean(b), w, color="tab:red",
                   yerr=ci, capsize=2, ecolor="k")
            labels.append(f"{ret}\n{k}")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("recall (%)")
    ax.bar(0, 0, color="tab:blue", label="confusers absent")
    ax.bar(0, 0, color="tab:red", label="confusers injected")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/controlled.pdf"); plt.savefig(f"{FIG}/controlled.png", dpi=160)
    print("fig: controlled")

# ---------------- main figure: confusable vs distinct accuracy ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.3))
x, y, yc, yd = series("dense5")
ax.plot(x, yd, "-s", color="tab:blue", ms=4, label="dense top-5, distinct")
ax.plot(x, yc, "-o", color="tab:red", ms=4, label="dense top-5, confusable")
xf, yf, ycf, ydf = series("full")
ax.plot(xf, ydf, "--s", color="navy", ms=3, alpha=.6, label="full schemas, distinct")
ax.plot(xf, ycf, "--o", color="darkred", ms=3, alpha=.6, label="full schemas, confusable")
xh, yh, ych, ydh = series("hybrid")
if xh:
    ax.plot(xh, ych, "*", color="tab:brown", ms=10, label="hybrid, confusable")
xr, yr, ycr, ydr = series("router_tuned")
if xr:
    ax.axhline(ycr[-1], ls=":", color="gray", lw=1)
    ax.text(28, ycr[-1] + 2, "router alone (confusable)", fontsize=5, color="gray")
ax.set_xscale("log"); ax.set_xlabel("catalogue size (tools)")
ax.set_ylabel("top-1 accuracy (%)"); ax.set_ylim(0, 102)
ax.grid(alpha=.3); ax.legend(fontsize=5.5, loc="lower left", framealpha=.9)
plt.tight_layout(pad=0.3)
plt.savefig(f"{FIG}/main_fig.pdf"); plt.savefig(f"{FIG}/main_fig.png", dpi=160)
print("fig: main_fig")
