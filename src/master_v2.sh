#!/bin/bash
set -x
cd /Users/vineeth/Documents/ccnc-tool-routing
P=.venv/bin/python
until [ -f data/queries.json ]; do sleep 20; done
sleep 3
cp data/queries.json data/queries_eval.json
$P src/sync_query_severity.py

echo "### V2 contamination-check $(date +%H:%M:%S)"
$P src/audit_contamination.py 2>&1 | head -4

echo "### V2 embed $(date +%H:%M:%S)"
$P src/embed_queries.py > logs/embed_q2.log 2>&1
echo "### V2 retrieval $(date +%H:%M:%S)"
$P src/retrieval_curve.py > logs/retrieval2.log 2>&1
tail -18 logs/retrieval2.log
$P src/controlled_conf.py > logs/controlled2.log 2>&1
tail -6 logs/controlled2.log

echo "### V2 routers $(date +%H:%M:%S)"
$P src/eval_router.py none router_base > logs/router_base2.log 2>&1
$P src/eval_router.py adapters_long router_tuned > logs/router_tuned2.log 2>&1
grep -E "^n= 1062|^mean" logs/router_tuned2.log
$P src/eval_router.py adapters router_tuned300 > logs/router_tuned300_2.log 2>&1

echo "### V2 selection $(date +%H:%M:%S)"
run(){ echo "=== $* $(date +%H:%M:%S)"; $P src/eval_select.py "$@" 2>&1 | tail -2; }
for n in 1062 800 400 200 100 50 25; do run dense5 $n; done
run bm25_5 1062
run dense5_rename 1062
run direct 1062
run dense5_abst 1062
echo "=== hybrid $(date +%H:%M:%S)"; $P src/eval_hybrid.py 1062 2>&1 | tail -2
echo "### V2 args $(date +%H:%M:%S)"
$P src/eval_args.py > logs/args_v2.log 2>&1; grep "^ARGS" logs/args_v2.log
run full 25
run full 50
run full 200

echo "### V2 analysis $(date +%H:%M:%S)"
$P src/rescore.py > logs/rescore2.log 2>&1; tail -22 logs/rescore2.log
$P src/error_anatomy.py > logs/anatomy2.log 2>&1
$P src/jaccard_sensitivity.py > logs/jaccard2.log 2>&1; tail -6 logs/jaccard2.log
$P src/aggregate.py > logs/aggregate_v2.log 2>&1; tail -30 logs/aggregate_v2.log
$P src/gen_tables.py > /dev/null 2>&1
$P src/fill_paper.py
echo "### V2 DONE $(date +%H:%M:%S)"
