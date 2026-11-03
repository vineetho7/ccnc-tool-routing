#!/bin/bash
cd /Users/vineeth/Documents/ccnc-tool-routing
P=.venv/bin/python
run(){ echo "=== $* $(date +%H:%M:%S)"; $P src/eval_select.py "$@" 2>&1 | tail -3; }
run dense5 1062
run bm25_5 1062
run dense5_rename 1062
run dense5 200
run dense5 50
run direct 1062
run dense5_abst 1062
echo "=== hybrid $(date +%H:%M:%S)"; $P src/eval_hybrid.py 1062 2>&1 | tail -3
run full 25
run full 50
run full 200
run full_rename 200
run full_abst 50
echo "PHASE2 DONE $(date +%H:%M:%S)"
