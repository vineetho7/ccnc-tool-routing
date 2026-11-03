#!/bin/bash
# Once full_n200 result exists, stop the remaining (redundant) phase2 runs so the
# pipeline advances to argument eval + aggregation.
cd /Users/vineeth/Documents/ccnc-tool-routing
until grep -q "DONE full_n200" logs/phase2.log; do sleep 60; done
sleep 5
pkill -f "eval_select.py full_rename" 2>/dev/null
pkill -f "run_phase2.sh" 2>/dev/null
pkill -f "eval_select.py full_abst" 2>/dev/null
echo "trimmed remaining phase2 runs $(date +%H:%M:%S)"
