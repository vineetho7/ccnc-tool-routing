#!/bin/bash
set -x
cd /Users/vineeth/Documents/ccnc-tool-routing
P=.venv/bin/python
until grep -q "### MASTER DONE" logs/master.log; do sleep 60; done

echo "### M2 extra-dense-points $(date +%H:%M:%S)"
for n in 25 100 400 800; do
  echo "=== dense5 $n $(date +%H:%M:%S)"
  $P src/eval_select.py dense5 $n 2>&1 | tail -2
done

echo "### M2 lora-long $(date +%H:%M:%S)"
rm -rf adapters_long
$P -m mlx_lm lora --model mlx-community/Qwen3-1.7B-bf16 --train --data data/lora \
  --iters 1500 --batch-size 4 --num-layers 24 --learning-rate 1e-4 --max-seq-length 160 \
  --steps-per-report 100 --steps-per-eval 750 --save-every 250 --adapter-path adapters_long \
  > logs/lora_long.log 2>&1
echo "### M2 router-long $(date +%H:%M:%S)"
$P src/eval_router.py adapters_long router_tunedlong > logs/router_tunedlong.log 2>&1
grep -E "^n=|^mean" logs/router_tunedlong.log

echo "### M2 hybrid-long $(date +%H:%M:%S)"
cp results/router_tuned.json results/router_tuned300.json
$P -c "
import json
d=json.load(open('results/sel_hybrid_n1062_hybrid.json'))
for r in d: r['method']='hybrid300'
json.dump(d, open('results/sel_hybrid300_n1062.json','w'))
print('preserved short hybrid as hybrid300')
"
cp results/router_tunedlong.json results/router_tuned.json
$P src/eval_hybrid.py 1062 > logs/hybrid_long.log 2>&1
tail -3 logs/hybrid_long.log

echo "### M2 aggregate $(date +%H:%M:%S)"
$P src/aggregate.py > logs/aggregate2.log 2>&1
tail -45 logs/aggregate2.log
echo "### M2 DONE $(date +%H:%M:%S)"
