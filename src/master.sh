#!/bin/bash
set -x
cd /Users/vineeth/Documents/ccnc-tool-routing
P=.venv/bin/python
L=logs

until [ -f data/queries.json ]; do sleep 20; done
sleep 3
cp data/queries.json data/queries_eval.json
pkill -f "mlx_lm lora"; sleep 3

echo "### STAGE embed $(date +%H:%M:%S)"
$P src/embed_queries.py > $L/embed_q.log 2>&1
echo "### STAGE retrieval $(date +%H:%M:%S)"
$P src/retrieval_curve.py > $L/retrieval.log 2>&1
tail -20 $L/retrieval.log

echo "### STAGE lora $(date +%H:%M:%S)"
rm -rf adapters
$P -m mlx_lm lora --model mlx-community/Qwen3-1.7B-bf16 --train --data data/lora \
  --iters 1200 --batch-size 8 --num-layers 20 --learning-rate 1e-4 --max-seq-length 192 \
  --steps-per-report 100 --steps-per-eval 600 --save-every 300 --adapter-path adapters \
  > $L/lora2.log 2>&1
echo "### lora done $(date +%H:%M:%S)"

echo "### STAGE router $(date +%H:%M:%S)"
$P src/eval_router.py none router_base > $L/router_base.log 2>&1
tail -10 $L/router_base.log
$P src/eval_router.py adapters router_tuned > $L/router_tuned.log 2>&1
tail -10 $L/router_tuned.log

echo "### STAGE phase2 $(date +%H:%M:%S)"
bash src/run_phase2.sh > $L/phase2.log 2>&1
tail -40 $L/phase2.log

echo "### STAGE args $(date +%H:%M:%S)"
$P src/eval_args.py > $L/args.log 2>&1
tail -3 $L/args.log

echo "### STAGE aggregate $(date +%H:%M:%S)"
$P src/aggregate.py > $L/aggregate.log 2>&1
tail -40 $L/aggregate.log
echo "### MASTER DONE $(date +%H:%M:%S)"
