# Confusable Tools — CCNC WiP experiment

Tool-routing study on a real telecom management API catalog (TM Forum Open APIs).
Everything runs locally: ollama (qwen3.5, nomic-embed-text) + MLX (Qwen3-1.7B LoRA).

## Pipeline
```
src/harvest_tmf.py      # 79 OpenAPI specs from github.com/tmforum-apis  -> data/specs/
src/build_catalog.py    # flatten operations                              -> data/catalog_full.json
src/filter_label.py     # drop listener boilerplate, label confusability  -> data/catalog.json
src/gen_queries.py      # one operator utterance per sampled tool         -> data/queries.json
src/make_train.py       # template-only router training corpus            -> data/lora/*.jsonl
src/embed.py            # tool embeddings                                 -> data/tool_emb.npy
src/embed_queries.py    # query embeddings                                -> data/query_emb.npy
src/retrieval_curve.py  # LLM-free recall@k vs catalog size               -> results/retrieval_curve.json
src/eval_select.py      # METHOD N [MODEL] end-to-end selection           -> results/sel_*.json
src/eval_router.py      # description-free SLM router (base | adapters)   -> results/router_*.json
src/eval_args.py        # stage-2 argument generation                     -> results/args_*.json
src/aggregate.py        # tables + figures                                -> results/table_main.json, figs/
src/gen_tables.py       # LaTeX tables                                    -> paper/tab_*.tex
src/master.sh           # runs the whole thing
```

## Paper
`paper/confusable_telecom_apis.tex` (IEEEtran conference). Build: `cd paper && ./build.sh`.

## License

Code: MIT (see `LICENSE`). Our queries, labels and results: CC BY 4.0.
The TM Forum Open API specs under `data/specs/` are redistributed under their original Apache-2.0 license.
