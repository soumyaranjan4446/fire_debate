fire-debate/
  ├─ fire_debate/
  │    ├─ __init__.py
  │    ├─ config.py
  │    ├─ retriever.py          # SAFE / RAG (Tavily/KB/Chroma)
  │    ├─ agents.py             # Proponent, Opponent, Judge wrappers
  │    ├─ fallacy.py            # DeBERTa (or rule-based) fallacy detection
  │    ├─ graph.py              # GraphManager + HGT / NetworkX
  │    ├─ orchestrator.py       # LangGraph or custom debate loop
  │    ├─ evaluation.py         # metrics, logging
  │    └─ utils.py
  ├─ experiments/
  │    ├─ run_experiment.py     # CLI for running on a dataset
  │    ├─ configs/              # YAML configs for different setups
  ├─ tests/                     # unit tests for each module
  ├─ notebooks/                 # for exploration / visualizations
  ├─ data/                      # (or path in config) for datasets
  ├─ requirements.txt / pyproject.toml
  └─ README.md
