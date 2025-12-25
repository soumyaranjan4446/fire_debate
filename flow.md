fire_debate/
├── configs/
│   ├── base.yaml         # Main config: API keys, LLM, retrieval, debate settings
│   └── .env              # (Sensitive keys, not for git)
├── data/
│   ├── raw/              # Raw datasets (train.json, test.json, etc.)
│   ├── processed/        # Processed debate logs, graphs, etc.
│   └── chroma_db/        # ChromaDB vector store (auto-generated)
├── fire_debate/
│   ├── __init__.py           # Dataset loader, graph builder utility
│   ├── agents/
│   │   ├── __init__.py       # (empty)
│   │   ├── base.py           # Abstract LLM client interface
│   │   ├── debater.py        # Rational debate agent logic
│   │   ├── librarian.py      # Filters and manages evidence
│   │   ├── local_client.py   # Local LLM wrapper (Qwen, Llama, etc.)
│   │   ├── moderator.py      # Monitors and moderates debate
│   │   ├── openai_client.py  # OpenAI LLM wrapper
│   │   ├── sophist.py        # Manipulative/fallacious agent
│   │   └── synthesizer.py    # Summarizes debate
│   ├── debate/
│   │   ├── __init__.py       # (empty)
│   │   └── manager.py        # Orchestrates debate rounds, agent turns
│   ├── insight/
│   │   ├── __init__.py       # (empty)
│   │   ├── fallacy.py        # Fallacy/logic/relevance detection
│   │   ├── graph_builder.py  # Builds argument graphs from logs
│   │   └── hgt_judge.py      # Heterogeneous GNN judge model
│   ├── rag/
│   │   ├── __init__.py       # (empty)
│   │   └── retriever.py      # Evidence retrieval (Tavily, BM25, BGE)
│   ├── schemas/
│   │   ├── __init__.py       # (empty)
│   │   ├── debate.py         # DebateTurn, DebateLog dataclasses
│   │   └── evidence.py       # EvidenceDoc dataclass
│   └── training/
│       ├── __init__.py       # (empty)
│       ├── test_prediction.py# Test judge prediction on mock debate
│       ├── train_full.py     # Full judge training pipeline
│       └── train_judge.py    # Main judge training script
├── scripts/
│   ├── analyze_debate.py     # Analyze debate logs, build graphs
│   ├── diagnose_system.py    # System diagnostics
│   ├── evaluate_model.py     # Evaluate judge model
│   ├── final_check.py        # System integrity check
│   ├── generate_data.py      # Generate debate data
│   ├── generate_openai_data.py # Generate data using OpenAI
│   ├── inspect_data.py       # Inspect and sample data
│   ├── predict_claim.py      # Predict claim factuality (main pipeline)
│   ├── predict_research.py   # Research pipeline (variant)
│   ├── run_baseline.py       # Zero-shot LLM baseline
│   ├── run_baseline_robust.py# Robust baseline with OpenAI
│   ├── run_debate.py         # Run a single debate (main entry)
│   ├── run_investigation.py  # CLI for claim investigation
│   └── visualize_graph.py    # Visualize debate argument graph
├── experiments/
│   └── run_experiment.py     # Experiment runner
├── requirements.txt          # Python dependencies
├── README.md                 # Main documentation
├── flow.md                   # (This file)
└── ...

# File/Module Connectivity
# (How files interact, as comments)
# - scripts/run_debate.py: Loads config, instantiates agents, runs debate via DebateManager
# - DebateManager: Orchestrates pro/con agents, moderator, synthesizer
# - DebaterAgent: Uses EvidenceRetriever for search, Librarian for filtering
# - insight/graph_builder.py: Builds argument graphs from logs
# - insight/hgt_judge.py: Judges graphs for factuality
# - All data flows through schemas in schemas/
# - Training scripts use processed data and graphs for judge model
# - All configs and API keys are set in configs/base.yaml

# See README.md for pipeline diagrams and further details.