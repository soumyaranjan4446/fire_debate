fire-debate/
├── configs/
│   └── base.yaml              # (Needs your Tavily API Key)
├── data/
│   ├── raw/                   # (MUST contain train.json, test.json, val.json)
│   ├── processed/             # (Will contain train_set/ and test_set/ folders)
│   └── chroma_db/             # (Auto-generated database)
├── fire_debate/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── debater.py         # (The Qwen Agent with Agentic Search)
│   │   ├── librarian.py       # (The Filter)
│   │   ├── local_client.py    # (The LLM Wrapper)
│   │   ├── moderator.py       # (The Referee)
│   │   ├── sophist.py         # (The Liar)
│   │   └── synthesizer.py     # (The Summarizer)
│   ├── debate/
│   │   ├── __init__.py
│   │   └── manager.py         # (The Loop Orchestrator)
│   ├── insight/
│   │   ├── __init__.py
│   │   ├── fallacy.py         # (DeBERTa detector)
│   │   ├── graph_builder.py   # (Neuro-Symbolic Graph creator)
│   │   └── hgt_judge.py       # (The Graph Neural Network)
│   ├── rag/
│   │   ├── __init__.py
│   │   └── retriever.py       # (Tavily + Cross-Encoder)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── debate.py          # (CRITICAL: Must have 'search_query' field)
│   │   └── evidence.py
│   └── training/
│       ├── __init__.py
│       └── train_judge.py     # (Training script)
├── scripts/
│   ├── evaluate_model.py      # (Test script)
│   ├── generate_data.py       # (Data factory)
│   └── run_baseline.py        # (Zero-shot comparison)
└── requirements.txt