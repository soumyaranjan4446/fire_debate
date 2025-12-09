import os
from pathlib import Path

structure = [
    "configs",
    "data/raw",
    "data/processed",
    "fire_debate/schemas",
    "fire_debate/agents",
    "fire_debate/rag",
    "fire_debate/debate",
    "fire_debate/insight",
    "fire_debate/training",
    "notebooks",
    "scripts",
]

files = [
    "configs/base.yaml",
    "fire_debate/__init__.py",
    "fire_debate/schemas/__init__.py",
    "fire_debate/schemas/debate.py",
    "fire_debate/schemas/evidence.py",
    "fire_debate/agents/__init__.py",
    "fire_debate/agents/base.py",
    "fire_debate/agents/debater.py",
    "fire_debate/agents/librarian.py",
    "fire_debate/rag/__init__.py",
    "fire_debate/rag/retriever.py",
    "README.md",
]

for folder in structure:
    os.makedirs(folder, exist_ok=True)

for file in files:
    Path(file).touch()

print("✅ FIRE-Debate Research Structure Created Successfully.")