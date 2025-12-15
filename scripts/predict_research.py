import sys
import os
import argparse
import re
import yaml
import torch

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
from fire_debate.agents.local_client import LocalHFClient
# [UPDATE] Class is now 'Debater', not 'DebaterAgent'
from fire_debate.agents.debater import Debater, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import Retriever as EvidenceRetriever
from fire_debate.debate.manager import DebateManager
# [UPDATE] Import the High-Level Wrapper
from fire_debate.insight.hgt_judge import HGTJudge

def predict(user_claim, ground_truth=None):
    print("\n" + "="*60)
    print(f"🕵️ FIRE-DEBATE LIVE INVESTIGATION")
    print(f"🔎 Claim: '{user_claim}'")
    if ground_truth is not None:
        truth_str = "TRUE (Real)" if ground_truth else "FALSE (Fake)"
        print(f"📝 Expected Result: {truth_str}")
    print("="*60 + "\n")

    # --- 1. ENGINE SETUP ---
    config_path = os.path.join(project_root, "configs", "base.yaml")
    
    # Load Config or use safe defaults
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        print("⚠️ Config not found. Using defaults.")
        cfg = {}

    print("⚙️  Loading Engines...")
    try:
        llm = LocalHFClient(cfg)
        retriever = EvidenceRetriever(cfg)
        librarian = Librarian()
        
        # Initialize Debaters (Alice & Bob)
        # We pass the shared LLM and Retriever to save VRAM
        alice = Debater(AgentConfig("Alice", "PRO"), llm=llm, retriever=retriever, librarian=librarian)
        bob = Debater(AgentConfig("Bob", "CON"), llm=llm, retriever=retriever, librarian=librarian)
        
        manager = DebateManager(alice, bob, retriever)
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # --- 2. DEBATE PHASE ---
    # The Manager orchestrates the search, argument generation, and turn logging
    log = manager.run_debate(user_claim, rounds=2)
    
    # --- 3. JUDGEMENT PHASE (Neuro-Symbolic) ---
    print("\n⚖️  Consulting the HGT Judge (Interactive Attention)...")
    
    # Path to trained weights (if they exist)
    weights_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    
    # Initialize the Judge Wrapper
    # It automatically builds the Heterogeneous Graph + Runs the GNN
    judge = HGTJudge(model_path=weights_path if os.path.exists(weights_path) else None)
    
    # Run the Verdict
    result = judge.judge(log)
    
    # --- 4. RESULT DISPLAY ---
    verdict_str = "TRUE (Real)" if result['verdict'] else "FALSE (Fake)"
    
    # Extract "THE TRUTH" from the textual summary if available
    correction = "Details in debate summary."
    if log.summary:
        match = re.search(r"THE TRUTH:(.*)", log.summary, re.IGNORECASE)
        if match:
            correction = match.group(1).strip()

    print("\n" + "#"*60)
    print(f"🔥 FINAL VERDICT: {verdict_str}")
    print(f"📊 Confidence Score: {result['confidence']:.2%}")
    print(f"🧠 Reasoning: {result['reason']}")
    print("-" * 60)
    print(f"✅ FACTUAL CORRECTION: {correction}")
    print("#"*60)
    
    if ground_truth is not None:
        print(f"\nModel Prediction: {result['verdict']} | Correct Answer: {ground_truth}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("claim", nargs="?", type=str, help="Claim text")
    parser.add_argument("--true", action="store_true", help="Set expected result to True")
    parser.add_argument("--false", action="store_true", help="Set expected result to False")
    
    args = parser.parse_args()
    gt = True if args.true else (False if args.false else None)
    
    if args.claim:
        predict(args.claim, gt)
    else:
        print("\nEnter a claim to check:")
        user_input = input("> ").strip()
        if user_input: predict(user_input, gt)