import sys
import os
import argparse
import re # Added regex for extraction

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
import torch
from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTJudge

def predict(user_claim, ground_truth=None):
    print("\n" + "="*60)
    print(f"🕵️ FIRE-DEBATE LIVE INVESTIGATION")
    print(f"🔎 Claim: '{user_claim}'")
    if ground_truth is not None:
        truth_str = "TRUE (Real)" if ground_truth else "FALSE (Fake)"
        print(f"📝 Expected Result: {truth_str}")
    print("="*60 + "\n")

    # --- ENGINE ---
    config_path = os.path.join(project_root, "configs", "base.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config not found.")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("⚙️  Loading Engines...")
    llm = LocalHFClient(cfg)
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()
    
    alice = DebaterAgent(AgentConfig("Alice", "PRO"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON"), llm, retriever, librarian)
    manager = DebateManager(alice, bob, retriever)

    # --- PHASE 1: DEBATE ---
    # Safe truncate for search context, but agents see full claim
    log = manager.run_debate(user_claim, rounds=2)
    
    # --- PHASE 2: GRAPH ---
    print("\n🕸️  Constructing Neuro-Symbolic Graph...")
    builder = GraphBuilder(device="cpu")
    graph = builder.build_graph(log)
    
    # --- PHASE 3: JUDGEMENT ---
    print("⚖️  Consulting the HGT Judge...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    metadata = graph.metadata()
    model = HGTJudge(64, 1, 2, 2, metadata).to(device)
    
    weights_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except FileNotFoundError:
        print("❌ Error: Trained model not found.")
        return

    model.eval()
    with torch.no_grad():
        logits = model(graph.x_dict, graph.edge_index_dict)
        probability = logits.item()

    # --- RESULT EXTRACTION ---
    is_true = probability > 0.5
    verdict_str = "TRUE (Real)" if is_true else "FALSE (Fake)"
    confidence = probability if is_true else 1 - probability
    
    # Extract "THE TRUTH" from summary
    correction = "Details in summary below."
    if log.summary:
        match = re.search(r"THE TRUTH:(.*)", log.summary, re.IGNORECASE)
        if match:
            correction = match.group(1).strip()

    print("\n" + "#"*60)
    print(f"🔥 FINAL VERDICT: {verdict_str}")
    print(f"📊 Confidence Score: {confidence:.2%}")
    print("-" * 60)
    print(f"✅ FACTUAL CORRECTION: {correction}")
    print("#"*60)
    
    if ground_truth is not None:
        print(f"\nModel Prediction: {is_true} | Correct Answer: {ground_truth}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("claim", nargs="?", type=str, help="Claim text")
    parser.add_argument("--true", action="store_true")
    parser.add_argument("--false", action="store_true")
    
    args = parser.parse_args()
    gt = True if args.true else (False if args.false else None)
    
    if args.claim:
        predict(args.claim, gt)
    else:
        print("\nEnter a claim to check:")
        user_input = input("> ").strip()
        if user_input: predict(user_input, gt)