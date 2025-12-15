import sys
import os
import argparse
import re
import yaml
import torch
import time

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up just ONE level from 'scripts/' to 'fire_debate2/'
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTModel 

def print_slow(text, delay=0.01):
    """Effect to make text appear as if it's being typed live."""
    if not text: return
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def predict(user_claim, ground_truth=None):
    print("\n" + "="*60)
    print(f"🕵️  FIRE-DEBATE LIVE INVESTIGATION")
    print(f"🔎 Claim: '{user_claim}'")
    if ground_truth is not None:
        truth_str = "TRUE (Real)" if ground_truth else "FALSE (Fake)"
        print(f"📝 Expected Result: {truth_str}")
    print("="*60 + "\n")

    # --- ENGINE ---
    config_path = os.path.join(project_root, "configs", "base.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("⚙️  Loading Engines (7B Quantized)...")
    llm = LocalHFClient(cfg)
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()
    
    alice = DebaterAgent(AgentConfig("Alice", "PRO"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON"), llm, retriever, librarian)
    manager = DebateManager(alice, bob, retriever)

    # --- PHASE 1: DEBATE ---
    print("\n🎤  STARTING LIVE DEBATE...")
    print("(Agents are searching the web and generating arguments...)\n")
    
    # Run the debate (Manager runs silently)
    log = manager.run_debate(user_claim, rounds=1)
    
    # --- VISUALIZE THE DEBATE ---
    for turn in log.turns:
        if turn.agent_name == "Moderator":
            icon = "👨‍⚖️ [MODERATOR]"
            color = "\033[93m" # Yellow
        elif turn.stance == "PRO":
            icon = "🔵 [ALICE - PRO]"
            color = "\033[94m" # Blue
        elif turn.stance == "CON":
            icon = "🔴 [BOB - CON]"
            color = "\033[91m" # Red
        else:
            icon = "⚪"
            color = "\033[0m"
        
        reset = "\033[0m"
        print(f"{color}{icon}{reset}")
        print_slow(f"   {turn.text.strip()}", delay=0.005)
        print("-" * 40)
        time.sleep(0.5)

    # --- PHASE 2: GRAPH ---
    print("\n🕸️  Constructing Neuro-Symbolic Graph...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    builder = GraphBuilder(device=str(device))
    graph = builder.build_graph(log)
    
    # --- PHASE 3: JUDGEMENT ---
    print("⚖️  Consulting the HGT Judge...")
    
    graph = graph.to(device)
    metadata = graph.metadata()
    
    # --- FIX: Detect Input Dimension ---
    # We must match the dimension used during training (likely 771)
    if not graph.x_dict:
        print("❌ Error: Generated graph has no features.")
        return

    sample_node_type = list(graph.x_dict.keys())[0]
    in_channels = graph.x_dict[sample_node_type].shape[1]
    # -----------------------------------

    # Load Model with correct in_channels
    model = HGTModel(
        in_channels=in_channels, # <--- ADDED THIS
        hidden_channels=64, 
        out_channels=1, 
        num_heads=2, 
        num_layers=2, 
        metadata=metadata
    ).to(device)
    
    weights_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    
    if not os.path.exists(weights_path):
        print(f"⚠️  Warning: Trained model not found at {weights_path}")
        print("    Using untrained model for demo purposes (Predictions will be random).")
    else:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception as e:
            print(f"⚠️  Model load failed: {e}")

    model.eval()
    with torch.no_grad():
        # Pass batch_dict=None explicitly for single inference
        logits = model(graph.x_dict, graph.edge_index_dict, batch_dict=None)
        probability = torch.sigmoid(logits).item()

    # --- RESULT EXTRACTION ---
    is_true = probability > 0.5
    verdict_str = "TRUE (Real)" if is_true else "FALSE (Fake)"
    confidence = probability if is_true else 1.0 - probability
    
    correction = "Details in summary above."
    
    if log.summary:
        clean_summary = log.summary.replace("**", "").replace("##", "")
        match = re.search(r"THE TRUTH:?\s*(.*)", clean_summary, re.IGNORECASE | re.DOTALL)
        if match:
            raw_correction = match.group(1).strip()
            correction = raw_correction.split('\n')[0] 
        else:
            paragraphs = clean_summary.split('\n\n')
            if paragraphs:
                correction = paragraphs[-1].strip()
    
    print("\n" + "#"*60)
    print(f"🔥 FINAL VERDICT: {verdict_str}")
    print(f"📊 Confidence Score: {confidence:.2%}")
    print("-" * 60)
    print(f"✅ FACTUAL CORRECTION:\n{correction}") 
    print("#"*60)
    
    if ground_truth is not None:
        print(f"\nModel Prediction: {is_true} | Correct Answer: {ground_truth}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("claim", nargs="?", type=str, help="Claim text")
    parser.add_argument("--true", action="store_true", help="Assert ground truth is True")
    parser.add_argument("--false", action="store_true", help="Assert ground truth is False")
    
    args = parser.parse_args()
    gt = True if args.true else (False if args.false else None)
    
    if args.claim:
        predict(args.claim, gt)
    else:
        print("\nEnter a claim to check:")
        user_input = input("> ").strip()
        if user_input: predict(user_input, gt)