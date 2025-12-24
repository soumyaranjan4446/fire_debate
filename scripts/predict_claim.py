import sys
import os
import argparse
import re
import yaml
import torch
import time
from datetime import datetime

# --- ANSI Colors for Terminal Beautification ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'      # Alice
    CYAN = '\033[96m'      # Info
    GREEN = '\033[92m'     # Success
    YELLOW = '\033[93m'    # Moderator/Warning
    RED = '\033[91m'       # Bob/Fail
    ENDC = '\033[0m'       # Reset
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

current_dir = os.path.dirname(os.path.abspath(__file__))

# FIX: Go up only ONE level (scripts -> project_root)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Debug print to confirm (Optional)
print(f"🔧 Debug: Project Root set to: {project_root}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.openai_client import OpenAIClient  # <--- Added Import
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTModel 

# --- Visual Helpers ---
def print_slow(text, delay=0.005, color=Colors.ENDC):
    """Effect to make text appear as if it's being typed live."""
    if not text: return
    sys.stdout.write(color)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(Colors.ENDC + '\n')

def print_header(title):
    width = 70
    print(f"\n{Colors.BOLD}{Colors.HEADER}╔{'═'*width}╗{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}║{title.center(width)}║{Colors.ENDC}") 
    print(f"{Colors.BOLD}{Colors.HEADER}╚{'═'*width}╝{Colors.ENDC}\n")

def print_separator(char='-'):
    print(f"{Colors.CYAN}{char * 72}{Colors.ENDC}")

# --- Main Logic ---
def predict(user_claim, ground_truth=None):
    # 1. SETUP DISPLAY
    print_header("🔥 FIRE-DEBATE: TRUTH INVESTIGATION ENGINE 🔥")
    
    print(f"{Colors.BOLD}🔎 CLAIM ANALYSIS:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Statement:{Colors.ENDC} \"{user_claim}\"")
    
    if ground_truth is not None:
        truth_str = "TRUE (Real)" if ground_truth else "FALSE (Fake)"
        color = Colors.GREEN if ground_truth else Colors.RED
        print(f"   {Colors.CYAN}Ground Truth:{Colors.ENDC} {color}{truth_str}{Colors.ENDC}")
    print_separator('=')

    # 2. INITIALIZE ENGINES
    config_path = os.path.join(project_root, "configs", "base.yaml")
    if not os.path.exists(config_path):
        print(f"{Colors.RED}❌ CRITICAL: Config not found at {config_path}{Colors.ENDC}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"\n{Colors.BOLD}⚙️  SYSTEM INITIALIZATION:{Colors.ENDC}")
    
    # --- DYNAMIC CLIENT LOADING ---
    llm_type = cfg.get('llm', {}).get('type', 'local')
    model_id = cfg.get('llm', {}).get('model_id', 'Unknown')
    
    print(f"   [1/4] Loading LLM Engine ({llm_type.upper()}: {model_id})...")
    
    try:
        if llm_type == "openai":
            llm = OpenAIClient(cfg)
            print(f"         {Colors.GREEN}✔ Connected to OpenAI API{Colors.ENDC}")
        else:
            llm = LocalHFClient(cfg)
            print(f"         {Colors.GREEN}✔ Loaded Local Model (GPU){Colors.ENDC}")
    except Exception as e:
        print(f"         {Colors.RED}❌ Failed to load LLM: {e}{Colors.ENDC}")
        return
    # ------------------------------
    
    print(f"   [2/4] Waking Retriever & Librarian...    {Colors.GREEN}✔{Colors.ENDC}")
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()
    
    print(f"   [3/4] Spawning Agents...                 {Colors.GREEN}✔{Colors.ENDC}")
    alice = DebaterAgent(AgentConfig("Alice", "PRO"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON"), llm, retriever, librarian)
    
    print(f"   [4/4] Configuring Debate Manager...      {Colors.GREEN}✔{Colors.ENDC}")
    manager = DebateManager(alice, bob, retriever)
    
    print(f"\n{Colors.GREEN}✅ SYSTEM READY. INITIATING DEBATE SEQUENCE.{Colors.ENDC}")
    time.sleep(1)

    # 3. RUN DEBATE
    print_header("🎤  LIVE DEBATE TRANSCRIPT  🎤")
    print(f"{Colors.CYAN}(Agents are now searching the web, analyzing sources, and generating arguments...){Colors.ENDC}\n")
    
    # Run the debate (Manager runs silently)
    log = manager.run_debate(user_claim, rounds=1)

    # Save Log
    output_filename = "debug_debate_log.json"
    output_path = os.path.join(project_root, "data", "logs", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"{Colors.CYAN}💾 Saving transcript to: {output_path}{Colors.ENDC}\n")
    manager.save_log(log, output_path)

    # 4. VISUALIZE TURNS
    current_phase = ""
    
    for turn in log.turns:
        # Detect and print Phase Change
        phase_raw = getattr(turn, 'phase', "ARGUMENT")
        if phase_raw != current_phase:
            current_phase = phase_raw
            print(f"\n{Colors.BOLD}{Colors.HEADER}>>> PHASE: {current_phase}{Colors.ENDC}\n")

        # Assign Icons and Colors
        if turn.agent_name == "Moderator":
            icon = "👨‍⚖️ [MODERATOR]"
            color = Colors.YELLOW
            indent = "   "
        elif turn.stance == "PRO":
            icon = "🔵 [ALICE - PRO]"
            color = Colors.BLUE
            indent = ""
        elif turn.stance == "CON":
            icon = "🔴 [BOB - CON]"
            color = Colors.RED
            indent = "" 
        else:
            icon = "⚪ [SYSTEM]"
            color = Colors.ENDC
            indent = ""
        
        # Print Agent Header
        print(f"{indent}{color}{Colors.BOLD}{icon}{Colors.ENDC}")
        
        # Print Text
        clean_text = turn.text.strip()
        print_slow(f"{clean_text}", delay=0.005, color=color)
        
        # Print Citations if any
        if turn.citations:
             print(f"{indent}{Colors.CYAN}   📎 Citations: {len(turn.citations)} source(s){Colors.ENDC}")
        
        print(f"{Colors.CYAN}{'-' * 72}{Colors.ENDC}")
        time.sleep(0.5)

    # 5. GRAPH CONSTRUCTION
    print_header("🕸️  NEURO-SYMBOLIC GRAPH ANALYSIS  🕸️")
    print(f"{Colors.BOLD}>>> Building Logic Graph...{Colors.ENDC}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    builder = GraphBuilder(device=str(device))
    graph = builder.build_graph(log)
    
    print(f"   Nodes Created: {graph.num_nodes}")
    print(f"   Edges Created: {graph.num_edges}")

    # 6. JUDGEMENT
    print(f"\n{Colors.BOLD}>>> Consulting HGT Judge...{Colors.ENDC}")
    
    graph = graph.to(device)
    
    # Metadata Check
    if hasattr(graph, 'metadata'):
        metadata = graph.metadata()
    else:
        # Fallback for older PyG versions or empty graphs
        metadata = (list(graph.x_dict.keys()), list(graph.edge_index_dict.keys()))
    
    # Safety Check: Input Dimension
    if not graph.x_dict:
        print(f"{Colors.RED}❌ Error: Generated graph has no features.{Colors.ENDC}")
        return

    sample_node_type = list(graph.x_dict.keys())[0]
    in_channels = graph.x_dict[sample_node_type].shape[1]

    # Load Model
    model = HGTModel(
        in_channels=in_channels,
        hidden_channels=64, 
        out_channels=1, 
        num_heads=2, 
        num_layers=2, 
        metadata=metadata
    ).to(device)
    
    weights_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    
    if not os.path.exists(weights_path):
        print(f"{Colors.YELLOW}⚠️  Warning: Trained weights not found. Using initialized model (Random).{Colors.ENDC}")
    else:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"{Colors.GREEN}   Weights Loaded Successfully.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}⚠️  Model load failed: {e}{Colors.ENDC}")

    model.eval()
    with torch.no_grad():
        logits = model(graph.x_dict, graph.edge_index_dict, batch_dict=None)
        probability = torch.sigmoid(logits).item()

    # 7. FINAL VERDICT DISPLAY
    is_true = probability > 0.5
    verdict_str = "TRUE (REAL)" if is_true else "FALSE (FAKE)"
    verdict_color = Colors.GREEN if is_true else Colors.RED
    confidence = probability if is_true else 1.0 - probability
    
    # Extract Correction
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

    print("\n")
    print(f"{Colors.BOLD}{Colors.HEADER}╔{'═'*70}╗{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}║{'FINAL JUDGEMENT'.center(70)}║{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}╠{'═'*70}╣{Colors.ENDC}")
    
    verdict_line = f"║  VERDICT:     {verdict_color}{Colors.BOLD}{verdict_str.ljust(54)}{Colors.ENDC} ║"
    conf_line    = f"║  CONFIDENCE:  {Colors.CYAN}{f'{confidence:.2%}'.ljust(54)}{Colors.ENDC} ║"
    
    print(verdict_line)
    print(conf_line)
    print(f"{Colors.BOLD}{Colors.HEADER}╚{'═'*70}╝{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}{Colors.YELLOW}✅ FACTUAL SUMMARY:{Colors.ENDC}")
    print(f"{correction}\n")
    
    print_separator('=')
    
    if ground_truth is not None:
        match_icon = "✅" if is_true == ground_truth else "❌"
        print(f"Validation: Prediction {is_true} | Ground Truth {ground_truth} -> {match_icon}")

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
        print_header("FIRE-DEBATE INTERFACE")
        print(f"{Colors.CYAN}Enter a claim to investigate:{Colors.ENDC}")
        user_input = input(f"{Colors.BOLD}> {Colors.ENDC}").strip()
        if user_input: 
            predict(user_input, gt)