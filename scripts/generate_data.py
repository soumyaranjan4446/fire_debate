import yaml
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager

def main():
    # 1. Setup
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize Core Systems
    llm = LocalHFClient(cfg)
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()
    
    # Agents
    alice = DebaterAgent(AgentConfig("Alice", "PRO"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON"), llm, retriever, librarian)
    manager = DebateManager(alice, bob, retriever)

    # 2. Load Real World Data (LIAR Dataset)
    print("📥 Loading LIAR dataset...")
    dataset = load_dataset("liar", split="train")
    
    # Take only first 5 examples for testing (Change to 100+ for real research)
    subset = dataset.select(range(5)) 
    
    output_dir = Path("data/processed/training_set")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔥 Starting Batch Generation...")
    for i, row in tqdm(enumerate(subset), total=len(subset)):
        claim = row['statement']
        label = row['label'] # 0=false, 1=half-true, ..., 5=true (Simplified below)
        
        # Simplify Label to Binary (True/False) for this experiment
        # LIAR labels: 0,1,2 usually considered 'false-leaning', 3,4,5 'true-leaning'
        binary_gt = True if label > 2 else False
        
        try:
            # RUN THE DEBATE
            log = manager.run_debate(claim, rounds=2)
            
            # Inject Ground Truth into the log so we can train on it
            log.ground_truth = binary_gt
            
            # Save individually
            manager.save_log(log, f"{output_dir}/sample_{i}_{log.debate_id}.json")
            
        except Exception as e:
            print(f"❌ Failed on sample {i}: {e}")
            continue

    print("✅ Data Generation Complete.")

if __name__ == "__main__":
    main()