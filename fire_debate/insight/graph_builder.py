import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from typing import List, Dict

from fire_debate.schemas.debate import DebateLog
from fire_debate.insight.fallacy import FallacyDetector

class GraphBuilder:
    def __init__(self, device="cuda"):
        self.device = device
        # We need an encoder for node features. 
        # Using a tiny model to keep it fast, separate from the RAG model.
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.fallacy_detector = FallacyDetector(device=device)

    def build_graph(self, log: DebateLog) -> HeteroData:
        data = HeteroData()
        print(f"🕸️ Building Graph for Debate: {log.debate_id}")

        # --- 1. PREPARE NODE DATA ---
        arg_texts = [t.text for t in log.turns]
        arg_embeddings = self.encoder.encode(arg_texts, convert_to_tensor=True)
        
        # Calculate Fallacy Scores (Feature Vector)
        fallacy_feats = []
        for t in log.turns:
            scores = self.fallacy_detector.analyze_turn(t.text)
            # Create a vector [score_logical, score_ad_hominem, ...]
            vec = [scores.get(l, 0.0) for l in self.fallacy_detector.labels]
            fallacy_feats.append(vec)
        
        fallacy_tensor = torch.tensor(fallacy_feats, device=self.device)

        # Concatenate Text Embedding + Fallacy Vector
        data['argument'].x = torch.cat([arg_embeddings, fallacy_tensor], dim=1)
        
        # Add Stance Feature (1 for PRO, -1 for CON)
        stances = [1.0 if t.stance == "PRO" else -1.0 for t in log.turns]
        data['argument'].stance = torch.tensor(stances, device=self.device).unsqueeze(1)

        # --- 2. PREPARE EVIDENCE NODES ---
        # (In a real run, you'd pull these from the log citations. 
        # For this demo, we assume citations are stored or we skip evidence nodes if empty)
        # Note: Ideally, pass the 'retriever' or 'evidence_map' here to look up snippets.
        # For now, we create a placeholder if no evidence is explicitly linked in objects.
        if hasattr(log, 'evidence_map') and log.evidence_map:
            # Code to handle evidence nodes would go here
            pass

        # --- 3. BUILD EDGES ---
        # Temporal Edges: Turn 0 -> Turn 1 -> Turn 2
        src = list(range(len(log.turns) - 1))
        dst = list(range(1, len(log.turns)))
        
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        data['argument', 'follows', 'argument'].edge_index = edge_index

        return data