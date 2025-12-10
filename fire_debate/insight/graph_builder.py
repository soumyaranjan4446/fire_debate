import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from fire_debate.schemas.debate import DebateLog
from fire_debate.insight.fallacy import FallacyDetector

class GraphBuilder:
    def __init__(self, device="cuda"):
        # Robust check: Only use CUDA if available AND requested
        self.use_gpu = (device == "cuda" and torch.cuda.is_available())
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Use a small, fast embedder for graph node features (384 dim)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.fallacy_detector = FallacyDetector(device=self.device)

    def build_graph(self, log: DebateLog) -> HeteroData:
        data = HeteroData()
        print(f"🕸️ Building Neuro-Symbolic Graph for Debate: {log.debate_id}")

        # --- 1. PROCESS TURNS ---
        arg_texts = []
        fallacy_weights = [] # This acts as the "Gate" for information
        stance_features = []

        for t in log.turns:
            arg_texts.append(t.text)
            
            # Stance encoding: PRO=1.0, CON=-1.0, NEUTRAL=0.0
            if t.stance == "PRO": s_val = 1.0
            elif t.stance == "CON": s_val = -1.0
            else: s_val = 0.0
            stance_features.append(s_val)

            # Neuro-Symbolic Logic: Detect Fallacy
            # We pass the stance so the detector can skip NEUTRAL turns (saving time)
            scores = self.fallacy_detector.analyze_turn(t.text, t.stance)
            
            # Get the probability that this turn is LOGICAL (0.0 to 1.0)
            logic_score = scores.get("logical reasoning", 0.5)
            fallacy_weights.append(logic_score)

        # --- 2. CREATE NODE FEATURES ---
        # Encode text to vectors [Num_Turns, 384]
        base_embeddings = self.encoder.encode(arg_texts, convert_to_tensor=True, device=self.device)
        
        # Create Tensors
        weights = torch.tensor(fallacy_weights, device=self.device).unsqueeze(1)
        stances = torch.tensor(stance_features, device=self.device).unsqueeze(1)

        # --- THE NEURO-SYMBOLIC INJECTION ---
        # We multiply the embedding by the "Logic Score".
        # If logic_score is low (0.1), the vector becomes very small.
        # This effectively "dims" fallacious nodes in the graph calculation.
        weighted_embeddings = base_embeddings * weights

        # Final Feature Concatenation:
        # [ Weighted_Embedding (384) | Stance (1) | Logic_Score (1) ]
        # Total Dimension = 386
        data['argument'].x = torch.cat([weighted_embeddings, stances, weights], dim=1)

        # --- 3. BUILD EDGES ---
        # Temporal Flow: Turn 0 -> Turn 1 -> Turn 2
        src = list(range(len(log.turns) - 1))
        dst = list(range(1, len(log.turns)))
        
        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
            data['argument', 'follows', 'argument'].edge_index = edge_index
        else:
            # Handle edge case of single-turn debate (rare but possible)
            data['argument', 'follows', 'argument'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return data