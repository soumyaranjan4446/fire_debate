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
        
        print(f"🕸️  Initializing GraphBuilder on {self.device}...")
        
        # Use a small, fast embedder for graph node features (384 dim)
        # This makes the "Argument Node" contain actual semantic meaning
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # The "Neuro-Symbolic" Logic Gate
        self.fallacy_detector = FallacyDetector(device=self.device)

    def build_graph(self, log: DebateLog) -> HeteroData:
        data = HeteroData()
        # Debug print (optional, can comment out for speed)
        # print(f"🕸️ Building Graph: {log.debate_id}")

        # --- 1. PROCESS TURNS ---
        arg_texts = []
        fallacy_weights = [] 
        stance_features = []

        for t in log.turns:
            # Skip empty turns or system messages if any
            if not t.text or len(t.text) < 5:
                continue

            arg_texts.append(t.text)
            
            # Stance encoding: PRO=1.0, CON=-1.0, NEUTRAL=0.0
            if t.stance == "PRO": s_val = 1.0
            elif t.stance == "CON": s_val = -1.0
            else: s_val = 0.0
            stance_features.append(s_val)

            # Neuro-Symbolic Logic: Detect Fallacy
            # FIX: Use the 'detect' method we defined in fallacy.py
            scores = self.fallacy_detector.detect(t.text)
            
            # Get the probability that this turn is LOGICAL (0.0 to 1.0)
            logic_score = scores.get("logical reasoning", 0.5)
            fallacy_weights.append(logic_score)

        # --- 2. CREATE NODE FEATURES ---
        if not arg_texts:
            # Handle edge case: Empty debate (prevent crash)
            data['argument'].x = torch.zeros(1, 386) # 384 + 1 + 1
            data['argument'].num_nodes = 1
            return data

        # Encode text to vectors [Num_Turns, 384]
        # This is where the graph gets its "Semantic Understanding"
        base_embeddings = self.encoder.encode(arg_texts, convert_to_tensor=True, device=self.device)
        
        # Create Tensors
        weights = torch.tensor(fallacy_weights, device=self.device).unsqueeze(1)
        stances = torch.tensor(stance_features, device=self.device).unsqueeze(1)

        # --- THE NEURO-SYMBOLIC INJECTION ---
        # We multiply the embedding by the "Logic Score".
        # 
        # Logic: If logic_score is 0.1 (Fallacy), the vector shrinks towards zero.
        # The GNN will learn to ignore these "weak" nodes.
        weighted_embeddings = base_embeddings * weights

        # Final Feature Concatenation:
        # [ Weighted_Embedding (384) | Stance (1) | Logic_Score (1) ]
        # Total Dimension = 386
        data['argument'].x = torch.cat([weighted_embeddings, stances, weights], dim=1)

        # --- 3. BUILD EDGES ---
        # Temporal Flow: Turn 0 -> Turn 1 -> Turn 2
        num_turns = len(arg_texts)
        if num_turns > 1:
            src = list(range(num_turns - 1))
            dst = list(range(1, num_turns))
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
            data['argument', 'follows', 'argument'].edge_index = edge_index
        else:
            # Handle edge case of single-turn debate
            data['argument', 'follows', 'argument'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return data