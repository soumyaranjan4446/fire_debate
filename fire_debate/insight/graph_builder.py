import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from fire_debate.schemas.debate import DebateLog
from fire_debate.insight.fallacy import FallacyDetector

class GraphBuilder:
    def __init__(self, device="cuda"):
        self.use_gpu = (device == "cuda" and torch.cuda.is_available())
        self.device = "cuda" if self.use_gpu else "cpu"
        
        print(f"🕸️  Initializing GraphBuilder on {self.device}...")
        
        # 1. Load the Encoder (Unified to BGE-Small)
        # This matches your base.yaml config
        self.encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        
        # 2. Initialize FallacyDetector with SHARED encoder
        # FIX: We pass the encoder here so we don't load the model twice into VRAM
        self.fallacy_detector = FallacyDetector(device=self.device, shared_encoder=self.encoder)

    def build_graph(self, log: DebateLog) -> HeteroData:
        data = HeteroData()

        # --- 1. PROCESS TURNS & CALCULATE SCORES ---
        arg_texts = []
        logic_scores = []
        relevance_scores = [] 
        stance_features = []

        # Pre-calculate Claim Embedding (Global Context)
        # This is the "Global Goal" vector
        claim_emb = self.encoder.encode(log.claim_text, convert_to_tensor=True, device=self.device)

        for t in log.turns:
            if not t.text or len(t.text) < 5: continue

            arg_texts.append(t.text)
            
            # Stance encoding
            if t.stance == "PRO": s_val = 1.0
            elif t.stance == "CON": s_val = -1.0
            else: s_val = 0.0
            stance_features.append(s_val)

            # --- UPGRADE: PASS CONTEXT TO DETECTOR ---
            scores = self.fallacy_detector.detect(t.text, context=log.claim_text)
            
            logic_scores.append(scores.get("logical reasoning", 0.5))
            relevance_scores.append(scores.get("relevance", 0.5))

        # --- 2. CREATE NODE FEATURES (The Upgrade) ---
        if not arg_texts:
            # Fallback size: 384 (Arg) + 384 (Claim) + 3 (Stance/Logic/Rel) = 771
            data['argument'].x = torch.zeros(1, 771) 
            data['argument'].num_nodes = 1
            return data

        # Encode Arguments
        arg_embeddings = self.encoder.encode(arg_texts, convert_to_tensor=True, device=self.device)
        
        # Expand Claim embedding to match number of arguments
        # [1, 384] -> [Num_Args, 384]
        claim_context = claim_emb.repeat(len(arg_texts), 1)

        # Create Score Tensors
        l_scores = torch.tensor(logic_scores, device=self.device).unsqueeze(1)
        r_scores = torch.tensor(relevance_scores, device=self.device).unsqueeze(1)
        stances = torch.tensor(stance_features, device=self.device).unsqueeze(1)

        # --- CONTEXT INJECTION ---
        # Instead of shrinking vectors, we Concatenate everything.
        # Node = [Argument Vector] || [Claim Context Vector] || [Stance] || [Logic] || [Relevance]
        # Size = 384 + 384 + 1 + 1 + 1 = 771
        
        features = torch.cat([arg_embeddings, claim_context, stances, l_scores, r_scores], dim=1)
        data['argument'].x = features

        # --- 3. BUILD EDGES ---
        num_turns = len(arg_texts)
        if num_turns > 1:
            # Sequential flow: Turn 0 -> Turn 1 -> Turn 2
            src = list(range(num_turns - 1))
            dst = list(range(1, num_turns))
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
            data['argument', 'follows', 'argument'].edge_index = edge_index
        else:
            data['argument', 'follows', 'argument'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return data