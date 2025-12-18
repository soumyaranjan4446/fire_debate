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
        self.encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        
        # 2. Initialize FallacyDetector with SHARED encoder
        self.fallacy_detector = FallacyDetector(device=self.device, shared_encoder=self.encoder)

    def build_graph(self, log: DebateLog) -> HeteroData:
        data = HeteroData()

        # --- 1. PROCESS TURNS (ARGUMENTS) ---
        arg_texts = []
        logic_scores = []
        relevance_scores = [] 
        stance_features = []
        
        # Phase 1 Prep: Rebuttal Logic
        arg_rebuts_pairs = [] 

        claim_emb = self.encoder.encode(log.claim_text, convert_to_tensor=True, device=self.device)

        # --- 2. PREPARE EVIDENCE CONTAINERS ---
        evidence_texts = []
        evidence_scores = [] 
        evidence_urls = []   
        
        # Deduplication Cache: content_string -> evidence_node_index
        evidence_cache = {}

        # Edge lists
        arg_to_evidence_supports = []    
        arg_to_evidence_contradicts = [] 

        for t in log.turns:
            if not t.text or len(t.text) < 5: 
                continue # Skip empty turns

            # --- CRITICAL FIX: TRACK REAL ARGUMENT INDEX ---
            # We use the current length of the list, not the loop index 'i'
            # This ensures edges point to valid nodes even if some turns were skipped.
            arg_idx = len(arg_texts)
            
            # A. Argument Feature Prep
            arg_texts.append(t.text)
            
            # Stance: PRO=1.0, CON=-1.0, Default=0.0
            s_val = 1.0 if t.stance == "PRO" else (-1.0 if t.stance == "CON" else 0.0)
            stance_features.append(s_val)

            # Context-Aware Fallacy Detection
            scores = self.fallacy_detector.detect(t.text, context=log.claim_text)
            logic_scores.append(scores.get("logical reasoning", 0.5))
            relevance_scores.append(scores.get("relevance", 0.5))

            # B. Phase 1 Logic: Detect Rebuttals
            # If REBUTTAL, it attacks the immediately previous argument (arg_idx - 1)
            phase = getattr(t, 'phase', 'ARGUMENT')
            if phase == "REBUTTAL" and arg_idx > 0:
                arg_rebuts_pairs.append((arg_idx, arg_idx - 1))

            # C. Evidence Extraction (Robust & Deduplicated)
            if hasattr(t, 'citations') and t.citations:
                for cite in t.citations:
                    # Handle Dicts vs Strings
                    if isinstance(cite, dict):
                         content = cite.get('text', cite.get('content', 'Evidence Link'))
                         score = cite.get('score', 0.9) 
                         url = cite.get('url', '')
                    else:
                         content = str(cite)
                         score = 0.9 
                         url = str(cite)

                    # --- DEDUPLICATION LOGIC ---
                    if content in evidence_cache:
                        # Existing Node: Reuse index
                        ev_idx = evidence_cache[content]
                    else:
                        # New Node: Create it
                        ev_idx = len(evidence_texts)
                        evidence_texts.append(content) 
                        evidence_scores.append(score)
                        evidence_urls.append(url)
                        evidence_cache[content] = ev_idx
                    
                    # D. Determine Edge Type (Using corrected arg_idx)
                    if t.stance == "PRO":
                        arg_to_evidence_supports.append((arg_idx, ev_idx))
                    elif t.stance == "CON":
                        arg_to_evidence_contradicts.append((arg_idx, ev_idx))

        # --- 3. BUILD ARGUMENT NODES (Dim 771) ---
        if not arg_texts:
            data['argument'].x = torch.zeros(1, 771) 
            data['argument'].num_nodes = 1
            return data

        arg_embeddings = self.encoder.encode(arg_texts, convert_to_tensor=True, device=self.device)
        claim_context = claim_emb.repeat(len(arg_texts), 1)
        
        l_scores = torch.tensor(logic_scores, device=self.device).unsqueeze(1)
        r_scores = torch.tensor(relevance_scores, device=self.device).unsqueeze(1)
        stances = torch.tensor(stance_features, device=self.device).unsqueeze(1)

        # Concat: 384 + 384 + 1 + 1 + 1 = 771
        features = torch.cat([arg_embeddings, claim_context, stances, l_scores, r_scores], dim=1)
        data['argument'].x = features
        data['argument'].num_nodes = features.size(0) # Explicit Count

        # --- 4. BUILD EVIDENCE NODES (Rich Features + Padding) ---
        if evidence_texts:
            # A. Embed Text (384 dims)
            ev_embeddings = self.encoder.encode(evidence_texts, convert_to_tensor=True, device=self.device)
            
            # B. Extract Signals
            domain_scores = []
            confidence_scores = []
            reliability_list = []

            for k, url in enumerate(evidence_urls):
                base_score = evidence_scores[k]
                url_lower = url.lower()
                
                # 1. Reliability (Trust the Librarian's output)
                reliability_list.append(base_score)
                
                # 2. Confidence Proxy
                confidence_scores.append(base_score if base_score > 0.5 else 0.1)
                
                # 3. Domain Type Feature (Explicit Signal)
                if any(d in url_lower for d in ['.gov', '.edu', '.mil', 'nature.com', 'science.org']):
                    d_val = 1.0 # Gold Standard
                elif any(d in url_lower for d in ['bbc', 'cnn', 'reuters', 'wikipedia', 'mayoclinic', 'apnews']):
                    d_val = 0.8 # Trusted Media
                else:
                    d_val = 0.5 # General Web
                
                domain_scores.append(d_val)

            # Convert to Tensors
            t_rel = torch.tensor(reliability_list, device=self.device).unsqueeze(1)
            t_dom = torch.tensor(domain_scores, device=self.device).unsqueeze(1)
            t_conf = torch.tensor(confidence_scores, device=self.device).unsqueeze(1)
            
            # C. Padding (Target: 771)
            # Current: 384 (Text) + 1 (Rel) + 1 (Dom) + 1 (Conf) = 387
            # Needed: 771 - 387 = 384
            padding = torch.zeros(ev_embeddings.size(0), 771 - 387, device=self.device)
            
            # D. Concatenate
            ev_features = torch.cat([ev_embeddings, t_rel, t_dom, t_conf, padding], dim=1)
            data['evidence'].x = ev_features
            data['evidence'].num_nodes = ev_features.size(0)
        else:
            # Safe pass
            pass 

        # --- 5. BUILD EDGES ---
        def make_edge_tensor(pairs):
            if not pairs: return torch.empty((2, 0), dtype=torch.long, device=self.device)
            s, d = zip(*pairs)
            return torch.tensor([s, d], dtype=torch.long, device=self.device)

        # Temporal Flow
        if len(arg_texts) > 1:
            src = list(range(len(arg_texts) - 1))
            dst = list(range(1, len(arg_texts)))
            data['argument', 'follows', 'argument'].edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        else:
            data['argument', 'follows', 'argument'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        # Evidence Connections
        if evidence_texts:
            data['argument', 'supports', 'evidence'].edge_index = make_edge_tensor(arg_to_evidence_supports)
            data['argument', 'contradicts', 'evidence'].edge_index = make_edge_tensor(arg_to_evidence_contradicts)

        # Phase 1 Rebuttal Edges
        rebuts_tensor = make_edge_tensor(arg_rebuts_pairs)
        # Uncomment when ready:
        # data['argument', 'rebuts', 'argument'].edge_index = rebuts_tensor

        return data