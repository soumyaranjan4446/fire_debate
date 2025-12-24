import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import numpy as np

class FallacyDetector:
    def __init__(self, device="cpu", shared_encoder=None):
        self.device = device
        
        # 1. Logic Checker (NLI Cross-Encoder)
        # We stick to DistilRoBERTa because it's faster and more stable for simple logic checks
        self.logic_model_name = "cross-encoder/nli-distilroberta-base"
        print(f"🕵️  Loading Logic Validator ({self.logic_model_name})...")
        try:
            self.logic_model = CrossEncoder(self.logic_model_name, device=self.device)
        except Exception as e:
            print(f"❌ Failed to load Logic Model: {e}")
            self.logic_model = None

        # 2. Relevance Checker (Semantic Embeddings)
        # OPTIMIZATION: Use the shared BGE encoder from GraphBuilder to save VRAM
        if shared_encoder:
            # print(f"🔗 Using shared Relevance Validator (Memory Optimized)...")
            self.rel_model = shared_encoder
        else:
            # Fallback if initialized alone (Updated to match base.yaml)
            self.rel_model_name = "BAAI/bge-small-en-v1.5"
            print(f"🔗 Loading Relevance Validator ({self.rel_model_name})...")
            try:
                self.rel_model = SentenceTransformer(self.rel_model_name, device=self.device)
            except Exception as e:
                print(f"❌ Failed to load Relevance Model: {e}")
                self.rel_model = None

    def detect(self, text: str, context: str = None) -> dict:
        """
        Analyzes text for:
        1. Logic Score (NLI: Is it a fallacy?)
        2. Relevance Score (Cosine Sim: Is it on topic?)
        """
        if not text or len(text.strip()) < 5:
            return {"logical reasoning": 0.5, "relevance": 0.5}

        results = {}

        # --- CHECK 1: LOGIC (NLI) ---
        # Hypothesis: "This argument contains a logical fallacy."
        # DistilRoBERTa NLI mapping: 0:Contradiction, 1:Entailment, 2:Neutral
        if self.logic_model:
            try:
                # Predict returns numpy array of logits
                pred = self.logic_model.predict([(text, "This argument contains a logical fallacy.")])
                
                # Convert to tensor for softmax
                logits = torch.tensor(pred, dtype=torch.float32)
                probs = torch.nn.functional.softmax(logits, dim=0)
                
                # Index 1 is Entailment. If high, it IS a fallacy.
                fallacy_prob = probs[1].item() 
                results["logical reasoning"] = 1.0 - fallacy_prob
            except Exception as e:
                # print(f"Logic check warning: {e}")
                results["logical reasoning"] = 0.5
        else:
             results["logical reasoning"] = 0.5

        # --- CHECK 2: RELEVANCE (Cosine Similarity) ---
        if context and self.rel_model:
            try:
                # Encode both sentences
                emb_text = self.rel_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
                emb_context = self.rel_model.encode(context, convert_to_tensor=True, show_progress_bar=False)
                
                # Calculate Cosine Similarity (0.0 to 1.0)
                similarity = util.cos_sim(emb_text, emb_context).item()
                
                # Normalize: Cosine sim can be negative (-1 to 1). We clamp to 0-1.
                results["relevance"] = max(0.0, similarity)
            except Exception as e:
                # print(f"Relevance check warning: {e}")
                results["relevance"] = 0.5
        else:
            results["relevance"] = 1.0 

        return results