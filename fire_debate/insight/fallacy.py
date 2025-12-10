import torch
from transformers import pipeline
from typing import Dict, List, Optional

class FallacyDetector:
    def __init__(self, device="cuda"):
        print("🕵️ Loading Fallacy Detector (DeBERTa-v3-base)...")
        
        # Robust check: Only use CUDA if available AND requested
        use_gpu = (device == "cuda" and torch.cuda.is_available())
        device_id = 0 if use_gpu else -1
        
        # DeBERTa-v3-base-mnli is excellent for zero-shot logic tasks
        self.pipe = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-base",
            device=device_id
        )
        self.labels = [
            "logical reasoning", 
            "ad hominem", 
            "strawman argument", 
            "appeal to emotion", 
            "circular reasoning",
            "false dichotomy"
        ]

    def analyze_turn(self, text: str, stance: str) -> Dict[str, float]:
        """
        Returns a dictionary of fallacy scores.
        SKIP if the stance is NEUTRAL (Moderator/Synthesizer) to save compute.
        """
        # 1. Optimization: Don't analyze Moderators or very short texts
        if stance == "NEUTRAL" or len(text) < 15:
            # Return a 'perfect logic' placeholder
            return {label: (1.0 if label == "logical reasoning" else 0.0) for label in self.labels}

        # 2. Pre-processing
        # DeBERTa has a 512 token limit. We take the first 512 chars (approx 100-150 tokens)
        # For a full paper, you might want to use a sliding window, but this is sufficient for now.
        chunk = text[:1024] 
        
        # 3. Inference
        result = self.pipe(chunk, self.labels, multi_label=False)
        
        scores = dict(zip(result['labels'], result['scores']))
        return scores

    def get_top_fallacy(self, scores: Dict[str, float]) -> Optional[str]:
        """
        Returns the name of the fallacy if it's the top score.
        Returns None if the top score is 'logical reasoning'.
        """
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = sorted_scores[0]
        
        if top_label == "logical reasoning":
            return None
        return top_label