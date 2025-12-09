import torch
from transformers import pipeline
from typing import Dict, List

class FallacyDetector:
    def __init__(self, device="cuda"):
        print("🕵️ Loading Fallacy Detector (DeBERTa-v3-base)...")
        # DeBERTa-v3-base-mnli is excellent for zero-shot logic tasks
        self.pipe = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-base",
            device=0 if device == "cuda" else -1
        )
        self.labels = [
            "logical reasoning", 
            "ad hominem", 
            "strawman argument", 
            "appeal to emotion", 
            "circular reasoning",
            "false dichotomy"
        ]

    def analyze_turn(self, text: str) -> Dict[str, float]:
        """
        Returns a dictionary of fallacy scores.
        e.g., {'logical reasoning': 0.8, 'ad hominem': 0.1, ...}
        """
        # Split long text to avoid truncation, take the worst segment (simplified)
        chunk = text[:512] 
        result = self.pipe(chunk, self.labels, multi_label=False)
        
        scores = dict(zip(result['labels'], result['scores']))
        return scores

    def get_top_fallacy(self, scores: Dict[str, float]) -> str:
        # If 'logical reasoning' is not the top score, return the fallacy
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = sorted_scores[0]
        
        if top_label == "logical reasoning":
            return None
        return top_label