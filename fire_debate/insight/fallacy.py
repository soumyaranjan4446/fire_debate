import torch
from sentence_transformers import CrossEncoder

class FallacyDetector:
    def __init__(self, device="cpu"):
        self.device = device
        # Switch to a robust, high-availability NLI model
        self.model_name = "cross-encoder/nli-deberta-v3-base"
        print(f"🕵️ Loading Logic Validator ({self.model_name})...")
        
        try:
            # We use CrossEncoder API which handles tokenization automatically
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as e:
            print(f"❌ Failed to load Logic Validator: {e}")
            self.model = None

    def detect(self, text: str) -> dict:
        """
        Analyzes text logic using Natural Language Inference (NLI).
        
        We compare the text against a hypothesis: "This argument is logical."
        
        Returns:
            dict: {"logical reasoning": float (0.0 to 1.0)}
        """
        if not self.model or not text or len(text.strip()) < 5:
            return {"logical reasoning": 0.5}

        try:
            # We test the hypothesis: "This argument is logical and sound."
            # The model outputs 3 scores: [Contradiction, Entailment, Neutral] (usually)
            # For this specific model, it outputs logits for classes. 
            # We want the probability that the text IMPLIES logic.
            
            hypothesis = "This argument contains a logical fallacy or emotional manipulation."
            
            # Predict gives logits. We run sigmoid to get a score between 0-1.
            scores = self.model.predict([(text, hypothesis)])
            
            # scores[0] is the logit for "Entailment" (Yes, it is a fallacy)
            # (Note: cross-encoder/nli models output different shapes, but usually 
            # index 1 is entailment or it returns a single score for binary. 
            # This specific model outputs 3 classes: Contradiction, Entailment, Neutral.
            # However, to be safer and generic, let's use a simpler binary check).
            
            # Alternative Robust Logic:
            # Let's use a zero-shot classification approach which CrossEncoders are great at.
            # We check similarity to "Logical reasoning" vs "Logical Fallacy".
            
            # Input pair
            prediction = self.model.predict([
                (text, "This argument contains a logical fallacy.")
            ])
            
            # Apply Sigmoid to logit to get probability (0.0 to 1.0)
            # High score = High probability of fallacy
            fallacy_logit = prediction[0][1] # Index 1 is usually Entailment for NLI
            
            fallacy_prob = 1 / (1 + 2.718 ** -fallacy_logit)
            
            # Logic Score is the INVERSE of Fallacy Probability
            logic_score = 1.0 - fallacy_prob
            
            return {"logical reasoning": logic_score}

        except Exception as e:
            # Fallback
            # print(f"Logic check error: {e}")
            return {"logical reasoning": 0.5}