import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .base import LLMClient

class LocalHFClient(LLMClient):
    def __init__(self, config):
        model_id = config['llm']['model_id']
        print(f"⚙️ Loading Local LLM: {model_id} (This may take time)...")
        
        # Use pipeline for simplicity
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto", # Automatically puts layers on GPU/CPU
        )
        self.max_tokens = config['llm']['max_new_tokens']

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Standard chat formatting for Mistral/Zephyr
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        outputs = self.pipe(
            messages, 
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        return outputs[0]["generated_text"][-1]["content"]