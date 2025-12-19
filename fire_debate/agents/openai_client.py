import os
from openai import OpenAI

class OpenAIClient:
    def __init__(self, config):
        """
        Initializes the OpenAI Client using the same config structure as LocalHFClient.
        """
        self.cfg = config['llm']
        
        # 1. Setup API Key
        # Prioritize key in config (base.yaml), then fallback to Environment Variable
        api_key = self.cfg.get('api_key') or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("❌ OpenAI API Key missing! Add 'api_key' to configs/base.yaml or set 'OPENAI_API_KEY' environment variable.")
            
        self.client = OpenAI(api_key=api_key)
        
        # 2. Setup Model ID
        # Reads 'model_id' from base.yaml (e.g., "gpt-4o-mini")
        self.model_name = self.cfg.get('model_id', "gpt-4o-mini")
        
        print(f"⚙️  Loading OpenAI LLM: {self.model_name}...")

    def generate(self, prompt: str, max_new_tokens=None, temperature=None) -> str:
        """
        Generates text using OpenAI API.
        Matches the signature and behavior of LocalHFClient.generate().
        """
        # 1. Handle Defaults (Same logic as LocalHFClient)
        if max_new_tokens is None: 
            max_new_tokens = self.cfg.get('max_new_tokens', 512)
        if temperature is None: 
            temperature = self.cfg.get('temperature', 0.7)

        # 2. Format Messages
        # LocalHFClient wraps the entire prompt in a single 'user' message.
        # We do the same here to ensure the model sees the exact same context.
        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            # 3. Generate
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens
            )
            
            # 4. Return Text (Strip to match LocalHFClient output)
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️ OpenAI API Error: {e}")
            # Return empty string to prevent pipeline crash
            return ""