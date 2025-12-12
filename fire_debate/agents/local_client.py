import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)
from .base import LLMClient

class LocalHFClient(LLMClient):
    def __init__(self, config):
        self.cfg = config['llm']
        model_id = self.cfg['model_id']
        
        print(f"⚙️ Loading Local LLM: {model_id}...")
        
        # 1. Setup Quantization (Crucial for running 7B+ models on Colab/Laptop)
        bnb_config = None
        if self.cfg.get('quantization') == '4bit':
            print("   📉 Using 4-bit Quantization (Saving Memory)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 3. Load Model
        # device_map="auto" automatically finds the GPU or falls back to CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True # Needed for some new architectures like Qwen
        )
        
        # 4. Create Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # If strictly using CPU (no quantization), device needs to be -1 or manual
            # But device_map="auto" usually handles this.
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates text using the correct Chat Template for Qwen/Llama.
        """
        # A. Format the prompt correctly (System vs User)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # B. Apply the template (converts JSON to the raw string the model expects)
        prompt_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # C. Run Inference
        outputs = self.pipe(
            prompt_str,
            max_new_tokens=self.cfg['max_new_tokens'],
            do_sample=True,
            temperature=self.cfg['temperature'],
            top_p=0.9,
            return_full_text=False # Critical: Don't repeat the input!
        )
        
        return outputs[0]["generated_text"].strip()