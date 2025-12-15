import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LocalHFClient:
    def __init__(self, config):
        self.cfg = config['llm']
        self.device = config['device']
        model_id = self.cfg['model_id']
        
        print(f"⚙️  Loading Local LLM: {model_id}...")
        
        # 1. Setup Quantization (Crucial for running 7B+ models)
        bnb_config = None
        if self.cfg.get('quantization') == '4bit':
            print("   📉 Using 4-bit Quantization (Saving Memory)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # 2. Load Tokenizer (With Critical Fix)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # FIX: Llama-3/Qwen often lack a pad token, causing crashes. We fix it here.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 3. Load Model
        # device_map="auto" handles GPU placement automatically
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens=None, temperature=None) -> str:
        """
        Generates text using the correct Chat Template.
        Compatible with DebaterAgent's 'full_prompt' string input.
        """
        if max_new_tokens is None: 
            max_new_tokens = self.cfg.get('max_new_tokens', 512)
        if temperature is None: 
            temperature = self.cfg.get('temperature', 0.7)

        # 1. Format for Chat Model
        # Since DebaterAgent sends a single string containing the system instructions
        # and context, we wrap it all in a 'user' message so the model treats it as a task.
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 2. Apply Template
        # This converts the list to the specific string format (e.g., <|im_start|>user...)
        # required by Qwen or Llama-3.
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        # 3. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 4. Decode (Skip the input prompt, return only the answer)
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response.strip()