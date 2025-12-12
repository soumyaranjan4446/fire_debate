from abc import ABC, abstractmethod

class LLMClient(ABC):
    """
    Abstract Interface for the LLM. 
    This allows you to swap Qwen for Llama or GPT-4 without breaking the Agents.
    """
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Takes a system instruction (persona) and a user input (task),
        and returns the model's text response.
        """
        pass