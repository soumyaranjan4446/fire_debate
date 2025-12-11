from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class EvidenceDoc:
    """
    Represents a single atomic piece of evidence retrieved from the world.
    """
    doc_id: str
    source_url: str
    title: str
    snippet: str
    
    # Default neutral score (0.5) if re-ranking fails
    reliability_score: float = 0.5  
    
    # 1. OPTIMIZATION: repr=False hides this huge vector from print() statements
    embedding: Optional[List[float]] = field(default=None, repr=False)
    
    def to_dict(self) -> dict:
        """
        Helper to convert to dictionary WITHOUT the embedding.
        Critical for saving logs to JSON without bloating file size.
        """
        return {
            "doc_id": self.doc_id,
            "source_url": self.source_url,
            "title": self.title,
            "snippet": self.snippet,
            "reliability_score": self.reliability_score
        }