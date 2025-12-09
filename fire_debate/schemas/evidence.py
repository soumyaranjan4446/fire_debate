from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EvidenceDoc:
    """
    Represents a single atomic piece of evidence retrieved from the world.
    """
    doc_id: str
    source_url: str
    title: str
    snippet: str
    reliability_score: float = 0.5  # Default neutral trust
    embedding: Optional[list] = field(default=None, repr=False) # Don't print embeddings