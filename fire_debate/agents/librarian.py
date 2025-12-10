from typing import List
from fire_debate.schemas.evidence import EvidenceDoc

class Librarian:
    def __init__(self):
        # Heuristic rules for authority (Simple version)
        self.trusted_domains = [".gov", ".edu", ".org", "reuters", "apnews"]
    def filter_and_score(self, docs: List[EvidenceDoc]) -> List[EvidenceDoc]:
        """Adjusts reliability scores based on domain authority."""
        for doc in docs:
            for domain in self.trusted_domains:
                if domain in doc.source_url:
                    doc.reliability_score = min(0.95, doc.reliability_score + 0.3)
            
            # Penalize very short snippets
            if len(doc.snippet) < 50:
                doc.reliability_score *= 0.5
                
        # Sort by reliability
        return sorted(docs, key=lambda x: x.reliability_score, reverse=True)