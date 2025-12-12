from typing import List
from fire_debate.schemas.evidence import EvidenceDoc

class Librarian:
    """
    The Gatekeeper.
    While the Retriever finds 'relevant' text, the Librarian ensures it is 'credible'.
    """
    def __init__(self, min_reliability: float = 0.6):
        # We set a strict threshold. If the Cross-Encoder is less than 60% sure,
        # we treat the evidence as noise and discard it.
        self.min_reliability = min_reliability
        
        # A small whitelist of "Gold Standard" domains to boost
        self.trusted_domains = [
            ".gov", ".edu", "reuters.com", "apnews.com", "nature.com", 
            "science.org", "bbc.com", "who.int"
        ]

    def filter_evidence(self, docs: List[EvidenceDoc]) -> List[EvidenceDoc]:
        """
        Filters out low-score evidence and boosts trusted domains.
        """
        approved_docs = []
        for doc in docs:
            # 1. Domain Boost: If it's from a trusted source, give it a small bonus
            # This helps standard logic override neural noise
            for domain in self.trusted_domains:
                if domain in doc.source_url:
                    doc.reliability_score = min(0.99, doc.reliability_score + 0.1)
                    break # Only boost once

            # 2. Threshold Check: Reject weak evidence
            # This prevents hallucinations caused by reading irrelevant text
            if doc.reliability_score >= self.min_reliability:
                approved_docs.append(doc)
            else:
                # Optional: Log rejected docs for debugging
                # print(f"Librarian rejected: {doc.title} ({doc.reliability_score:.2f})")
                pass
            
        return approved_docs