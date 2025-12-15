from typing import List, Dict, Any

class Librarian:
    """
    The Gatekeeper.
    While the Retriever finds 'relevant' text, the Librarian ensures it is 'credible'.
    """
    def __init__(self, reliability_threshold: float = 0.6):
        # We set a strict threshold. If the Retriever/Cross-Encoder is less than 60% sure,
        # we treat the evidence as noise and discard it.
        self.threshold = reliability_threshold
        
        # A small whitelist of "Gold Standard" domains to boost
        self.trusted_domains = [
            ".gov", ".edu", "reuters.com", "apnews.com", "nature.com", 
            "science.org", "bbc.com", "who.int", "npr.org"
        ]

    def filter_evidence(self, docs: List[Dict[str, Any]], claim_context: str = None) -> List[Dict[str, Any]]:
        """
        Filters out low-score evidence and boosts trusted domains.
        Expects docs to be a list of dictionaries (e.g., from Tavily or Chroma).
        """
        approved_docs = []
        
        for doc in docs:
            # Safe dictionary access (Tavily usually returns 'url' or 'source')
            url = doc.get('url', doc.get('source', ''))
            score = doc.get('score', 0.0)
            
            # 1. Domain Boost: If it's from a trusted source, give it a small bonus
            # This helps standard logic override neural noise
            for domain in self.trusted_domains:
                if domain in url:
                    score = min(0.99, score + 0.15) # Boost score
                    doc['score'] = score # Update in place
                    break # Only boost once

            # 2. Threshold Check: Reject weak evidence
            # This prevents hallucinations caused by reading irrelevant text
            if score >= self.threshold:
                approved_docs.append(doc)
            else:
                # Optional: Log rejected docs for debugging
                # print(f"   [Librarian] Rejected: {doc.get('title', 'Unknown')} ({score:.2f})")
                pass
        
        # 3. CRITICAL FALLBACK (Silence Prevention)
        # If the filter is too strict and removes EVERYTHING, the agent will have no info.
        # In this case, we grudgingly return the top 1 result so the agent isn't blind.
        if not approved_docs and docs:
            # print("   [Librarian] ⚠️ All evidence rejected. Returning top result as fallback.")
            return [docs[0]]

        return approved_docs