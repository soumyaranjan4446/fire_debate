import os
import uuid
import time
import numpy as np
from typing import List, Dict, Any

# Database & Search
from tavily import TavilyClient
# --- NEW IMPORT: DuckDuckGo ---
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False
    print("⚠️ Warning: 'duckduckgo-search' not installed. Fallback will be disabled.")

import chromadb
from chromadb.utils import embedding_functions

# Advanced Retrieval Logic
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

class EvidenceRetriever:
    def __init__(self, config):
        self.cfg = config['retrieval']
        tavily_cfg = config.get('tavily', {})
        
        # --- 1. Tavily Setup (Multi-Key Support) ---
        # Supports both single 'api_key' and list 'api_keys'
        self.api_keys = tavily_cfg.get('api_keys', [])
        if not self.api_keys and 'api_key' in tavily_cfg:
            self.api_keys = [tavily_cfg['api_key']]
            
        # Fallback to env var
        if not self.api_keys:
            env_key = os.getenv("TAVILY_API_KEY")
            if env_key: self.api_keys = [env_key]

        if not self.api_keys: 
            raise ValueError("❌ Missing Tavily API Key in config!")
        
        self.current_key_idx = 0
        self.tavily = self._get_client()
        
        # 2. Embedder Setup (BGE-Small)
        print(f"📚 Loading Embedder: {self.cfg['embedding_model']}")
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.cfg['embedding_model']
        )
        
        # 3. Re-ranker Setup (Cross-Encoder)
        print("⚙️  Loading Cross-Encoder (Re-ranker)...")
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"⚠️ Could not load CrossEncoder: {e}. Re-ranking will be disabled.")
            self.reranker = None
        
        # 4. Database Setup
        self.client = chromadb.PersistentClient(path=self.cfg['db_path'])
        self.clear_cache()

    def _get_client(self):
        """Initializes client with the current active key."""
        key = self.api_keys[self.current_key_idx]
        return TavilyClient(api_key=key)

    def _rotate_key(self):
        """Switches to the next available API key."""
        if len(self.api_keys) <= 1:
            # print(f"⚠️ Warning: Only 1 Tavily key available. Cannot rotate.")
            return False
            
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        print(f"🔄 Rotating Tavily Key -> Now using Key #{self.current_key_idx + 1}")
        self.tavily = self._get_client()
        return True

    def clear_cache(self):
        """Wipes the vector DB."""
        try: 
            self.client.delete_collection("evidence_cache")
        except ValueError: 
            pass 
        self.collection = self.client.get_or_create_collection(
            name="evidence_cache", 
            embedding_function=self.emb_fn
        )

    def log_retrieval(self, query: str, docs: List[Dict[str, Any]]):
        """DEBUG: Saves retrieval results to a log file."""
        log_path = "data/processed/retrieval_debug.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n🔍 QUERY: {query}\n")
                if not docs:
                    f.write("   ❌ NO RESULTS FOUND.\n")
                for i, doc in enumerate(docs):
                    score = doc.get('score', 0.0)
                    title = doc.get('title', 'Unknown')
                    url = doc.get('url', doc.get('source', 'N/A'))
                    snippet = doc.get('content', '')[:150]
                    f.write(f"   [{i+1}] {title} (Score: {score:.2f})\n")
        except Exception:
            pass # Never crash on logging

    def search_web(self, query: str) -> List[Dict[str, Any]]:
        docs = []
        
        # --- PHASE A: TAVILY (With Key Rotation) ---
        max_retries = len(self.api_keys) + 1
        
        for attempt in range(max_retries):
            try:
                res = self.tavily.search(query=query, search_depth="basic", max_results=5)
                for r in res.get('results', []):
                    docs.append({
                        'doc_id': str(uuid.uuid4())[:8],
                        'url': r.get('url'),
                        'source': r.get('url'),
                        'title': r.get('title'),
                        'content': r.get('content'),
                        'text': r.get('content'),
                        'score': float(r.get('score', 0.8)) # Force Float
                    })
                self.log_retrieval(f"WEB SEARCH (Tavily): {query}", docs)
                return docs # Success! Return immediately.
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check for Rate Limit (429) or Quota issues
                if "429" in error_msg or "quota" in error_msg or "limit" in error_msg:
                    print(f"🛑 Key #{self.current_key_idx + 1} Hit Limit! ({e})")
                    switched = self._rotate_key()
                    if not switched:
                        break # Stop trying Tavily if we can't switch
                else:
                    # Random network error, wait briefly
                    time.sleep(1)

        # --- PHASE B: DUCKDUCKGO (Free Fallback) ---
        if HAS_DDG:
            try:
                print(f"🦆 Switching to DuckDuckGo Fallback for: '{query}'...")
                ddg_results = DDGS().text(query, max_results=4)
                if ddg_results:
                    for r in ddg_results:
                        docs.append({
                            'doc_id': str(uuid.uuid4())[:8],
                            'url': r.get('href'),        # DDG uses 'href'
                            'source': r.get('href'),
                            'title': r.get('title'),
                            'content': r.get('body'),    # DDG uses 'body'
                            'text': r.get('body'),
                            'score': 0.7                 # slightly lower confidence for free search
                        })
                    self.log_retrieval(f"WEB SEARCH (DDG): {query}", docs)
                    return docs
            except Exception as e:
                print(f"   ⚠️ DDG Fallback failed: {e}")

        print("❌ All Search Methods Failed.")
        return docs

    def index_documents(self, docs: List[Dict[str, Any]]):
        if not docs: return
        try:
            ids = [d['doc_id'] for d in docs]
            documents = [d['content'] for d in docs]
            metadatas = [{"url": d['url'], "title": d['title']} for d in docs]
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        except Exception:
            pass 

    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        # Guard clause for empty collection
        if self.collection.count() == 0: return []

        # A. Broad Vector Search
        n_fetch = min(self.cfg['top_k_evidence'] * 4, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=n_fetch)
        
        candidates = []
        if results['ids']:
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            documents = results['documents'][0]
            for i in range(len(ids)):
                candidates.append({
                    'doc_id': ids[i],
                    'title': metadatas[i]['title'],
                    'url': metadatas[i]['url'],
                    'source': metadatas[i]['url'],
                    'content': documents[i],
                    'text': documents[i],
                    'score': 0.5 
                })

        if not candidates: return []

        # B. Keyword Search (BM25)
        # We use a flag to avoid "Ambiguous Truth Value" errors on numpy arrays
        bm25_available = False
        bm25_scores = None
        
        try:
            tokenized_corpus = [doc['content'].split() for doc in candidates]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(query.split())
            bm25_available = True
        except Exception:
            bm25_available = False

        # C. Re-ranking (Cross-Encoder)
        cross_scores = None
        rerank_available = False
        if self.reranker:
            try:
                pairs = [[query, doc['content']] for doc in candidates]
                cross_scores = self.reranker.predict(pairs)
                rerank_available = True
            except Exception:
                rerank_available = False

        # D. Fusion (Hybrid Score)
        # Pre-calculate max for normalization
        max_bm25 = 1.0
        if bm25_available and len(bm25_scores) > 0:
             max_bm25 = max(bm25_scores)
             if max_bm25 == 0: max_bm25 = 1.0

        for i, doc in enumerate(candidates):
            # 1. Normalize BM25
            norm_bm25 = 0.0
            if bm25_available:
                # Safe access to numpy array
                norm_bm25 = bm25_scores[i] / max_bm25

            # 2. Get Cross Score
            c_score = 0.0
            if rerank_available:
                c_score = cross_scores[i]

            # 3. Combine
            # If we have re-ranker, use it heavily (70%). Otherwise rely on BM25.
            if rerank_available:
                final_logit = (0.7 * c_score) + (0.3 * norm_bm25)
                final_prob = 1 / (1 + np.exp(-final_logit)) # Sigmoid
            else:
                final_prob = norm_bm25

            # CRITICAL FIX: Explicit cast to python float
            doc['score'] = float(final_prob)

        # Sort and Cut
        ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)
        final_docs = ranked[:self.cfg['top_k_evidence']]
        
        self.log_retrieval(f"MEMORY RETRIEVAL: {query}", final_docs)
        return final_docs
    
    def retrieve(self, query: str):
        return self.retrieve_context(query)