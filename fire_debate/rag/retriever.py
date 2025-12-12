import chromadb
from chromadb.utils import embedding_functions
from tavily import TavilyClient
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List
import uuid
import numpy as np
import os
from fire_debate.schemas.evidence import EvidenceDoc

class EvidenceRetriever:
    def __init__(self, config):
        self.cfg = config['retrieval']
        
        # 1. Tavily
        key = config.get('tavily', {}).get('api_key')
        if not key: raise ValueError("❌ Missing Tavily API Key in config!")
        self.tavily = TavilyClient(api_key=key)
        
        # 2. Embedder
        print(f"📚 Loading Embedder: {self.cfg['embedding_model']}")
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.cfg['embedding_model']
        )
        
        # 3. Re-ranker
        print("⚙️ Loading Cross-Encoder (Re-ranker)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 4. Database
        self.client = chromadb.PersistentClient(path=self.cfg['db_path'])
        try: self.client.delete_collection("evidence_cache")
        except: pass
        self.collection = self.client.get_or_create_collection(
            name="evidence_cache", embedding_function=self.emb_fn
        )

    def log_retrieval(self, query: str, docs: List[EvidenceDoc]):
        """DEBUG: Saves retrieval results to a log file."""
        log_path = "data/processed/retrieval_debug.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n🔍 QUERY: {query}\n")
            if not docs:
                f.write("   ❌ NO RESULTS FOUND.\n")
            for i, doc in enumerate(docs):
                f.write(f"   [{i+1}] {doc.title} (Score: {doc.reliability_score:.2f})\n")
                f.write(f"       URL: {doc.source_url}\n")
                f.write(f"       Snippet: {doc.snippet[:150]}...\n")

    def search_web(self, query: str) -> List[EvidenceDoc]:
        print(f"   🔎 Web Search: '{query}'")
        docs = []
        try:
            res = self.tavily.search(query=query, search_depth="basic", max_results=5)
            for r in res.get('results', []):
                docs.append(EvidenceDoc(
                    doc_id=str(uuid.uuid4())[:8],
                    source_url=r.get('url'),
                    title=r.get('title'),
                    snippet=r.get('content'),
                    reliability_score=r.get('score', 0.8)
                ))
            # Log what we found on the web
            self.log_retrieval(f"WEB SEARCH: {query}", docs)
        except Exception as e:
            print(f"   ❌ Tavily Error: {e}")
        return docs

    def index_documents(self, docs: List[EvidenceDoc]):
        if not docs: return
        self.collection.add(
            documents=[d.snippet for d in docs],
            metadatas=[{"url": d.source_url, "title": d.title} for d in docs],
            ids=[d.doc_id for d in docs]
        )

    def retrieve_context(self, query: str) -> List[EvidenceDoc]:
        if self.collection.count() == 0: return []

        # A. Broad Vector Search
        n_fetch = self.cfg['top_k_evidence'] * 4
        results = self.collection.query(query_texts=[query], n_results=n_fetch)
        
        candidates = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                candidates.append(EvidenceDoc(
                    doc_id=results['ids'][0][i],
                    title=results['metadatas'][0][i]['title'],
                    source_url=results['metadatas'][0][i]['url'],
                    snippet=results['documents'][0][i],
                    reliability_score=0.5
                ))

        if not candidates: return []

        # B. Keyword (BM25)
        try:
            tokenized_corpus = [doc.snippet.split() for doc in candidates]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(query.split())
        except:
            bm25_scores = [0.0] * len(candidates)

        # C. Re-ranking
        pairs = [[query, doc.snippet] for doc in candidates]
        cross_scores = self.reranker.predict(pairs)

        # D. Fusion
        for i, doc in enumerate(candidates):
            norm_bm25 = bm25_scores[i] / (max(bm25_scores) + 1e-9)
            final_score = (0.7 * cross_scores[i]) + (0.3 * norm_bm25)
            doc.reliability_score = float(1 / (1 + np.exp(-final_score)))

        ranked = sorted(candidates, key=lambda x: x.reliability_score, reverse=True)
        final_docs = ranked[:self.cfg['top_k_evidence']]
        
        # Log what the agent actually sees
        self.log_retrieval(f"MEMORY RETRIEVAL: {query}", final_docs)
        
        return final_docs
    
    def clear_cache(self):
        try: self.client.delete_collection("evidence_cache")
        except: pass
        self.collection = self.client.get_or_create_collection(
            name="evidence_cache", embedding_function=self.emb_fn
        )