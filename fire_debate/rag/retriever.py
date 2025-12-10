import chromadb
from chromadb.utils import embedding_functions
from tavily import TavilyClient
from typing import List
import uuid
from fire_debate.schemas.evidence import EvidenceDoc

class EvidenceRetriever:
    def __init__(self, config):
        self.cfg = config['retrieval']
        
        # 1. Initialize Tavily (The Upgrade)
        # We handle the case where the key might be missing to give a clear error
        tavily_key = config.get('tavily', {}).get('api_key')
        if not tavily_key:
            raise ValueError("❌ Missing 'tavily.api_key' in your config.yaml file!")
            
        print(f"📚 Initializing Librarian (Tavily + {self.cfg['embedding_model']})...")
        self.tavily = TavilyClient(api_key=tavily_key)
        
        # 2. Hugging Face Embedding Function
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.cfg['embedding_model']
        )
        
        # 3. Setup ChromaDB
        self.client = chromadb.PersistentClient(path=self.cfg['db_path'])
        
        # Reset collection on init to avoid stale data during testing
        try:
            self.client.delete_collection("evidence_cache")
        except:
            pass
            
        self.collection = self.client.get_or_create_collection(
            name="evidence_cache",
            embedding_function=self.emb_fn
        )

    def search_web(self, query: str) -> List[EvidenceDoc]:
        """Fetches high-quality context from Tavily."""
        print(f"   🔎 Tavily Researching: '{query}'")
        docs = []
        
        try:
            # Tavily Search
            # search_depth="basic" is faster; use "advanced" for deeper research
            response = self.tavily.search(
                query=query, 
                search_depth="basic", 
                max_results=self.cfg['max_search_results']
            )
            
            results = response.get('results', [])
            
            if not results:
                print("   ⚠️  Warning: Tavily returned 0 results.")
                return []

            for r in results:
                # Tavily returns 'content' which is cleaner than DDG's 'body'
                doc = EvidenceDoc(
                    doc_id=str(uuid.uuid4())[:8],
                    source_url=r.get('url', 'unknown'),
                    title=r.get('title', 'No Title'),
                    snippet=r.get('content', ''), 
                    # Tavily provides a relevance score, defaulting to 0.8 if missing
                    reliability_score=r.get('score', 0.8)
                )
                docs.append(doc)
            
            print(f"   ✅ Found {len(docs)} reliable sources.")
                
        except Exception as e:
            print(f"   ❌ Search Error: {e}")
            
        return docs

    def index_documents(self, docs: List[EvidenceDoc]):
        """Saves documents to ChromaDB for semantic search."""
        if not docs: 
            return
        
        ids = [d.doc_id for d in docs]
        texts = [d.snippet for d in docs]
        metadatas = [{"url": d.source_url, "title": d.title} for d in docs]
        
        # Batch add to avoid errors
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def retrieve_context(self, query: str) -> List[EvidenceDoc]:
        """Semantic Search: Finds best evidence for the query."""
        
        # Guard clause: If DB is empty, return nothing immediately
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=self.cfg['top_k_evidence']
        )
        
        retrieved_docs = []
        if results['ids']:
            # Handle ChromaDB's list-of-lists format
            count = len(results['ids'][0])
            for i in range(count):
                # Calculate safe reliability score from distance
                dist = results['distances'][0][i] if 'distances' in results and results['distances'] else 0.5
                score = max(0.0, min(1.0, 1.0 - dist))

                doc = EvidenceDoc(
                    doc_id=results['ids'][0][i],
                    title=results['metadatas'][0][i]['title'],
                    source_url=results['metadatas'][0][i]['url'],
                    snippet=results['documents'][0][i],
                    reliability_score=score
                )
                retrieved_docs.append(doc)
        return retrieved_docs
    
    def clear_cache(self):
        """Reset DB for a new experiment."""
        try:
            self.client.delete_collection("evidence_cache")
        except:
            pass
        self.collection = self.client.get_or_create_collection(
            name="evidence_cache", embedding_function=self.emb_fn
        )