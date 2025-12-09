import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
from typing import List
import uuid
from fire_debate.schemas.evidence import EvidenceDoc

class EvidenceRetriever:
    def __init__(self, config):
        self.cfg = config['retrieval']
        print(f"📚 Initializing Librarian with {self.cfg['embedding_model']}...")
        
        # Hugging Face Embedding Function
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.cfg['embedding_model']
        )
        
        self.client = chromadb.PersistentClient(path=self.cfg['db_path'])
        self.collection = self.client.get_or_create_collection(
            name="evidence_cache",
            embedding_function=self.emb_fn
        )
        self.ddgs = DDGS()

    def search_web(self, query: str) -> List[EvidenceDoc]:
        """Fetches raw data from DuckDuckGo."""
        results = self.ddgs.text(query, max_results=self.cfg['max_search_results'])
        docs = []
        if results:
            for r in results:
                doc = EvidenceDoc(
                    doc_id=str(uuid.uuid4())[:8],
                    source_url=r['href'],
                    title=r['title'],
                    snippet=r['body'],
                    reliability_score=0.5 # Baseline score
                )
                docs.append(doc)
        return docs

    def index_documents(self, docs: List[EvidenceDoc]):
        """Saves documents to ChromaDB for semantic search."""
        if not docs: return
        
        ids = [d.doc_id for d in docs]
        texts = [d.snippet for d in docs]
        metadatas = [{"url": d.source_url, "title": d.title} for d in docs]
        
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def retrieve_context(self, query: str) -> List[EvidenceDoc]:
        """Semantic Search: Finds best evidence for the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=self.cfg['top_k_evidence']
        )
        
        retrieved_docs = []
        # Parse Chroma's complex return format
        if results['ids']:
            count = len(results['ids'][0])
            for i in range(count):
                doc = EvidenceDoc(
                    doc_id=results['ids'][0][i],
                    title=results['metadatas'][0][i]['title'],
                    source_url=results['metadatas'][0][i]['url'],
                    snippet=results['documents'][0][i],
                    # Convert distance to similarity score approx
                    reliability_score=1.0 - max(0, results['distances'][0][i]) 
                )
                retrieved_docs.append(doc)
        return retrieved_docs
    
    def clear_cache(self):
        """Reset DB for a new experiment."""
        self.client.delete_collection("evidence_cache")
        self.collection = self.client.get_or_create_collection(
            name="evidence_cache", embedding_function=self.emb_fn
        )