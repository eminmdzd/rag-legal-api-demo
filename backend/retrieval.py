import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config


class LegalRetriever:
    def __init__(self):
        # Load FAISS index and embedding model
        self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load chunked documents and their IDs
        print("Loading documents and doc_ids...")
        self.docs = np.load(Config.DOCS_PATH, allow_pickle=True)
        self.doc_ids = np.load(Config.DOC_IDS_PATH, allow_pickle=True)
        print(f"Loaded {len(self.docs)} document chunks.")

    def retrieve_documents(self, query, k=5):
        # Embed the query and search for nearest neighbors
        query_embedding = self.embedder.encode([query])[0].astype("float32")
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [(self.docs[i], self.doc_ids[i]) for i in indices[0]]
