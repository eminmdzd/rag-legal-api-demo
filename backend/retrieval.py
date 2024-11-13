import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config


def load_documents_from_directory(directory_path):
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
    return documents


class LegalRetriever:
    def __init__(self):
        # Load FAISS index and embedding model
        self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load documents manually from the specified directory
        print("Loading documents from directory...")
        self.docs = load_documents_from_directory(Config.LEGAL_DATASET_PATH)
        print(f"Loaded {len(self.docs)} documents.")

    def retrieve_documents(self, query, k=5):
        # Embed the query and search for nearest neighbors
        query_embedding = self.embedder.encode([query])[0].astype("float32")
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.docs[i] for i in indices[0]]
