import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

INDEX_FILE = Config.FAISS_INDEX_PATH
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents_from_directory(directory_path):
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
    return documents


def create_and_save_faiss_index():
    # Load documents from the specified directory
    print("Loading documents...")
    docs = load_documents_from_directory(Config.LEGAL_DATASET_PATH)

    # Initialize the embedding model
    print("Embedding documents...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(docs, convert_to_tensor=False)

    # Ensure embeddings are in the correct format
    embeddings = np.array(embeddings, dtype="float32")
    if len(embeddings.shape) != 2:
        raise ValueError(
            "Embeddings array must be 2D with shape [num_documents, dimension]."
        )

    num_documents, dimension = embeddings.shape
    print(
        f"Embeddings shape: {embeddings.shape} (num_documents={num_documents}, dimension={dimension})"
    )

    # Create a FAISS index
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # type: ignore

    # Save the index
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS index created and saved to {INDEX_FILE}")


if __name__ == "__main__":
    create_and_save_faiss_index()
