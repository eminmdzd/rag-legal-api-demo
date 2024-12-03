import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config
import PyPDF2
import os

INDEX_FILE = Config.FAISS_INDEX_PATH
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500


def load_and_chunk_documents(directory_path):
    documents = []
    doc_ids = []  # To keep track of document IDs and chunk numbers
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                # Extract text from PDF
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            content += text
                # Split the document into chunks
                chunks = [
                    content[i : i + CHUNK_SIZE]
                    for i in range(0, len(content), CHUNK_SIZE)
                ]
                documents.extend(chunks)
                doc_ids.extend([(file, idx) for idx in range(len(chunks))])
    return documents, doc_ids


def create_and_save_faiss_index():
    # Load and chunk documents
    print("Loading and chunking documents...")
    docs, doc_ids = load_and_chunk_documents(Config.LEGAL_DATASET_PATH)

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

    # Save the index, docs, and doc_ids
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    np.save(Config.DOCS_PATH, docs)  # Save the docs array
    np.save(Config.DOC_IDS_PATH, doc_ids)
    print(
        f"FAISS index, docs, and doc_ids saved to {INDEX_FILE}, {Config.DOCS_PATH}, and {Config.DOC_IDS_PATH}"
    )


if __name__ == "__main__":
    create_and_save_faiss_index()
