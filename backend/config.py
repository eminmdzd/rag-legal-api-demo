import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index_file.index")
    LEGAL_DATASET_PATH = os.getenv("LEGAL_DATASET_PATH", "data/legalbenchrag")
    DOCS_PATH = "data/docs.npy"
    DOC_IDS_PATH = "data/doc_ids.npy"
