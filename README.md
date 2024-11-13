# Legal RAG Project

This project is a Retrieval-Augmented Generation (RAG) API for legal questions using FAISS and OpenAI.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download legal documents

https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&e=1&st=0hu354cq&dl=0

add the "corpus" folder to data folder in the project and rename it to "legalbenchrag"

### 3. Set up the .env file with openai keys

### 4. Run create_faiss_index.py to build and save the FAISS index:

```bash
python create_faiss_index.py
```

### 5. Start the flask server

```bash
python app.py
```
