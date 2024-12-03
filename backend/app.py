from typing import List, Any
from flask import Flask, request, jsonify
from retrieval import LegalRetriever
from openai import OpenAI
from config import Config
import tiktoken

# Initialize the Flask app and configurations
app = Flask(__name__)
app.config.from_object(Config)

# Initialize OpenAI client and LegalRetriever
client = OpenAI(api_key=app.config["OPENAI_API_KEY"])
retriever = LegalRetriever()

tokenizer = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text):
    return len(tokenizer.encode(text))


def generate_answer(query):
    # Step 1: Retrieve relevant document chunks
    retrieved = retriever.retrieve_documents(query, k=10)

    # Step 2: Assemble context while managing token limits
    max_context_tokens = 3000  # Adjust based on model's max token limit (e.g., 4096)
    context_texts = []
    total_tokens = 0

    for idx, (chunk, (doc_name, chunk_idx)) in enumerate(retrieved):
        chunk_tokens = count_tokens(chunk)
        if total_tokens + chunk_tokens > max_context_tokens:
            break
        context_texts.append(f"Document {doc_name} - Part {chunk_idx}:\n{chunk}\n")
        total_tokens += chunk_tokens

    context = "\n".join(context_texts)

    # Step 3: Construct the messages for the ChatCompletion API
    messages: List[Any] = [
        {
            "role": "system",
            "content": (
                "You are a professional assistant specialized in international trade law matters. Use the provided documents to answer the question. When referring to the documents, always cite the document name, case names, and part numbers. If the answer to the question comes from a specific case, make sure to cite the full case name. If the answer is not in the provided documents, generate an answer but mention that it is purely AI-generated."
            ),
        },
        {
            "role": "user",
            "content": f"Documents:\n{context}\n\nQuestion: {query}",
        },
    ]

    # Step 4: Call the OpenAI API to generate the answer
    completion = client.chat.completions.create(model="gpt-4o", messages=messages)

    answer = completion.choices[0].message.content.strip()  # type: ignore

    return {
        "rag_context": context,
        "openai_answer": answer,
    }


@app.route("/api/query", methods=["POST"])
def answer_query():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON data"}), 400

    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    # Generate the answer and context information
    result = generate_answer(query)

    return jsonify(
        {
            "query": query,
            "rag_context": result["rag_context"],
            "openai_answer": result["openai_answer"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
