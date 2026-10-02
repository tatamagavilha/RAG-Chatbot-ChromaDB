# RAG Chatbot with ChromaDB
# Retrieval-Augmented Generation System


# Import required libraries
import os
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


DOCUMENTS_FOLDER = "documents_project4"
CHROMA_FOLDER = "chroma_project4_db"
COLLECTION_NAME = "project4_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def create_documents_folder():
    if not os.path.exists(DOCUMENTS_FOLDER):
        os.makedirs(DOCUMENTS_FOLDER)
        print(f"Created folder: {DOCUMENTS_FOLDER}")
        print("Please add .txt files to this folder and run the program again.")
        return False
    return True

# Read all text files from documents_project4
def read_txt_files():
    documents = []

    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCUMENTS_FOLDER, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            if text.strip():
                documents.append((filename, text))

    return documents

# Split documents into smaller chunks
def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks

# Build and store embeddings in ChromaDB
def build_chromadb(documents, embedding_model):
    client = chromadb.PersistentClient(path=CHROMA_FOLDER)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids = []
    chunks = []
    metadatas = []

    chunk_id = 0

    for filename, text in documents:
        file_chunks = chunk_text(text)

        for chunk in file_chunks:
            ids.append(f"chunk_{chunk_id}")
            chunks.append(chunk)
            metadatas.append({"source": filename})
            chunk_id += 1

    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    return collection

# Retrieve relevant context from ChromaDB
def retrieve_context(question, collection, embedding_model, top_k=3):
    question_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        source = sources[i]["source"]
        context_parts.append(f"Source: {source}\n{chunk}")

    return "\n\n---\n\n".join(context_parts)

# Generate answer using Hugging Face Llama model
def generate_answer(question, context, hf_token):
    client = InferenceClient(
        model=HF_MODEL,
        token=hf_token
    )

    system_prompt = """
You are a helpful RAG chatbot.
Answer the user's question using ONLY the provided context.
If the answer is not found in the context, say: "I do not know based on the provided context."
Do not make up information.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=300,
        temperature=0.2
    )

    return response.choices[0].message["content"]

# Main program execution
def main():
    print("Project 4: ChromaDB RAG Chatbot")
    print("--------------------------------")

    if not create_documents_folder():
        return

    documents = read_txt_files()

    if not documents:
        print(f"No .txt files found in {DOCUMENTS_FOLDER}.")
        print("Please add .txt files and run the program again.")
        return

    print(f"Found {len(documents)} text file(s).")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Building ChromaDB vector database...")
    collection = build_chromadb(documents, embedding_model)

    hf_token = input("\nEnter your Hugging Face API key: ").strip()

    while True:
        question = input("\nEnter your question, or type 'exit' to quit: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        context = retrieve_context(question, collection, embedding_model)

        print("\nRetrieved Context:")
        print(context)

        print("\nGenerated Answer:")
        answer = generate_answer(question, context, hf_token)
        print(answer)


if __name__ == "__main__":
    main()
