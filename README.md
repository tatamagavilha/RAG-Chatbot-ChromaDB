# RAG Chatbot with ChromaDB

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions using information retrieved from a collection of local text documents.

Instead of relying only on the language model's existing knowledge, the system searches the user's documents for relevant information and provides the retrieved context to the language model before generating an answer.

The project combines document processing, semantic embeddings, vector search, and a large language model to create a grounded question-answering pipeline.

## Key Features

- Loads `.txt` documents from a local folder
- Splits documents into smaller overlapping chunks
- Generates semantic embeddings using Sentence Transformers
- Stores document embeddings in ChromaDB
- Retrieves the most relevant document chunks for a user's question
- Tracks the source filename for retrieved information
- Uses Llama 3.1 through the Hugging Face Inference API
- Instructs the language model to answer only from retrieved context
- Returns an "I do not know" response when the answer is not supported by the provided context

## How the RAG Pipeline Works

```text
Local Text Documents
        ↓
Document Chunking
        ↓
Sentence Embeddings
        ↓
ChromaDB Vector Database
        ↓
User Question
        ↓
Question Embedding
        ↓
Semantic Retrieval
        ↓
Top Relevant Document Chunks
        ↓
Context + User Question
        ↓
Llama 3.1
        ↓
Grounded Answer
```

## Document Processing

The application reads text files from a local document folder.

Each document is divided into smaller chunks before being stored in the vector database.

The current configuration uses:

```text
Chunk size: 700 characters
Chunk overlap: 100 characters
```

The overlap helps preserve context that may span the boundary between two chunks.

## Semantic Embeddings

The project uses the Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

The model converts document chunks and user questions into numerical embedding vectors that represent semantic meaning.

This allows the system to retrieve information based on similarity in meaning rather than relying only on exact keyword matches.

## Vector Database

The project uses ChromaDB as a persistent vector database.

Each document chunk is stored with:

- A unique chunk ID
- The original text
- Its embedding vector
- Metadata identifying the source document

When a user asks a question, the question is converted into an embedding and compared with the stored document embeddings.

The system retrieves the three most relevant chunks by default.

## Retrieval-Augmented Generation

After retrieval, the relevant document chunks are combined into context for the language model.

The project uses:

```text
meta-llama/Llama-3.1-8B-Instruct
```

through the Hugging Face Inference API.

The model receives both the retrieved context and the user's question.

The system prompt instructs the model to use only the provided context and avoid making up unsupported information.

If the answer cannot be found in the retrieved context, the model is instructed to respond:

```text
I do not know based on the provided context.
```

## Source Tracking

Each chunk stored in ChromaDB includes metadata containing the name of its original source file.

Retrieved context is displayed with its source so that the user can see which document contributed information to the response.

## Technologies Used

- Python
- ChromaDB
- Sentence Transformers
- Hugging Face Inference API
- Llama 3.1 8B Instruct
- Semantic embeddings
- Vector similarity search
- Retrieval-Augmented Generation (RAG)

## Requirements

Install the required Python libraries with:

```bash
pip install -r requirements.txt
```

The project requires:

```text
chromadb
sentence-transformers
huggingface-hub
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add documents

Create or use the document folder expected by the application:

```text
documents_project4
```

Add one or more `.txt` files to this folder.

The application will read these documents, split them into chunks, generate embeddings, and store them in ChromaDB.

### 3. Hugging Face Access

The chatbot uses the Hugging Face Inference API.

When the program starts, it asks for a Hugging Face API token at runtime.

```text
Enter your Hugging Face API key:
```

The API token is not stored directly in the source code.

Users should never commit private API keys or access tokens to a public repository.

### 4. Run the application

Run:

```bash
python project_rag_chatbot.py
```

The exact command should match the name of the Python file in the repository.

## Example Workflow

```text
User adds text documents
        ↓
Application creates document chunks
        ↓
Embeddings are generated
        ↓
Chunks are stored in ChromaDB
        ↓
User asks a question
        ↓
Relevant chunks are retrieved
        ↓
Retrieved context is sent to Llama 3.1
        ↓
The chatbot generates a grounded response
```

## Current Limitations

The current version works with `.txt` documents and rebuilds the ChromaDB collection when the application starts.

Answer quality also depends on the relevance of the retrieved chunks and the information contained in the source documents.

## Future Improvements

Future versions could include:

- PDF and DOCX document support
- Improved document chunking strategies
- Retrieval relevance scoring
- Citations in final chatbot responses
- Conversation history
- A graphical or web-based user interface
- Environment-variable based API credential management
- Support for multiple vector collections
- Evaluation of retrieval and answer quality

## Project Goal

This project demonstrates how Retrieval-Augmented Generation can combine semantic search with a large language model to answer questions using a specific collection of documents.

The system separates retrieval from generation: ChromaDB identifies relevant information, while the language model uses that retrieved context to construct the final response.

This architecture helps reduce unsupported responses and makes language-model applications more useful for document-based question answering.