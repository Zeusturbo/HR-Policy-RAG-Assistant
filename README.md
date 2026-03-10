# HR Policy RAG Assistant

HR Policy RAG Assistant is a local AI-powered document Q&A application built with **Python, Streamlit, LangChain, LangGraph, Ollama, and ChromaDB**. It allows users to upload HR policy PDF documents, build a local searchable knowledge base, and ask natural-language questions to get grounded answers with source references.

## Features

- Upload one or more HR policy PDF files
- Extract PDF content locally
- Split documents into chunks for retrieval
- Generate embeddings using **Ollama** with `nomic-embed-text`
- Store embeddings in **ChromaDB**
- Ask HR-related questions in a chat-style interface
- Generate answers using **Qwen2.5:7b** via Ollama
- Show source references with file name and page number
- Return a fallback response when the answer is not found in the uploaded documents
- Fully local setup with no external API dependency

## Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **LangGraph**
- **Ollama**
- **Qwen2.5:7b**
- **nomic-embed-text**
- **ChromaDB**
- **PyMuPDF**

## Project Structure

```bash
hr-policy-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
├── chroma_db/
└── src/
    ├── loader.py
    ├── splitter.py
    ├── embeddings.py
    ├── vectorstore.py
    ├── retriever.py
    ├── graph.py
    └── utils.py
```

## How It Works

1. Upload HR policy PDF files
2. Extract text page by page
3. Split text into chunks
4. Generate embeddings for each chunk using `nomic-embed-text`
5. Store chunks and metadata in ChromaDB
6. Retrieve the most relevant chunks for a user question
7. Use `qwen2.5:7b` to generate an answer based only on retrieved context
8. Display the response along with source references

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Zeusturbo/HR-Policy-RAG-Assistant.git
cd HR-Policy-RAG-Assistant
```

### 2. Create and activate a virtual environment

#### Windows

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama on your machine, then pull the required models:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

## Example Questions

- How many annual leave days are allowed?
- Can employees carry forward unused leave?
- What is the probation period?
- How many remote work days are allowed per week?
- When does private health insurance start?

## Example Use Cases

This project can be adapted for:

- HR policy assistants
- employee handbook Q&A
- internal knowledge assistants
- document-based enterprise chatbots

## Notes

- This project runs fully locally using Ollama
- Best suited for small to medium PDF collections
- Response quality depends on document quality, chunking, retrieval settings, and model performance
- If no relevant information is found, the assistant returns a safe fallback response

## Future Improvements

- Better UI and UX refinements
- Chat history persistence
- Confidence scoring
- Better document management
- Support for more file types
- Admin panel for document upload and collection management

## Author

**Pratheesh Kumar**

## License

This project is for learning, experimentation, and portfolio showcase purposes.

