# ChatRAG: Retrieval-Augmented Generation Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot with a Streamlit UI. It allows you to ingest documents, store their embeddings in SQLite, and ask questions that are answered using both retrieval and OpenAI's GPT models.

## Features
- Add documents and store their embeddings
- Query using natural language and get context-aware answers
- Streamlit UI for easy interaction
- SQLite for local document storage

## Quickstart

### 1. Clone the repository
```
git clone <your-repo-url>
cd ChatRAG
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Set up your environment variables
Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

### 4. Run the Streamlit app
```
streamlit run app_rag_ui.py
```

## Project Structure
- `app_rag.py`: Core RAG logic and database functions
- `app_rag_ui.py`: Streamlit UI for document ingestion and Q&A
- `check_rag_db.py`: Utility to inspect the SQLite database
- `requirements.txt`: Python dependencies
- `rag_docs.db`: SQLite database (auto-created)

## Notes
- Do **not** commit your `.env` file or API keys to public repositories.
- For production, consider using a managed vector database and secure secret management.

---

Feel free to fork and extend!
