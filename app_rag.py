import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# ------- Setup -------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DB_PATH = "rag_docs.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ------- Connect DB -------
def get_conn():
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"[DB ERROR] Could not connect to DB: {e}")
        raise

def setup_db():
    try:
        conn = get_conn()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT,
                embedding BLOB
            );
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] Could not set up DB: {e}")
        raise

# ------- Embedding -------
def embed_text(model, text):
    try:
        emb = model.encode([text])[0]
        return emb.astype(np.float32).tobytes()
    except Exception as e:
        print(f"[EMBEDDING ERROR] Could not embed text: {e}")
        raise

def add_document(model, text):
    try:
        emb = embed_text(model, text)
        conn = get_conn()
        conn.execute(
            "INSERT INTO documents (content, embedding) VALUES (?, ?)", (text, emb)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] Could not add document: {e}")
        raise

def get_all_embeddings():
    try:
        conn = get_conn()
        docs = conn.execute("SELECT id, content, embedding FROM documents").fetchall()
        conn.close()
        ids, contents, embs = [], [], []
        for i, c, e in docs:
            ids.append(i)
            contents.append(c)
            embs.append(np.frombuffer(e, dtype=np.float32))
        return ids, contents, np.vstack(embs) if embs else np.array([])
    except Exception as e:
        print(f"[DB ERROR] Could not fetch embeddings: {e}")
        return [], [], np.array([])

# ------- Similarity Search -------
def search_similar(model, query, top_k=3):
    try:
        ids, contents, embs = get_all_embeddings()
        if embs.shape[0] == 0:
            return []
        q_emb = model.encode([query])[0]
        sims = np.dot(embs, q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        top_idxs = sims.argsort()[-top_k:][::-1]
        return [contents[i] for i in top_idxs]
    except Exception as e:
        print(f"[SEARCH ERROR] Could not search similar docs: {e}")
        return []

# ------- Call LLM -------
def ask_llm(context, question):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant using provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] Could not get response from LLM: {e}")
        return "[Error: Could not get response from LLM]"

# ------- Demo Flow -------
if __name__ == "__main__":
    setup_db()
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 1. Ingest a document
    print("Add new doc: (type or paste, then Enter twice to finish)")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    doc = "\n".join(lines)
    if doc.strip():
        add_document(model, doc)
        print("Document added!\n")

    # 2. Query
    print("Ask a question (Enter to skip):")
    query = input()
    if query.strip():
        docs = search_similar(model, query)
        if not docs:
            print("No docs found.")
        else:
            context = "\n\n".join(docs)
            answer = ask_llm(context, query)
            print("AI answer:", answer)
