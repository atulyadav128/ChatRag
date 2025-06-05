import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from app_rag import setup_db, add_document, search_similar, ask_llm, EMBEDDING_MODEL

# Initialize DB and model
setup_db()
model = SentenceTransformer(EMBEDDING_MODEL)

st.set_page_config(page_title="RAG Demo", page_icon="📄")
st.title("RAG (Retrieval-Augmented Generation) Demo")

st.header("1. Add a Document")
doc_text = st.text_area("Paste or type your document here:")
if st.button("Add Document"):
    if doc_text.strip():
        try:
            add_document(model, doc_text)
            st.success("Document added!")
        except Exception as e:
            st.error(f"Error adding document: {e}")
    else:
        st.warning("Please enter some text.")

st.header("2. Ask a Question")
query = st.text_input("Your question:")
if st.button("Get Answer"):
    if query.strip():
        docs = search_similar(model, query)
        if not docs:
            st.info("No relevant documents found.")
        else:
            context = "\n\n".join(docs)
            answer = ask_llm(context, query)
            st.markdown(f"**AI answer:** {answer}")
            st.markdown("---")
            st.markdown("**Context used:**")
            for d in docs:
                st.markdown(f"> {d}")
    else:
        st.warning("Please enter a question.")
