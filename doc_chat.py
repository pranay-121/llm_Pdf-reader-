import os
import json
from typing import List, Dict
import nbformat
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

import streamlit as st
st.write(st.secrets)

import streamlit as st
import os

api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key



# ---------- CONFIG ----------
DOCS_FOLDER = "pdf"  # put your PDFs, txts, ipynb here
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 5  # how many chunks to retrieve
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 100  # words
# ----------------------------


def load_env_and_client():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Set it in .env or environment variables.")
    client = Groq(api_key=api_key)
    return client


# --------- DOCUMENT LOADING ---------
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def read_ipynb(path: str) -> str:
    """Extracts markdown + code (as plain text) from .ipynb."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        nb = nbformat.read(f, as_version=4)

    parts = []
    for cell in nb.cells:
        src = cell.get("source", "")
        if not src:
            continue
        if cell.cell_type == "markdown":
            parts.append(src)
        elif cell.cell_type == "code":
            # You can choose to include or skip code
            parts.append("# Code cell:\n" + src)
    return "\n\n".join(parts)


def load_all_documents(folder: str) -> List[Dict]:
    docs = []
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()

            try:
                if ext in [".txt", ".md"]:
                    text = read_txt(path)
                elif ext in [".pdf"]:
                    text = read_pdf(path)
                elif ext in [".ipynb"]:
                    text = read_ipynb(path)
                else:
                    # unsupported type, skip silently
                    continue
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

            if text.strip():
                docs.append(
                    {
                        "source": path,
                        "text": text,
                    }
                )
    return docs


# --------- CHUNKING & EMBEDDING ---------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_index(docs: List[Dict], embed_model_name: str):
    if not docs:
        return None, None, None

    # load embedding model
    embedder = SentenceTransformer(embed_model_name)

    all_chunks = []
    meta = []  # store which doc each chunk came from

    for d in docs:
        chunks = chunk_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for ch in chunks:
            all_chunks.append(ch)
            meta.append({"source": d["source"]})

    if not all_chunks:
        return None, None, None

    embeddings = embedder.encode(all_chunks, show_progress_bar=True)

    nn = NearestNeighbors(
        n_neighbors=min(TOP_K, len(all_chunks)),
        metric="cosine",
    )
    nn.fit(embeddings)

    return nn, embeddings, {"chunks": all_chunks, "meta": meta, "embedder": embedder}


def retrieve_relevant_chunks(query: str, index, embeddings, store, top_k: int = TOP_K):
    if index is None or embeddings is None or store is None:
        return []

    embedder = store["embedder"]
    q_emb = embedder.encode([query])
    distances, indices = index.kneighbors(q_emb, n_neighbors=min(top_k, len(store["chunks"])))

    idxs = indices[0]
    results = []
    for i in idxs:
        results.append(
            {
                "text": store["chunks"][i],
                "source": store["meta"][i]["source"],
            }
        )
    return results


# --------- GROQ LLM CALL ---------
def ask_groq(client: Groq, question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a question-answering assistant. "
                "Use ONLY the given context from the user's documents to answer. "
                "If the answer is not in the context, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based ONLY on the context above.",
        },
    ]

    resp = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# --------- STREAMLIT APP ---------
def main():
    st.set_page_config(page_title="Docs Q&A Chatbot (Groq)", page_icon="📚")
    st.title("📚 Document Q&A Chatbot (Groq)")

    st.markdown(
        """
This chatbot answers questions **only** from the documents in the `docs/` folder.

Supported file types:
- `.pdf`
- `.txt` / `.md`
- `.ipynb` (Jupyter notebooks – markdown + code text)
"""
    )

    # Load Groq client
    try:
        client = load_env_and_client()
    except Exception as e:
        st.error(f"Problem with GROQ_API_KEY: {e}")
        st.stop()

    # Build / cache index
    if "index_built" not in st.session_state:
        with st.spinner("Loading documents and building index..."):
            docs = load_all_documents(DOCS_FOLDER)
            if not docs:
                st.warning(f"No readable documents found in folder `{DOCS_FOLDER}`.")
                st.stop()

            index, embeddings, store = build_index(docs, EMBED_MODEL_NAME)
            if index is None:
                st.warning("Failed to build index from documents.")
                st.stop()

            st.session_state.index_built = True
            st.session_state.index = index
            st.session_state.embeddings = embeddings
            # Note: SentenceTransformer isn't JSON-serializable, keep in session directly
            st.session_state.store = store

    st.success("Documents loaded and index ready ✅")

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("Ask something about your documents...")

    if question:
        # Display user message
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve context
        chunks = retrieve_relevant_chunks(
            question,
            st.session_state.index,
            st.session_state.embeddings,
            st.session_state.store,
            top_k=TOP_K,
        )

        if not chunks:
            answer = "I couldn't retrieve any relevant information from the documents."
        else:
            context = "\n\n".join(
                [f"[From: {c['source']}]\n{c['text']}" for c in chunks]
            )
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_groq(client, question, context)
                    except Exception as e:
                        answer = f"Error calling Groq API: {e}"
                st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()
