"""
RAG chat app for the Fine-Tuning LLMs PDF.

The notebook builds the index (chunks -> embeddings -> ChromaDB in ./rag_db).
This app only answers questions:
    user input -> query rewriting (LLM) -> embed (tokenize + vector) -> hybrid search -> LLM answer

Run from a terminal in this folder:
    python -m streamlit run app.py
"""

import os
from pathlib import Path

import numpy as np
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_openai import ChatOpenAI

# ---------- Settings (must match the notebook) ----------
BASE_DIR = Path(__file__).parent          # folder of app.py, not the terminal's folder
DB_PATH = BASE_DIR / "rag_db"
COLLECTION_NAME = "fine_tuning_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_NAME = "gpt-4o-mini"

# bge models expect this prefix on short search queries (not on the stored chunks)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

st.set_page_config(page_title="Fine-Tuning LLMs Q&A", page_icon="📘")


# ---------- Load everything once (cached across reruns) ----------
@st.cache_resource(show_spinner="Loading embedding model and database...")
def load_retriever():
    dense_model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=None)

    stored = collection.get(include=["documents"])
    stored_ids = stored["ids"]
    stored_docs = stored["documents"]

    # TF-IDF is fitted on the stored docs, same as the notebook
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english")
    sparse_vectors = tfidf.fit_transform(stored_docs)

    return dense_model, collection, tfidf, sparse_vectors, stored_ids, stored_docs


@st.cache_resource
def get_llm(api_key):
    return ChatOpenAI(model=LLM_NAME, temperature=0, api_key=api_key)


# ---------- Retrieval (same logic as the notebook) ----------
def min_max(scores):
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_search(query, retriever, alpha=0.5, top_k=3):
    dense_model, collection, tfidf, sparse_vectors, stored_ids, stored_docs = retriever

    # Dense: the model tokenizes the query internally, then embeds it
    query_dense = dense_model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)
    # Sparse: transform only, never fit on the query
    query_sparse = tfidf.transform([query])

    results = collection.query(
        query_embeddings=query_dense.tolist(),
        n_results=collection.count(),
        include=["distances"],
    )
    dense_by_id = {
        chunk_id: 1 - distance
        for chunk_id, distance in zip(results["ids"][0], results["distances"][0])
    }
    dense_scores = np.array([dense_by_id[chunk_id] for chunk_id in stored_ids])

    sparse_scores = (sparse_vectors @ query_sparse.T).toarray().ravel()

    hybrid_scores = alpha * min_max(dense_scores) + (1 - alpha) * min_max(sparse_scores)
    top_positions = np.argsort(hybrid_scores)[::-1][:top_k]

    return [
        {"id": stored_ids[pos], "text": stored_docs[pos], "score": float(hybrid_scores[pos])}
        for pos in top_positions
    ]


# ---------- Document map ----------
def extract_section_titles(docs, max_len=120):
    """First two lines of each paragraph = slide title + subtitle."""
    titles = []
    for doc in docs:
        for para in doc.split("\n\n"):
            lines = [line.strip() for line in para.splitlines() if line.strip()]
            if not lines:
                continue
            title = f"{lines[0]} — {lines[1]}" if len(lines) > 1 else lines[0]
            title = title[:max_len]
            if title not in titles:
                titles.append(title)
    return titles


# ---------- Query rewriting ----------
def rewrite_query(llm, user_input, section_titles):
    titles_text = "\n".join(f"- {t}" for t in section_titles)

    prompt = f"""You rewrite a user's input into one clear search question
for a document about fine-tuning large language models.

These are the section titles of the document:
{titles_text}

Rules:
- Fix spelling mistakes.
- The user may use an informal or slightly wrong name for a topic. If their words mean
  the same thing as a term in the titles, use the document's term and its abbreviation.
- If the input is just a topic, turn it into a question about that topic.
- If the input is not related to any title, only clean it up. Do not force it onto a title.
Return ONLY the question, nothing else.

User input: {user_input}"""
    return llm.invoke(prompt).content.strip()


# ---------- Generation ----------
def build_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in retrieved_chunks)

    return f"""You are a helpful tutor answering questions about a document on fine-tuning LLMs.
The context below was retrieved by a search engine as the most relevant parts of the document.

How to answer:
1. Find every sentence in the context related to the question, even if it uses different words
   (for example "assign", "choose", "set" and "pick" a value all mean the same thing).
2. Answer using only those sentences. Do not add outside knowledge.
3. If the document covers the question only partly, answer the covered part,
   then add one line that starts with "Not covered in the document:".
4. If no sentence in the context is related to the question at all, reply with exactly
   this sentence and nothing else: "I don't know based on the document."
   Never add this sentence after an answer.
5. End your answer with the chunk ids you used, like [chunk_11].

Context:
{context}

Question: {query}

Answer:"""

def stream_answer(llm, prompt):
    for chunk in llm.stream(prompt):
        yield chunk.content


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)} chunks)"):
        for s in sources:
            st.markdown(f"**{s['id']}** · score {s['score']:.2f}")
            st.caption(s["text"][:300].replace("\n", " ") + "...")


# ---------- Sidebar ----------
load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.header("Settings")

    if not api_key:
        api_key = st.text_input("OpenAI API key", type="password")

    top_k = st.slider("Chunks to retrieve (top_k)", 1, 5, 3)
    alpha = st.slider(
        "Search balance (alpha)", 0.0, 1.0, 0.5, 0.1,
        help="0 = keywords only (TF-IDF), 1 = meaning only (embeddings)",
    )

    use_rewrite = st.toggle(
        "Rewrite my question before searching", value=True,
        help="An extra LLM call that fixes spelling and uses the document's technical terms",
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------- Main ----------
st.title("📘 Fine-Tuning LLMs Q&A")
st.caption("Answers come only from the Fine_Tuning_LLMs.pdf document.")

if not api_key:
    st.info("Add your OpenAI API key in the sidebar, or put it in a .env file next to app.py.")
    st.stop()

try:
    retriever = load_retriever()
except Exception as e:
    st.error(
        f"Could not load the database from {DB_PATH}. "
        f"Run the notebook first so it creates the '{COLLECTION_NAME}' collection.\n\n{e}"
    )
    st.stop()

llm = get_llm(api_key)
section_titles = extract_section_titles(retriever[5])   # retriever[5] = stored_docs

with st.sidebar:
    with st.expander(f"Topics in the document ({len(section_titles)})"):
        for t in section_titles:
            st.caption(t)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("searched_for"):
            st.caption(f"Searched for: {msg['searched_for']}")
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# New question
if query := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            search_query = query
            if use_rewrite:
                try:
                    search_query = rewrite_query(llm, query, section_titles)
                except Exception:
                    pass  # if rewriting fails, search with the original input
            sources = hybrid_search(search_query, retriever, alpha=alpha, top_k=top_k)

        if search_query != query:
            st.caption(f"Searched for: {search_query}")

        prompt = build_prompt(search_query, sources)

        try:
            answer = st.write_stream(stream_answer(llm, prompt))
        except Exception as e:
            answer = f"The model call failed: {e}"
            st.error(answer)

        show_sources(sources)

        with st.expander("Prompt sent to the LLM"):
            st.code(prompt, language=None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "searched_for": search_query if search_query != query else None,
    })
