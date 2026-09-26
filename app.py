"""
HCIA-AI Study Assistant (Streamlit app).

The notebook builds everything (rag_db/ and question_bank.json). This app only READS them.
Flow:  question → hybrid search → grounded explanation → "Quiz me on this" → graded quiz

Run from a terminal in this folder:
    python -m streamlit run app.py
"""

import os
import re
import html
import json
import time
import base64
from pathlib import Path

import numpy as np
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_openai import ChatOpenAI

# ---------- Paths + settings (must match the notebook) ----------
BASE_DIR = Path(__file__).parent          # folder of app.py, not the terminal's folder
DB_PATH = BASE_DIR / "rag_db"
BANK_JSON = BASE_DIR / "question_bank.json"
ASSETS_DIR = BASE_DIR / "assets"

COLLECTION_NAME = "lectures"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_NAME = "gpt-4o-mini"
ALPHA = 0.5
TOP_K = 3
N_QUIZ_QUESTIONS = 3

EXAMPLE_QUESTIONS = [
    "What is QLoRA?",
    "How does K-means choose the centroids?",
    "Why do we scale features before KNN?",
]

st.set_page_config(page_title="HCIA-AI Study Assistant", page_icon="📘", layout="wide")
load_dotenv(BASE_DIR / ".env", override=True)


# =====================================================================
# 1) Load everything ONCE (Streamlit re-runs this file on every click)
# =====================================================================
@st.cache_resource(show_spinner="Loading the lectures index...")
def load_resources():
    try:
        # From the local cache (the notebook already downloaded it): no internet check → fast, works offline
        dense_model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except OSError:
        # Not in the cache yet (first run on a new computer) → download it once
        dense_model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=None)
    stored = collection.get(include=["documents", "metadatas"])

    # TF-IDF is fit on the stored chunks only (the query is only transformed)
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english")
    sparse_vectors = tfidf.fit_transform(stored["documents"])

    with open(BANK_JSON, encoding="utf-8") as f:
        bank = json.load(f)
    bank_texts = []
    for q in bank:
        bank_texts.append(q["question"])
    bank_vectors = dense_model.encode(bank_texts, normalize_embeddings=True)

    return {
        "dense_model": dense_model,
        "collection": collection,
        "stored_ids": stored["ids"],
        "stored_docs": stored["documents"],
        "stored_metas": stored["metadatas"],
        "tfidf": tfidf,
        "sparse_vectors": sparse_vectors,
        "bank": bank,
        "bank_vectors": bank_vectors,
    }


@st.cache_resource
def load_llms():
    explain_llm = ChatOpenAI(model=LLM_NAME, temperature=0)
    quiz_llm = ChatOpenAI(model=LLM_NAME, temperature=0.7).bind(response_format={"type": "json_object"})
    return explain_llm, quiz_llm


# =====================================================================
# 2) Retrieval + explanation (same logic as notebook Sections 5 and 6)
# =====================================================================
def min_max(scores):
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_search(query, res, alpha=ALPHA, top_k=TOP_K):
    query_dense = res["dense_model"].encode([query], normalize_embeddings=True)
    query_sparse = res["tfidf"].transform([query])

    # 1) Dense scores from ChromaDB (cosine distance → similarity)
    results = res["collection"].query(
        query_embeddings=query_dense.tolist(),
        n_results=res["collection"].count(),
        include=["distances"],
    )
    dense_by_id = {}
    for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
        dense_by_id[chunk_id] = 1 - distance
    dense_scores = np.array([dense_by_id[chunk_id] for chunk_id in res["stored_ids"]])

    # 2) Sparse scores from TF-IDF
    sparse_scores = (res["sparse_vectors"] @ query_sparse.T).toarray().ravel()

    # 3) Normalize + combine, 4) top results
    hybrid_scores = alpha * min_max(dense_scores) + (1 - alpha) * min_max(sparse_scores)
    top_positions = np.argsort(hybrid_scores)[::-1][:top_k]

    top_chunks = []
    for pos in top_positions:
        top_chunks.append({
            "id": res["stored_ids"][pos],
            "lecture": res["stored_metas"][pos]["lecture"],
            "page": res["stored_metas"][pos]["page"],
            "text": res["stored_docs"][pos],
            "score": float(hybrid_scores[pos]),
        })
    return top_chunks


def build_prompt(query, retrieved_chunks):
    context_parts = []
    for c in retrieved_chunks:
        context_parts.append(f"[{c['lecture']} p.{c['page']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful tutor helping a student study an AI / Machine Learning course.
The context below contains the lecture slides most relevant to the question.

How to answer:
1. Find every sentence in the context related to the question, even if it uses different words.
2. Explain clearly using only those sentences. Do not add outside knowledge.
3. If the slides cover the question only partly, explain what they cover,
   then add one line saying what they do not cover.
4. Reply "I don't know based on the lectures." ONLY if nothing in the context is related.
5. Cite the slides you used, like [K-means p.12].

Context:
{context}

Question: {query}

Answer:"""


# =====================================================================
# 3) Quiz (same logic as notebook Section 7)
# =====================================================================
QUIZ_JSON_FORMAT = """{
  "questions": [
    {
      "type": "single",
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": ["B"],
      "explanation": "...",
      "source": "one of the slide ids above"
    }
  ]
}"""


def get_style_examples(topic, res, n=3):
    topic_vector = res["dense_model"].encode([topic], normalize_embeddings=True)[0]
    similarities = res["bank_vectors"] @ topic_vector
    top_positions = np.argsort(similarities)[::-1][:n]

    examples = []
    for pos in top_positions:
        examples.append(res["bank"][pos])
    return examples


def build_quiz_prompt(retrieved_chunks, style_examples, n_questions=N_QUIZ_QUESTIONS):
    slide_parts = []
    for c in retrieved_chunks:
        slide_parts.append(f"[slide id: {c['id']}]\n{c['text']}")
    slides = "\n\n---\n\n".join(slide_parts)

    example_parts = []
    for q in style_examples:
        lines = [f"({q['type']}) {q['question']}"]
        for letter, option_text in q["options"].items():
            lines.append(f"{letter}. {option_text}")
        example_parts.append("\n".join(lines))
    examples = "\n\n".join(example_parts)

    return f"""You write multiple-choice exam questions to help a student revise an AI / Machine Learning course.

Slides (the ONLY source of content you may use):
{slides}

Style examples from the Huawei HCIA-AI exam (copy their STYLE only, NOT their content; they may be about other topics):
{examples}

Rules:
1. Write exactly {n_questions} questions. Every question and every correct answer must come from the slides above. No outside knowledge.
2. Each question has exactly 4 options: A, B, C, D. Wrong options must be plausible but clearly wrong according to the slides.
3. "type" is "single" (exactly 1 correct letter) or "multiple" (2 or 3 correct letters, "select all that apply").
   At least 1 question must be "multiple".
4. "answer" is always a list of letters, e.g. ["B"] or ["A", "C"].
5. "explanation": 1-2 sentences explaining the correct answer using the slides.
6. "source": the slide id the answer comes from, e.g. "K-means_p21" (only the id: no brackets, no "slide id:").

Return ONLY a JSON object in this format:
{QUIZ_JSON_FORMAT}"""


def clean_source(source):
    # The LLM sometimes copies the whole label: "[slide id: KNN_p14]" → "KNN_p14"
    if not isinstance(source, str):
        return source
    source = source.strip()
    source = source.strip("[]")
    source = source.replace("slide id:", "")
    return source.strip()


def check_question(q, allowed_ids):
    """Returns a list of problems. Empty list = the question is valid."""
    if not isinstance(q, dict):
        return ["not a JSON object"]

    problems = []

    q_type = q.get("type")
    if q_type not in ["single", "multiple"]:
        problems.append(f"bad type: {q_type}")

    if not q.get("question"):
        problems.append("empty question")

    options = q.get("options")
    if not isinstance(options, dict):
        options = {}
    if sorted(options.keys()) != ["A", "B", "C", "D"]:
        problems.append("options must be exactly A, B, C, D")

    answer = q.get("answer")
    if not isinstance(answer, list) or len(answer) == 0:
        problems.append("answer must be a non-empty list")
        answer = []

    seen_letters = []
    for letter in answer:
        if not isinstance(letter, str) or letter not in options:
            problems.append(f"answer letter {letter} is not an option")
        elif letter in seen_letters:
            problems.append(f"answer letter {letter} is repeated")
        seen_letters.append(letter)

    if q_type == "single" and len(answer) != 1:
        problems.append("single question must have exactly 1 answer")
    if q_type == "multiple" and len(answer) < 2:
        problems.append("multiple question must have 2+ answers")

    if q.get("source") not in allowed_ids:
        problems.append(f"unknown source: {q.get('source')}")

    return problems


def generate_quiz(query, retrieved_chunks, res, quiz_llm):
    """Returns (valid_questions, reasons). reasons = why questions were dropped, shown in the app."""
    style_examples = get_style_examples(query, res)
    prompt = build_quiz_prompt(retrieved_chunks, style_examples)
    response = quiz_llm.invoke(prompt)

    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        return [], ["The LLM did not return valid JSON"]

    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        return [], ["The JSON does not contain a 'questions' list"]

    allowed_ids = []
    for c in retrieved_chunks:
        allowed_ids.append(c["id"])

    valid_questions = []
    reasons = []
    for i, q in enumerate(data["questions"], start=1):
        if isinstance(q, dict):
            q["source"] = clean_source(q.get("source"))
        problems = check_question(q, allowed_ids)
        if problems:
            reasons.append(f"Dropped question {i}: {problems}")
        else:
            valid_questions.append(q)
    return valid_questions, reasons


def grade_answer(chosen, answer):
    return set(chosen) == set(answer)


# =====================================================================
# 4) Small UI helpers (HTML pieces styled by assets/style.css)
# =====================================================================
def svg_as_img(path, css_class):
    # The SVG is embedded as base64 so its own animations (blink, float) keep working
    svg_bytes = Path(path).read_bytes()
    encoded = base64.b64encode(svg_bytes).decode()
    return f'<img class="{css_class}" src="data:image/svg+xml;base64,{encoded}" alt="Study buddy">'


def highlight_citations(answer):
    # "[K-means p.21]" → "`K-means p.21`" so the CSS shows each citation as a small chip
    return re.sub(r"\[([^\[\]]+? p\.\d+)\]", r"`\1`", answer)


def top_bar(stage, n_lectures, n_slides):
    explain_class = "seg active" if stage == "explain" else "seg"
    quiz_class = "seg active" if stage == "quiz" else "seg"
    st.markdown(f"""
<div class="topbar">
  <div class="brand"><span class="logo">AI</span><span>Study Assistant</span></div>
  <div class="segments"><span class="{explain_class}">Explain</span><span class="{quiz_class}">Quiz</span></div>
  <div class="chip">{n_lectures} lectures · {n_slides} slides</div>
</div>""", unsafe_allow_html=True)


def task_card(title, lines, floating=False):
    body = "<br>".join(lines)
    css_class = "task-card floating" if floating else "task-card"
    st.markdown(f"""
<div class="{css_class}">
  <div class="task-title"><span>Task</span> {title}</div>
  <div class="task-body">{body}</div>
</div>""", unsafe_allow_html=True)


def show_sources(chunks):
    with st.expander(f"📄 Slides used ({len(chunks)})"):
        for c in chunks:
            st.markdown(f"**{c['lecture']} · p.{c['page']}**  <span class='score'>score {c['score']:.2f}</span>",
                        unsafe_allow_html=True)
            slide_text = c["text"].split("\n", 1)[-1]      # drop the "[Lecture]" line we added for retrieval
            st.caption(slide_text[:500])


def option_row(letter, text, css_class, tag):
    # html.escape: option text from the LLM may contain "<" or ">" (e.g. "x < 0")
    tag_html = f"<span class='opt-tag'>{tag}</span>" if tag else ""
    return f"<div class='opt {css_class}'><b>{letter}.</b> {html.escape(text)}{tag_html}</div>"


def show_answer_review(q, chosen):
    """All 4 options: chosen + right = green, chosen + wrong = red, right but not chosen = green outline."""
    rows = []
    for letter, text in q["options"].items():
        is_right = letter in q["answer"]
        is_chosen = letter in chosen
        if is_right and is_chosen:
            rows.append(option_row(letter, text, "right", "✓ Your answer"))
        elif is_chosen:
            rows.append(option_row(letter, text, "wrong", "✗ Your answer"))
        elif is_right:
            rows.append(option_row(letter, text, "right missed", "✓ Correct answer"))
        else:
            rows.append(option_row(letter, text, "", ""))
    st.markdown("".join(rows), unsafe_allow_html=True)


def find_chunk(chunk_id):
    for c in st.session_state.chunks:
        if c["id"] == chunk_id:
            return c
    return None


# =====================================================================
# 5) Page
# =====================================================================
css = (ASSETS_DIR / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---------- Session state: what must survive the re-run after every click ----------
if "question" not in st.session_state:
    st.session_state.question = None       # the last question asked
    st.session_state.answer = None         # the explanation
    st.session_state.chunks = []           # retrieved slides → reused by the quiz
    st.session_state.quiz = []             # generated questions
    st.session_state.quiz_round = 0        # new number for every quiz → fresh widget keys
    st.session_state.submitted = False
    st.session_state.chosen = {}           # question index → chosen letters
    st.session_state.timing = {}

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY was not found. Put it in a `.env` file next to app.py, then restart the app.")
    st.stop()

try:
    res = load_resources()
except Exception as e:
    st.error(f"Could not load `rag_db/` or `question_bank.json`. Run the notebook first.\n\n{e}")
    st.stop()

explain_llm, quiz_llm = load_llms()

n_slides = len(res["stored_ids"])
lectures = set()
for meta in res["stored_metas"]:
    lectures.add(meta["lecture"])

if st.session_state.quiz:
    stage = "quiz"
else:
    stage = "explain"
top_bar(stage, len(lectures), n_slides)

# ---------- Input: typed question or an example button ----------
typed_question = st.chat_input("Ask about a lecture topic (in English)...")
new_question = typed_question

if st.session_state.question is None:
    # ---------------- Home screen ----------------
    st.markdown("""
<div class="hero">
  <div class="badge"><span class="dot"></span>Answers only from your NTI lecture slides</div>
  <h1>Turn Your <span class="hl">Lectures</span> Into<br>Answers &amp; <span class="hl">Quizzes</span></h1>
  <p>Explained from the slides, cited by page, then a quiz in the Huawei HCIA-AI exam style.</p>
</div>""", unsafe_allow_html=True)

    background_ids = "  ·  ".join(res["stored_ids"][:60])
    st.markdown(f"""
<div class="stage">
  <div class="code-rain">{background_ids}</div>
  {svg_as_img(ASSETS_DIR / 'study_buddy.svg', 'buddy')}
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='try-label'>Try one:</div>", unsafe_allow_html=True)
    # empty columns on both sides center the 3 example buttons
    _, col1, col2, col3, _ = st.columns([0.6, 1, 1.7, 1.6, 0.4], gap="small")
    for col, example in zip([col1, col2, col3], EXAMPLE_QUESTIONS):
        if col.button(example):
            new_question = example

    task_card("Ready", [f"{len(lectures)} lectures indexed", "Ask in English for the best results"], floating=True)

# ---------- A new question: search + explain, then reset the quiz ----------
if new_question:
    with st.spinner("Searching the slides and writing the explanation..."):
        t0 = time.time()
        chunks = hybrid_search(new_question, res)
        t1 = time.time()
        try:
            answer = explain_llm.invoke(build_prompt(new_question, chunks)).content
        except Exception as e:
            answer = None
            st.error(f"The LLM call failed: {e}")
        t2 = time.time()

    if answer is not None:
        st.session_state.question = new_question
        st.session_state.answer = answer
        st.session_state.chunks = chunks
        st.session_state.quiz = []
        st.session_state.submitted = False
        st.session_state.chosen = {}
        st.session_state.timing = {"search": t1 - t0, "answer": t2 - t1}
        st.rerun()

if st.session_state.question is not None:
    # ---------------- Answer screen ----------------
    left, right = st.columns([1, 2.3], gap="large")

    with left:
        st.markdown(f"<div class='stage small'>{svg_as_img(ASSETS_DIR / 'study_buddy.svg', 'buddy')}</div>",
                    unsafe_allow_html=True)
        lecture_names = []
        for c in st.session_state.chunks:
            if c["lecture"] not in lecture_names:
                lecture_names.append(c["lecture"])
        if stage == "quiz":
            task_card("Quiz", [f"{len(st.session_state.quiz)} questions", "From: " + ", ".join(lecture_names)])
        else:
            task_card("Explain", [f"{len(st.session_state.chunks)} slides found", "From: " + ", ".join(lecture_names)])

    with right:
        st.markdown(f"<div class='question-bubble'>{st.session_state.question}</div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(highlight_citations(st.session_state.answer))
            show_sources(st.session_state.chunks)

        if not st.session_state.quiz:
            if st.button("📝 Quiz me on this", type="primary"):
                quiz = []
                all_reasons = []
                with st.spinner("Writing questions from these slides..."):
                    # up to 2 tries: temperature 0.7 → the second try gives different questions
                    for attempt in [1, 2]:
                        try:
                            quiz, reasons = generate_quiz(st.session_state.question, st.session_state.chunks, res, quiz_llm)
                        except Exception as e:
                            quiz, reasons = [], [f"Quiz generation failed: {e}"]
                        for reason in reasons:
                            all_reasons.append(f"Try {attempt}: {reason}")
                            print(f"Try {attempt}: {reason}")      # also in the terminal
                        if quiz:
                            break
                if quiz:
                    st.session_state.quiz = quiz
                    st.session_state.quiz_round += 1
                    st.session_state.submitted = False
                    st.session_state.chosen = {}
                    st.rerun()
                else:
                    st.warning("Could not generate a quiz this time. Please click again.")
                    with st.expander("Why? (details)"):
                        for reason in all_reasons:
                            st.caption(reason)

        # ---------------- Quiz ----------------
        if st.session_state.quiz and not st.session_state.submitted:
            with st.form(f"quiz_{st.session_state.quiz_round}"):
                st.markdown("#### 📝 Quiz")
                for i, q in enumerate(st.session_state.quiz):
                    key = f"r{st.session_state.quiz_round}_q{i}"
                    st.markdown(f"**Q{i + 1}. {q['question']}**")

                    if q["type"] == "single":
                        st.radio(
                            "Choose one answer",
                            options=list(q["options"].keys()),
                            format_func=lambda letter, q=q: f"{letter}. {q['options'][letter]}",
                            index=None,
                            key=key,
                        )
                    else:
                        st.caption("Select all that apply")
                        for letter, option_text in q["options"].items():
                            st.checkbox(f"{letter}. {option_text}", key=f"{key}_{letter}")

                submitted = st.form_submit_button("Submit answers", type="primary")

            if submitted:
                chosen = {}
                for i, q in enumerate(st.session_state.quiz):
                    key = f"r{st.session_state.quiz_round}_q{i}"
                    if q["type"] == "single":
                        picked = st.session_state.get(key)
                        if picked is None:
                            chosen[i] = []
                        else:
                            chosen[i] = [picked]
                    else:
                        chosen[i] = []
                        for letter in q["options"]:
                            if st.session_state.get(f"{key}_{letter}"):
                                chosen[i].append(letter)
                st.session_state.chosen = chosen
                st.session_state.submitted = True
                st.rerun()

        # ---------------- Results ----------------
        if st.session_state.quiz and st.session_state.submitted:
            quiz = st.session_state.quiz
            score = 0
            for i, q in enumerate(quiz):
                if grade_answer(st.session_state.chosen[i], q["answer"]):
                    score += 1

            st.markdown(f"<div class='score-card'>Score <b>{score} / {len(quiz)}</b></div>", unsafe_allow_html=True)

            for i, q in enumerate(quiz):
                chosen = st.session_state.chosen[i]
                is_correct = grade_answer(chosen, q["answer"])
                with st.container(border=True):
                    mark = "✅" if is_correct else "❌"
                    st.markdown(f"{mark} **Q{i + 1}. {q['question']}**")
                    if not chosen:
                        st.caption("You did not answer this question.")
                    show_answer_review(q, chosen)
                    st.caption(q["explanation"])

                    if not is_correct:
                        slide = find_chunk(q["source"])
                        if slide is not None:
                            with st.expander(f"📖 Review this slide: {slide['lecture']} p.{slide['page']}"):
                                st.caption(slide["text"].split("\n", 1)[-1])

            if st.button("🔄 New quiz on this topic"):
                st.session_state.quiz = []
                st.session_state.submitted = False
                st.rerun()

    timing = st.session_state.timing
    st.markdown(
        f"<div class='stats'>Search {timing.get('search', 0):.2f}s · Answer {timing.get('answer', 0):.2f}s</div>",
        unsafe_allow_html=True,
    )
