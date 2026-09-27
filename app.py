"""
HCIA-AI Study Assistant (Streamlit app).

The notebook builds everything (rag_db/ and question_bank.json). This app only READS them.

Modes:
    1. Explain Topic
    2. Quiz
    3. Flashcards

Run from a terminal in this folder:
    python -m streamlit run app.py
"""

import os
import re
import html
import difflib
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


# =====================================================================
# Paths + settings
# =====================================================================

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "rag_db"
BANK_JSON = BASE_DIR / "question_bank.json"
ASSETS_DIR = BASE_DIR / "assets"

COLLECTION_NAME = "lectures"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_NAME = "gpt-4o-mini"

ALPHA = 0.5
TOP_K = 3
N_QUIZ_QUESTIONS = 3

# Quiz page settings
QUIZ_TOP_K = 5
QUIZ_MAX_COUNT = 10

# Flashcard settings
FLASHCARD_TOP_K = 6
FLASHCARD_MAX_COUNT = 30
FLASHCARD_DEFAULT_COUNT = 10

EXAMPLE_QUESTIONS = [
    "What is QLoRA?",
    "How does K-means choose the centroids?",
    "Why do we scale features before KNN?",
]


st.set_page_config(
    page_title="HCIA-AI Study Assistant",
    page_icon="📘",
    layout="wide",
)

load_dotenv(BASE_DIR / ".env", override=True)


# =====================================================================
# 1) Load everything ONCE
# =====================================================================

@st.cache_resource(show_spinner="Loading the lectures index...")
def load_resources():
    try:
        # Use local model cache first
        dense_model = SentenceTransformer(
            MODEL_NAME,
            local_files_only=True
        )
    except OSError:
        # Download only if model is not already cached
        dense_model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None
    )

    stored = collection.get(
        include=["documents", "metadatas"]
    )

    # ---------------------------------------------------------------
    # TF-IDF for sparse retrieval
    # ---------------------------------------------------------------

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    sparse_vectors = tfidf.fit_transform(
        stored["documents"]
    )

    # ---------------------------------------------------------------
    # Question bank
    # ---------------------------------------------------------------

    with open(BANK_JSON, encoding="utf-8") as f:
        bank = json.load(f)

    bank_texts = []

    for q in bank:
        bank_texts.append(q["question"])

    bank_vectors = dense_model.encode(
        bank_texts,
        normalize_embeddings=True
    )

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

    explain_llm = ChatOpenAI(
        model=LLM_NAME,
        temperature=0
    )

    quiz_llm = ChatOpenAI(
        model=LLM_NAME,
        temperature=0.7
    ).bind(
        response_format={"type": "json_object"}
    )

    flashcard_llm = ChatOpenAI(
        model=LLM_NAME,
        temperature=0.4
    ).bind(
        response_format={"type": "json_object"}
    )

    return explain_llm, quiz_llm, flashcard_llm


# =====================================================================
# 2) Retrieval + explanation
# =====================================================================

def min_max(scores):

    if scores.max() == scores.min():
        return np.zeros_like(scores)

    return (
        (scores - scores.min())
        / (scores.max() - scores.min())
    )


def hybrid_search(
    query,
    res,
    alpha=ALPHA,
    top_k=TOP_K
):

    query_dense = res["dense_model"].encode(
        [query],
        normalize_embeddings=True
    )

    query_sparse = res["tfidf"].transform([query])

    # ---------------------------------------------------------------
    # Dense scores
    # ---------------------------------------------------------------

    results = res["collection"].query(
        query_embeddings=query_dense.tolist(),
        n_results=res["collection"].count(),
        include=["distances"],
    )

    dense_by_id = {}

    for chunk_id, distance in zip(
        results["ids"][0],
        results["distances"][0]
    ):
        dense_by_id[chunk_id] = 1 - distance

    dense_scores = np.array(
        [
            dense_by_id[chunk_id]
            for chunk_id in res["stored_ids"]
        ]
    )

    # ---------------------------------------------------------------
    # Sparse scores
    # ---------------------------------------------------------------

    sparse_scores = (
        res["sparse_vectors"] @ query_sparse.T
    ).toarray().ravel()

    # ---------------------------------------------------------------
    # Combine
    # ---------------------------------------------------------------

    hybrid_scores = (
        alpha * min_max(dense_scores)
        + (1 - alpha) * min_max(sparse_scores)
    )

    top_positions = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

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


NOT_FOUND_ANSWER = "I don't know based on the lectures."


def is_not_in_lectures(answer):

    text = (
        answer
        .strip()
        .lower()
        .replace("’", "'")
    )

    return text.startswith(
        NOT_FOUND_ANSWER.lower().rstrip(".")
    )


def build_prompt(
    query,
    retrieved_chunks
):

    context_parts = []

    for c in retrieved_chunks:

        context_parts.append(
            f"[{c['lecture']} p.{c['page']}]\n"
            f"{c['text']}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    return f"""You are a helpful tutor helping a student study an AI / Machine Learning course.

The context below contains the lecture slides most relevant to the question.

How to answer:
1. Find every sentence in the context related to the question, even if it uses different words.
2. Explain clearly using only those sentences. Do not add outside knowledge.
3. If the slides cover the question only partly, explain what they cover,
   then add one line saying what they do not cover.
4. Reply "{NOT_FOUND_ANSWER}" ONLY if nothing in the context is related.
5. Cite the slides you used, like [K-means p.12].

Context:
{context}

Question: {query}

Answer:"""


# =====================================================================
# 3) Quiz
# =====================================================================

QUIZ_JSON_FORMAT = """{
  "questions": [
    {
      "type": "single",
      "question": "...",
      "options": {
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      },
      "answer": ["B"],
      "explanation": "...",
      "source": "one of the slide ids above"
    }
  ]
}"""


def get_style_examples(
    topic,
    res,
    n=3
):

    topic_vector = res["dense_model"].encode(
        [topic],
        normalize_embeddings=True
    )[0]

    similarities = (
        res["bank_vectors"] @ topic_vector
    )

    top_positions = np.argsort(
        similarities
    )[::-1][:n]

    examples = []

    for pos in top_positions:
        examples.append(
            res["bank"][pos]
        )

    return examples


def build_quiz_prompt(
    retrieved_chunks,
    style_examples,
    n_questions=N_QUIZ_QUESTIONS,
    avoid_questions=None
):

    slide_parts = []

    for c in retrieved_chunks:

        slide_parts.append(
            f"[slide id: {c['id']}]\n"
            f"{c['text']}"
        )

    slides = "\n\n---\n\n".join(
        slide_parts
    )

    example_parts = []

    for q in style_examples:

        lines = [
            f"({q['type']}) {q['question']}"
        ]

        for letter, option_text in q["options"].items():

            lines.append(
                f"{letter}. {option_text}"
            )

        example_parts.append(
            "\n".join(lines)
        )

    examples = "\n\n".join(
        example_parts
    )

    avoid_text = ""

    if avoid_questions:

        avoid_lines = []

        for old_question in avoid_questions:

            avoid_lines.append(
                f"- {old_question}"
            )

        avoid_text = (
            "The student already answered these questions. "
            "Write NEW questions that test different facts "
            "or ideas from the slides, not the same ones reworded. "
            "This includes the multiple-choice question: build it "
            "from different facts too:\n"
            + "\n".join(avoid_lines)
            + "\n"
        )

    return f"""You write multiple-choice exam questions to help a student revise an AI / Machine Learning course.

Slides (the ONLY source of content you may use):
{slides}

Style examples from the Huawei HCIA-AI exam (copy their STYLE only, NOT their content; they may be about other topics):
{examples}

Rules:
1. Write exactly {n_questions} questions. Every question and every correct answer must come from the slides above. No outside knowledge.
2. Each question has exactly 4 options: A, B, C, D. Wrong options must be plausible but clearly wrong according to the slides.
3. "type" is "single" (exactly 1 correct letter) or "multiple" (2 or 3 correct letters, "select all that apply").
4. At least 1 question must be "multiple".
5. "answer" is always a list of letters, e.g. ["B"] or ["A", "C"].
6. "explanation": 1-2 sentences explaining the correct answer using the slides.
7. "source": the slide id the answer comes from.

{avoid_text}

Return ONLY a JSON object in this format:

{QUIZ_JSON_FORMAT}"""


def clean_source(source):

    if not isinstance(source, str):
        return source

    source = source.strip()
    source = source.strip("[]")
    source = source.replace(
        "slide id:",
        ""
    )

    return source.strip()


def check_question(
    q,
    allowed_ids
):

    if not isinstance(q, dict):
        return ["not a JSON object"]

    problems = []

    q_type = q.get("type")

    if q_type not in [
        "single",
        "multiple"
    ]:
        problems.append(
            f"bad type: {q_type}"
        )

    if not q.get("question"):
        problems.append(
            "empty question"
        )

    options = q.get("options")

    if not isinstance(options, dict):
        options = {}

    if sorted(options.keys()) != [
        "A",
        "B",
        "C",
        "D"
    ]:
        problems.append(
            "options must be exactly A, B, C, D"
        )

    answer = q.get("answer")

    if not isinstance(answer, list) or len(answer) == 0:

        problems.append(
            "answer must be a non-empty list"
        )

        answer = []

    seen_letters = []

    for letter in answer:

        if (
            not isinstance(letter, str)
            or letter not in options
        ):
            problems.append(
                f"answer letter {letter} is not an option"
            )

        elif letter in seen_letters:

            problems.append(
                f"answer letter {letter} is repeated"
            )

        seen_letters.append(letter)

    if (
        q_type == "single"
        and len(answer) != 1
    ):
        problems.append(
            "single question must have exactly 1 answer"
        )

    if (
        q_type == "multiple"
        and len(answer) < 2
    ):
        problems.append(
            "multiple question must have 2+ answers"
        )

    if q.get("source") not in allowed_ids:

        problems.append(
            f"unknown source: {q.get('source')}"
        )

    return problems


def is_repeat(
    question,
    old_questions
):

    for old in old_questions:

        ratio = difflib.SequenceMatcher(
            None,
            question.lower(),
            old.lower()
        ).ratio()

        if ratio > 0.8:
            return True

    return False


def generate_quiz(
    query,
    retrieved_chunks,
    res,
    quiz_llm,
    n_questions=N_QUIZ_QUESTIONS,
    avoid_questions=None
):

    style_examples = get_style_examples(
        query,
        res
    )

    prompt = build_quiz_prompt(
        retrieved_chunks,
        style_examples,
        n_questions=n_questions,
        avoid_questions=avoid_questions
    )

    response = quiz_llm.invoke(prompt)

    try:

        data = json.loads(
            response.content
        )

    except json.JSONDecodeError:

        return [], [
            "The LLM did not return valid JSON"
        ]

    if (
        not isinstance(data, dict)
        or not isinstance(
            data.get("questions"),
            list
        )
    ):

        return [], [
            "The JSON does not contain a 'questions' list"
        ]

    allowed_ids = [
        c["id"]
        for c in retrieved_chunks
    ]

    valid_questions = []
    reasons = []

    for i, q in enumerate(
        data["questions"],
        start=1
    ):

        if isinstance(q, dict):

            q["source"] = clean_source(
                q.get("source")
            )

        problems = check_question(
            q,
            allowed_ids
        )

        if (
            not problems
            and avoid_questions
            and is_repeat(
                q["question"],
                avoid_questions
            )
        ):
            problems = [
                "repeats a question from the last quiz"
            ]

        if problems:

            reasons.append(
                f"Dropped question {i}: {problems}"
            )

        else:

            valid_questions.append(q)

    return valid_questions, reasons


def grade_answer(
    chosen,
    answer
):

    return set(chosen) == set(answer)


# =====================================================================
# 4) FLASHCARDS
# =====================================================================

FLASHCARD_JSON_FORMAT = """{
  "flashcards": [
    {
      "point": "...",
      "source": "one of the slide ids above",
      "importance": "high"
    }
  ]
}"""


def build_flashcard_prompt(
    topic,
    retrieved_chunks,
    n_cards,
    difficulty,
    focus,
    include_formulas,
    include_examples
):

    slide_parts = []

    for c in retrieved_chunks:

        slide_parts.append(
            f"[slide id: {c['id']} | "
            f"{c['lecture']} p.{c['page']}]\n"
            f"{c['text']}"
        )

    slides = "\n\n---\n\n".join(
        slide_parts
    )

    formula_rule = (
        "Include important formulas when they appear in the slides."
        if include_formulas
        else "Do NOT include formula-based points."
    )

    example_rule = (
        "Include important examples when they appear in the slides."
        if include_examples
        else "Do NOT include example-based points."
    )

    return f"""You are creating study flashcards for an AI / Machine Learning student.

The student wants flashcards about this topic:

TOPIC:
{topic}

The slides below are the ONLY source of information.

SLIDES:
{slides}

FLASHCARD SETTINGS:

Number of cards:
{n_cards}

Difficulty:
{difficulty}

Information focus:
{focus}

Formula setting:
{formula_rule}

Example setting:
{example_rule}

IMPORTANT RULES:

1. Create exactly {n_cards} flashcards.
2. Each flashcard is NOT a question-and-answer pair. It is a single,
   self-contained piece of important information written as a clear,
   direct statement (a fact, a definition, a key principle, a step, a
   comparison, or a relationship) that the student should remember.
3. Use ONLY information explicitly contained in the provided slides.
4. Do NOT add outside knowledge.
5. Do NOT invent facts, formulas, examples, definitions, or explanations.
6. Prioritize important information such as:
   - definitions
   - key concepts
   - important principles
   - steps or processes
   - comparisons
   - relationships
   - formulas when allowed
   - important examples when allowed
7. Each flashcard should cover ONE main idea, written concisely
   (roughly 1-3 sentences).
8. Keep the wording clear and suitable for the selected difficulty.
9. Each flashcard MUST have a valid source slide id from the provided slides.
10. "importance" must be either "high" or "medium".
11. Avoid generating duplicate or almost-identical points.
12. Return ONLY JSON.

JSON FORMAT:

{FLASHCARD_JSON_FORMAT}
"""


def check_flashcard(
    card,
    allowed_ids
):

    if not isinstance(card, dict):
        return [
            "not a JSON object"
        ]

    problems = []

    if not card.get("point"):
        problems.append(
            "empty point"
        )

    if card.get("source") not in allowed_ids:
        problems.append(
            f"unknown source: {card.get('source')}"
        )

    if card.get("importance") not in [
        "high",
        "medium"
    ]:
        problems.append(
            "importance must be high or medium"
        )

    return problems


def generate_flashcards(
    topic,
    retrieved_chunks,
    flashcard_llm,
    n_cards,
    difficulty,
    focus,
    include_formulas,
    include_examples
):

    prompt = build_flashcard_prompt(
        topic=topic,
        retrieved_chunks=retrieved_chunks,
        n_cards=n_cards,
        difficulty=difficulty,
        focus=focus,
        include_formulas=include_formulas,
        include_examples=include_examples,
    )

    response = flashcard_llm.invoke(
        prompt
    )

    try:

        data = json.loads(
            response.content
        )

    except json.JSONDecodeError:

        return [], [
            "The LLM did not return valid JSON."
        ]

    if (
        not isinstance(data, dict)
        or not isinstance(
            data.get("flashcards"),
            list
        )
    ):

        return [], [
            "The JSON does not contain a 'flashcards' list."
        ]

    allowed_ids = [
        c["id"]
        for c in retrieved_chunks
    ]

    valid_cards = []
    reasons = []

    for i, card in enumerate(
        data["flashcards"],
        start=1
    ):

        if isinstance(card, dict):

            card["source"] = clean_source(
                card.get("source")
            )

            card["point"] = (
                str(card.get("point", ""))
                .strip()
            )

        problems = check_flashcard(
            card,
            allowed_ids
        )

        # Check duplicate points
        if not problems:

            for old_card in valid_cards:

                similarity = difflib.SequenceMatcher(
                    None,
                    card["point"].lower(),
                    old_card["point"].lower()
                ).ratio()

                if similarity > 0.8:

                    problems.append(
                        "duplicate or very similar point"
                    )

                    break

        if problems:

            reasons.append(
                f"Dropped flashcard {i}: {problems}"
            )

        else:

            valid_cards.append(card)

    return valid_cards, reasons


def find_flashcard_source(
    card,
    retrieved_chunks
):

    for c in retrieved_chunks:

        if c["id"] == card["source"]:
            return c

    return None


def make_flashcards(
    res,
    flashcard_llm
):

    topic = st.session_state.flashcard_topic

    with st.spinner(
        "Searching the lectures and creating your flashcards..."
    ):

        # Retrieve more content than normal explanation
        retrieved_chunks = hybrid_search(
            topic,
            res,
            alpha=ALPHA,
            top_k=FLASHCARD_TOP_K
        )

        try:

            cards, reasons = generate_flashcards(
                topic=topic,
                retrieved_chunks=retrieved_chunks,
                flashcard_llm=flashcard_llm,
                n_cards=st.session_state.flashcard_count,
                difficulty=st.session_state.flashcard_difficulty,
                focus=st.session_state.flashcard_focus,
                include_formulas=st.session_state.flashcard_formulas,
                include_examples=st.session_state.flashcard_examples,
            )

        except Exception as e:

            cards = []
            reasons = [
                f"Flashcard generation failed: {e}"
            ]

    if not cards:

        return reasons

    st.session_state.flashcards = cards
    st.session_state.flashcard_chunks = retrieved_chunks
    st.session_state.flashcard_index = 0

    return reasons


def flashcard_source_text(
    card,
    chunks
):

    source = find_flashcard_source(
        card,
        chunks
    )

    if source is None:
        return None

    return source


def show_flashcard():

    cards = st.session_state.flashcards

    if not cards:
        return

    index = st.session_state.flashcard_index

    # Safety
    if index < 0:
        index = 0

    if index >= len(cards):
        index = len(cards) - 1

    st.session_state.flashcard_index = index

    card = cards[index]

    # ---------------------------------------------------------------
    # Progress
    # ---------------------------------------------------------------

    st.markdown(
        f"<div style='text-align:center; font-size:15px; margin-bottom:10px;'>"
        f"Flashcard <b>{index + 1}</b> of <b>{len(cards)}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

    progress = (index + 1) / len(cards)

    st.progress(progress)

    # ---------------------------------------------------------------
    # Key info card
    # ---------------------------------------------------------------

    importance = card.get(
        "importance",
        "medium"
    )

    badge_color = (
        "#e8734a"
        if importance == "high"
        else "#8a8a8a"
    )

    st.markdown(
        f"<div style='display:flex; justify-content:space-between; "
        f"align-items:center; margin-top:20px; margin-bottom:8px;'>"
        f"<span style='font-size:13px; opacity:0.7;'>KEY INFO</span>"
        f"<span style='font-size:11px; font-weight:600; text-transform:uppercase; "
        f"letter-spacing:0.03em; padding:3px 10px; border-radius:999px; "
        f"color:white; background:{badge_color};'>{importance}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.markdown(
            f"<div style='font-size:22px; font-weight:600; line-height:1.6;'>"
            f"{html.escape(card['point'])}"
            f"</div>",
            unsafe_allow_html=True
        )

    # ---------------------------------------------------------------
    # Source
    # ---------------------------------------------------------------

    source = flashcard_source_text(
        card,
        st.session_state.flashcard_chunks
    )

    if source:

        st.caption(
            f"📖 Source: {source['lecture']} · "
            f"p.{source['page']}"
        )

    # ---------------------------------------------------------------
    # Navigation
    # ---------------------------------------------------------------

    st.markdown("")

    col1, col2, col3 = st.columns(
        [1, 1, 1]
    )

    with col1:

        if st.button(
            "← Previous",
            disabled=(index == 0),
            use_container_width=True
        ):

            st.session_state.flashcard_index -= 1
            st.rerun()

    with col2:

        if st.button(
            "🔀 Shuffle",
            use_container_width=True
        ):

            import random

            random.shuffle(
                st.session_state.flashcards
            )

            st.session_state.flashcard_index = 0
            st.rerun()

    with col3:

        if st.button(
            "Next →",
            disabled=(index == len(cards) - 1),
            use_container_width=True
        ):

            st.session_state.flashcard_index += 1
            st.rerun()


def render_flashcards_page(
    res,
    flashcard_llm,
    n_lectures,
    n_slides
):

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------

    st.markdown(
        """
<div class="hero">

  <div class="badge">
    <span class="dot"></span>
    Flashcards from your NTI lecture slides
  </div>

  <h1>
    Study With <span class="hl">Flashcards</span>
  </h1>

  <p>
    Choose exactly how you want your flashcards to be generated.
    Every point comes straight from your lecture slides.
  </p>

</div>
""",
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------

    st.markdown("### ⚙️ Flashcard Settings")

    col1, col2, col3 = st.columns(3)

    with col1:

        topic = st.text_input(
            "Topic",
            value=st.session_state.flashcard_topic,
            placeholder="e.g. K-means clustering"
        )

        count = st.number_input(
            "Number of flashcards",
            min_value=1,
            max_value=FLASHCARD_MAX_COUNT,
            value=FLASHCARD_DEFAULT_COUNT,
            step=1
        )

    with col2:

        difficulty = st.selectbox(
            "Difficulty",
            [
                "Easy",
                "Medium",
                "Hard",
                "Mixed"
            ],
            index=3
        )

        focus = st.selectbox(
            "Information focus",
            [
                "Core concepts only",
                "Core + supporting concepts"
            ]
        )

    with col3:

        st.caption(
            f"📚 {n_lectures} lectures · "
            f"{n_slides} slides available"
        )

    st.markdown("#### Additional content")

    col1, col2 = st.columns(2)

    with col1:

        include_formulas = st.checkbox(
            "➗ Include formulas",
            value=True
        )

    with col2:

        include_examples = st.checkbox(
            "💡 Include examples",
            value=True
        )

    # ---------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------

    if st.button(
        "✨ Generate Flashcards",
        type="primary",
        use_container_width=True
    ):

        if not topic.strip():

            st.warning(
                "Please enter a topic first."
            )

        else:

            st.session_state.flashcard_topic = topic.strip()
            st.session_state.flashcard_count = int(count)
            st.session_state.flashcard_difficulty = difficulty
            st.session_state.flashcard_focus = focus
            st.session_state.flashcard_formulas = include_formulas
            st.session_state.flashcard_examples = include_examples

            reasons = make_flashcards(
                res,
                flashcard_llm
            )

            if reasons and st.session_state.flashcards:

                with st.expander(
                    "Generation details"
                ):

                    for reason in reasons:
                        st.caption(reason)

            if not st.session_state.flashcards:

                st.warning(
                    "I couldn't create flashcards from the retrieved lecture content. "
                    "Try a more specific topic."
                )

            else:

                st.rerun()

    # ---------------------------------------------------------------
    # Existing flashcards
    # ---------------------------------------------------------------

    if st.session_state.flashcards:

        st.divider()

        col1, col2 = st.columns(
            [3, 1]
        )

        with col1:

            st.markdown(
                f"### 📚 {st.session_state.flashcard_topic}"
            )

            st.caption(
                f"{len(st.session_state.flashcards)} flashcards · "
                f"{st.session_state.flashcard_difficulty}"
            )

        with col2:

            if st.button(
                "🗑️ Clear Cards",
                use_container_width=True
            ):

                st.session_state.flashcards = []
                st.session_state.flashcard_chunks = []
                st.session_state.flashcard_index = 0
                st.rerun()

        show_flashcard()


# =====================================================================
# 5) UI helpers
# =====================================================================

def buddy_html():

    image_bytes = (
        ASSETS_DIR / "study_buddy.webp"
    ).read_bytes()

    encoded = base64.b64encode(
        image_bytes
    ).decode()

    return f"""
<div class="buddy">
<img src="data:image/webp;base64,{encoded}" alt="Study buddy">
<div class="thinking">
<span></span>
<span></span>
<span></span>
</div>
</div>
"""


def highlight_citations(answer):

    return re.sub(
        r"\[([^\[\]]+? p\.\d+)\]",
        r"`\1`",
        answer
    )


def top_bar(
    n_lectures,
    n_slides
):

    st.markdown(
        f"<div class='topbar'>"
        f"<div class='brand'>"
        f"<span class='logo'>AI</span>"
        f"<span>Study Assistant</span>"
        f"</div>"
        f"<div class='chip'>{n_lectures} lectures · {n_slides} slides</div>"
        f"</div>",
        unsafe_allow_html=True
    )


def task_card(
    title,
    lines,
    floating=False
):

    body = "<br>".join(lines)

    css_class = (
        "task-card floating"
        if floating
        else "task-card"
    )

    st.markdown(
        f"""
<div class="{css_class}">
  <div class="task-title">
    <span>Task</span> {title}
  </div>

  <div class="task-body">
    {body}
  </div>
</div>
""",
        unsafe_allow_html=True
    )


def show_sources(chunks):

    with st.expander(
        f"📄 Slides used ({len(chunks)})"
    ):

        for c in chunks:

            st.markdown(
                f"""
**{c['lecture']} · p.{c['page']}**
<span class='score'>
score {c['score']:.2f}
</span>
""",
                unsafe_allow_html=True
            )

            slide_text = (
                c["text"]
                .split("\n", 1)[-1]
            )

            st.caption(
                slide_text[:500]
            )


def question_title(
    number,
    q,
    mark=""
):

    if q["type"] == "single":

        badge = (
            "<span class='qtype single'>"
            "Single choice · pick 1"
            "</span>"
        )

    else:

        badge = (
            "<span class='qtype multiple'>"
            "Multi choice · select all that apply"
            "</span>"
        )

    st.markdown(
        f"<div class='qtitle'>{badge}"
        f"<div style='margin-top:10px;'>{mark} "
        f"<b>Q{number}. {html.escape(q['question'])}</b>"
        f"</div></div>",
        unsafe_allow_html=True
    )


def option_row(
    letter,
    text,
    css_class,
    tag
):

    tag_html = (
        f"<span class='opt-tag'>{tag}</span>"
        if tag
        else ""
    )

    return (
        f"<div class='opt {css_class}'>"
        f"<b>{letter}.</b> "
        f"{html.escape(text)}"
        f"{tag_html}"
        f"</div>"
    )


def show_answer_review(
    q,
    chosen
):

    rows = []

    for letter, text in q["options"].items():

        is_right = (
            letter in q["answer"]
        )

        is_chosen = (
            letter in chosen
        )

        if is_right and is_chosen:

            rows.append(
                option_row(
                    letter,
                    text,
                    "right",
                    "✓ Your answer"
                )
            )

        elif is_chosen:

            rows.append(
                option_row(
                    letter,
                    text,
                    "wrong",
                    "✗ Your answer"
                )
            )

        elif is_right:

            rows.append(
                option_row(
                    letter,
                    text,
                    "right missed",
                    "✓ Correct answer"
                )
            )

        else:

            rows.append(
                option_row(
                    letter,
                    text,
                    "",
                    ""
                )
            )

    st.markdown(
        "".join(rows),
        unsafe_allow_html=True
    )


def make_new_quiz_for_topic(
    res,
    quiz_llm
):

    quiz = []
    all_reasons = []

    n_questions = st.session_state.quiz_count

    with st.spinner(
        "Writing questions from these slides..."
    ):

        for attempt in [1, 2]:

            avoid = list(
                st.session_state.quiz_previous_questions
            )

            for q in quiz:

                avoid.append(
                    q["question"]
                )

            try:

                new_questions, reasons = generate_quiz(
                    st.session_state.quiz_topic,
                    st.session_state.quiz_chunks,
                    res,
                    quiz_llm,
                    n_questions=n_questions,
                    avoid_questions=avoid
                )

            except Exception as e:

                new_questions = []

                reasons = [
                    f"Quiz generation failed: {e}"
                ]

            for reason in reasons:

                all_reasons.append(
                    f"Try {attempt}: {reason}"
                )

                print(
                    f"Try {attempt}: {reason}"
                )

            for q in new_questions:

                if len(quiz) < n_questions:

                    quiz.append(q)

            if len(quiz) >= n_questions:
                break

    if not quiz:

        return all_reasons

    for q in quiz:

        st.session_state.quiz_previous_questions.append(
            q["question"]
        )

    st.session_state.quiz = quiz
    st.session_state.quiz_round += 1
    st.session_state.quiz_submitted = False
    st.session_state.quiz_chosen = {}

    return []


def make_quiz(
    res,
    quiz_llm
):

    topic = st.session_state.quiz_topic

    with st.spinner(
        "Searching the lectures..."
    ):

        chunks = hybrid_search(
            topic,
            res,
            alpha=ALPHA,
            top_k=QUIZ_TOP_K
        )

    st.session_state.quiz_chunks = chunks
    st.session_state.quiz_previous_questions = []

    return make_new_quiz_for_topic(
        res,
        quiz_llm
    )


def show_quiz_error(
    reasons
):

    st.warning(
        "Could not generate a quiz this time. Please click again."
    )

    with st.expander(
        "Why? (details)"
    ):

        for reason in reasons:
            st.caption(reason)


def find_chunk_in(
    chunks,
    chunk_id
):

    for c in chunks:

        if c["id"] == chunk_id:
            return c

    return None


def render_quiz_page(
    res,
    quiz_llm,
    n_lectures,
    n_slides
):

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------

    st.markdown(
        """
        <div class="hero">
          <div class="badge">
            <span class="dot"></span>
            Quizzes from your NTI lecture slides
          </div>

          <h1>
            Test Yourself With <span class="hl">Quizzes</span>
          </h1>

          <p>
            Pick a topic and get multiple-choice questions,
            written only from your lecture slides.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------

    st.markdown("### ⚙️ Quiz Settings")

    col1, col2 = st.columns(2)

    with col1:

        topic = st.text_input(
            "Topic",
            value=st.session_state.quiz_topic,
            placeholder="e.g. K-means clustering"
        )

    with col2:

        count = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=QUIZ_MAX_COUNT,
            value=st.session_state.quiz_count,
            step=1
        )

    if st.button(
        "📝 Generate Quiz",
        type="primary",
        use_container_width=True
    ):

        if not topic.strip():

            st.warning(
                "Please enter a topic first."
            )

        else:

            st.session_state.quiz_topic = topic.strip()
            st.session_state.quiz_count = int(count)

            reasons = make_quiz(
                res,
                quiz_llm
            )

            if reasons and st.session_state.quiz:

                with st.expander(
                    "Generation details"
                ):

                    for reason in reasons:
                        st.caption(reason)

            if not st.session_state.quiz:

                st.warning(
                    "I couldn't create a quiz from the retrieved lecture content. "
                    "Try a more specific topic."
                )

            else:

                st.rerun()

    if not st.session_state.quiz:

        st.caption(
            f"📚 {n_lectures} lectures · "
            f"{n_slides} slides available"
        )

        return

    st.divider()

    # ---------------------------------------------------------------
    # Quiz header + clear
    # ---------------------------------------------------------------

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        st.markdown(
            f"### 📝 {st.session_state.quiz_topic}"
        )

        lecture_names = []

        for c in st.session_state.quiz_chunks:

            if c["lecture"] not in lecture_names:

                lecture_names.append(
                    c["lecture"]
                )

        st.caption(
            f"{len(st.session_state.quiz)} questions · "
            "From: " + ", ".join(lecture_names)
        )

    with col2:

        if st.button(
            "🗑️ Clear Quiz",
            use_container_width=True
        ):

            st.session_state.quiz = []
            st.session_state.quiz_chunks = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_chosen = {}
            st.rerun()

    # ---------------------------------------------------------------
    # QUIZ
    # ---------------------------------------------------------------

    if not st.session_state.quiz_submitted:

        with st.form(
            f"quiz_{st.session_state.quiz_round}"
        ):

            st.markdown(
                "#### 📝 Quiz"
            )

            for i, q in enumerate(
                st.session_state.quiz
            ):

                key = (
                    f"r{st.session_state.quiz_round}"
                    f"_q{i}"
                )

                question_title(
                    i + 1,
                    q
                )

                if q["type"] == "single":

                    st.radio(
                        "Choose one answer",
                        options=list(
                            q["options"].keys()
                        ),
                        format_func=lambda letter, q=q:
                            f"{letter}. {q['options'][letter]}",
                        index=None,
                        key=key,
                        label_visibility="collapsed",
                    )

                else:

                    for letter, option_text in q["options"].items():

                        st.checkbox(
                            f"{letter}. {option_text}",
                            key=f"{key}_{letter}"
                        )

            submitted = st.form_submit_button(
                "Submit answers",
                type="primary"
            )

        if submitted:

            chosen = {}

            for i, q in enumerate(
                st.session_state.quiz
            ):

                key = (
                    f"r{st.session_state.quiz_round}"
                    f"_q{i}"
                )

                if q["type"] == "single":

                    picked = st.session_state.get(
                        key
                    )

                    if picked is None:

                        chosen[i] = []

                    else:

                        chosen[i] = [
                            picked
                        ]

                else:

                    chosen[i] = []

                    for letter in q["options"]:

                        if st.session_state.get(
                            f"{key}_{letter}"
                        ):

                            chosen[i].append(
                                letter
                            )

            st.session_state.quiz_chosen = chosen

            st.session_state.quiz_submitted = True

            st.rerun()

    # ---------------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------------

    if st.session_state.quiz_submitted:

        quiz = st.session_state.quiz

        score = 0

        for i, q in enumerate(quiz):

            if grade_answer(
                st.session_state.quiz_chosen[i],
                q["answer"]
            ):

                score += 1

        st.markdown(
            f"<div class='score-card'>Score <b>{score} / {len(quiz)}</b></div>",
            unsafe_allow_html=True
        )

        for i, q in enumerate(quiz):

            chosen = (
                st.session_state.quiz_chosen[i]
            )

            is_correct = grade_answer(
                chosen,
                q["answer"]
            )

            with st.container(
                border=True
            ):

                mark = (
                    "✅"
                    if is_correct
                    else "❌"
                )

                question_title(
                    i + 1,
                    q,
                    mark
                )

                if not chosen:

                    st.caption(
                        "You did not answer this question."
                    )

                show_answer_review(
                    q,
                    chosen
                )

                if st.session_state.quiz_show_explanations:

                    st.caption(
                        q["explanation"]
                    )

                    if not is_correct:

                        slide = find_chunk_in(
                            st.session_state.quiz_chunks,
                            q["source"]
                        )

                        if slide is not None:

                            with st.expander(
                                f"📖 Review this slide: "
                                f"{slide['lecture']} "
                                f"p.{slide['page']}"
                            ):

                                st.caption(
                                    slide["text"]
                                    .split("\n", 1)[-1]
                                )

        if st.button(
            "🔄 New quiz on this topic"
        ):

            reasons = make_new_quiz_for_topic(
                res,
                quiz_llm
            )

            if reasons:

                show_quiz_error(
                    reasons
                )

            else:

                st.rerun()


# =====================================================================
# 6) Page setup
# =====================================================================

css = (
    ASSETS_DIR / "style.css"
).read_text(
    encoding="utf-8"
)

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# =====================================================================
# 7) Session state
# =====================================================================

if "question" not in st.session_state:

    st.session_state.question = None
    st.session_state.answer = None
    st.session_state.chunks = []
    st.session_state.timing = {}


# ---------------- Quiz page state ----------------

if "quiz_topic" not in st.session_state:

    st.session_state.quiz_topic = ""


if "quiz_count" not in st.session_state:

    st.session_state.quiz_count = N_QUIZ_QUESTIONS


if "quiz_chunks" not in st.session_state:

    st.session_state.quiz_chunks = []


if "quiz" not in st.session_state:

    st.session_state.quiz = []


if "quiz_round" not in st.session_state:

    st.session_state.quiz_round = 0


if "quiz_submitted" not in st.session_state:

    st.session_state.quiz_submitted = False


if "quiz_chosen" not in st.session_state:

    st.session_state.quiz_chosen = {}


if "quiz_previous_questions" not in st.session_state:

    st.session_state.quiz_previous_questions = []


if "quiz_show_explanations" not in st.session_state:

    st.session_state.quiz_show_explanations = True


# ---------------- Flashcard state ----------------

if "flashcards" not in st.session_state:

    st.session_state.flashcards = []


if "flashcard_chunks" not in st.session_state:

    st.session_state.flashcard_chunks = []


if "flashcard_index" not in st.session_state:

    st.session_state.flashcard_index = 0


if "flashcard_topic" not in st.session_state:

    st.session_state.flashcard_topic = ""


if "flashcard_count" not in st.session_state:

    st.session_state.flashcard_count = FLASHCARD_DEFAULT_COUNT


if "flashcard_difficulty" not in st.session_state:

    st.session_state.flashcard_difficulty = "Mixed"


if "flashcard_focus" not in st.session_state:

    st.session_state.flashcard_focus = (
        "Core + supporting concepts"
    )


if "flashcard_formulas" not in st.session_state:

    st.session_state.flashcard_formulas = True


if "flashcard_examples" not in st.session_state:

    st.session_state.flashcard_examples = True


# =====================================================================
# 8) API key
# =====================================================================

if not os.getenv("OPENAI_API_KEY"):

    st.error(
        "OPENAI_API_KEY was not found. "
        "Put it in a `.env` file next to app.py, "
        "then restart the app."
    )

    st.stop()


# =====================================================================
# 9) Load resources
# =====================================================================

try:

    res = load_resources()

except Exception as e:

    st.error(
        "Could not load `rag_db/` or "
        "`question_bank.json`. "
        "Run the notebook first.\n\n"
        f"{e}"
    )

    st.stop()


explain_llm, quiz_llm, flashcard_llm = load_llms()


# =====================================================================
# 10) Basic statistics
# =====================================================================

n_slides = len(
    res["stored_ids"]
)

lectures = set()

for meta in res["stored_metas"]:

    lectures.add(
        meta["lecture"]
    )


# =====================================================================
# 11) Study Mode selector
# =====================================================================

st.sidebar.markdown(
    "## 📚 Study Mode"
)

page = st.sidebar.radio(
    "Choose a mode",
    [
        "💡 Explain Topic",
        "📝 Quiz",
        "📚 Flashcards"
    ],
    label_visibility="collapsed"
)

if page == "📝 Quiz":

    st.sidebar.markdown("---")

    st.session_state.quiz_show_explanations = st.sidebar.checkbox(
        "Show explanations after grading",
        value=st.session_state.quiz_show_explanations,
        help="Turn off to just see your score and which answers were right or wrong, without the explanation text."
    )


# =====================================================================
# 12) QUIZ PAGE
# =====================================================================

if page == "📝 Quiz":

    render_quiz_page(
        res=res,
        quiz_llm=quiz_llm,
        n_lectures=len(lectures),
        n_slides=n_slides
    )

    # IMPORTANT:
    # Stop here so Explain Topic does not also execute.
    st.stop()


# =====================================================================
# 12b) FLASHCARD PAGE
# =====================================================================

if page == "📚 Flashcards":

    render_flashcards_page(
        res=res,
        flashcard_llm=flashcard_llm,
        n_lectures=len(lectures),
        n_slides=n_slides
    )

    # IMPORTANT:
    # Stop here so Explain Topic does not also execute.
    st.stop()


# =====================================================================
# 13) EXPLAIN TOPIC PAGE
# =====================================================================


top_bar(
    len(lectures),
    n_slides
)


# =====================================================================
# 14) Input
# =====================================================================

typed_question = st.chat_input(
    "Ask about a lecture topic (in English)..."
)

new_question = typed_question


# =====================================================================
# 15) Home screen
# =====================================================================

if st.session_state.question is None:

    st.markdown(
        """
<div class="hero">

  <div class="badge">
    <span class="dot"></span>
    Answers only from your NTI lecture slides
  </div>

  <h1>
    Turn Your <span class="hl">Lectures</span> Into<br>
    Clear <span class="hl">Answers</span>
  </h1>

  <p>
    Explained from the slides, cited by page.
    Head to the Quiz tab to test yourself after.
  </p>

</div>
""",
        unsafe_allow_html=True
    )

    background_ids = "  ·  ".join(
        res["stored_ids"][:60]
    )

    st.markdown(
        f"""
<div class="stage">

  <div class="code-rain">
    {background_ids}
  </div>

  {buddy_html()}

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='try-label'>Try one:</div>",
        unsafe_allow_html=True
    )

    _, col1, col2, col3, _ = st.columns(
        [
            0.6,
            1,
            1.7,
            1.6,
            0.4
        ],
        gap="small"
    )

    for col, example in zip(
        [col1, col2, col3],
        EXAMPLE_QUESTIONS
    ):

        if col.button(example):

            new_question = example

    task_card(
        "Ready",
        [
            f"{len(lectures)} lectures indexed",
            "Ask in English for the best results"
        ],
        floating=True
    )


# =====================================================================
# 16) New question
# =====================================================================

if new_question:

    with st.spinner(
        "Searching the slides and writing the explanation..."
    ):

        t0 = time.time()

        chunks = hybrid_search(
            new_question,
            res
        )

        t1 = time.time()

        try:

            answer = explain_llm.invoke(
                build_prompt(
                    new_question,
                    chunks
                )
            ).content

        except Exception as e:

            answer = None

            st.error(
                f"The LLM call failed: {e}"
            )

        t2 = time.time()

    if answer is not None:

        st.session_state.question = (
            new_question
        )

        st.session_state.answer = (
            answer
        )

        st.session_state.chunks = (
            chunks
        )

        st.session_state.timing = {
            "search": t1 - t0,
            "answer": t2 - t1
        }

        st.rerun()


# =====================================================================
# 17) Answer screen
# =====================================================================

if st.session_state.question is not None:

    left, right = st.columns(
        [1, 2.3],
        gap="large"
    )

    # ---------------------------------------------------------------
    # Left
    # ---------------------------------------------------------------

    with left:

        st.markdown(
            f"<div class='stage small'>{buddy_html()}</div>",
            unsafe_allow_html=True
        )

        lecture_names = []

        for c in st.session_state.chunks:

            if c["lecture"] not in lecture_names:

                lecture_names.append(
                    c["lecture"]
                )

        not_found = is_not_in_lectures(
            st.session_state.answer
        )

        if not_found:

            task_card(
                "Not in the lectures",
                [
                    "Try a topic from the course",
                    f"{len(lectures)} lectures indexed"
                ]
            )

        else:

            task_card(
                "Explain",
                [
                    f"{len(st.session_state.chunks)} slides found",
                    "From: " + ", ".join(lecture_names)
                ]
            )

    # ---------------------------------------------------------------
    # Right
    # ---------------------------------------------------------------

    with right:

        st.markdown(
            f"""
<div class='question-bubble'>
    {html.escape(st.session_state.question)}
</div>
""",
            unsafe_allow_html=True
        )

        with st.container(
            border=True
        ):

            st.markdown(
                highlight_citations(
                    st.session_state.answer
                )
            )

            if not not_found:

                show_sources(
                    st.session_state.chunks
                )

        # -----------------------------------------------------------
        # No topic found
        # -----------------------------------------------------------

        if not_found:

            st.info(
                "This topic is not in the lectures. "
                "Ask about a topic from the course."
            )

    # ---------------------------------------------------------------
    # Timing
    # ---------------------------------------------------------------

    timing = (
        st.session_state.timing
    )

    st.markdown(
        f"""
<div class='stats'>
    Search {timing.get('search', 0):.2f}s
    ·
    Answer {timing.get('answer', 0):.2f}s
</div>
""",
        unsafe_allow_html=True
    )