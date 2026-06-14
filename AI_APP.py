import streamlit as st
import google.generativeai as genai
from pathlib import Path
import re
import math
import pandas as pd
import requests

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
gemini_API_KEY = st.secrets["gemini_API"]
tavily_API_KEY = st.secrets["tavily_API"]

genai.configure(api_key=gemini_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_INSTRUCTION = (
    "You are a supportive, enthusiastic, straight-to-the-point AI Assistant. "
    "Have some personality, sound human, and use emojis very lightly."
)

DOCS_DIR = Path("uploaded_docs")
DOCS_DIR.mkdir(exist_ok=True)

CHUNK_SIZE    = 550
CHUNK_OVERLAP = 100
TOP_K         = 4

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
st.set_page_config(page_title="AI Assistant")
st.title("Aarin's :blue[AI Assistant]")

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "force_web_search" not in st.session_state:
    st.session_state.force_web_search = False

# ──────────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload your text or CSV files",
    type=["txt", "csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    for f in uploaded_files:
        file_path = DOCS_DIR / f.name
        with open(file_path, "wb") as out:
            out.write(f.read())

# ──────────────────────────────────────────────
# RAG HELPERS
# ──────────────────────────────────────────────
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def load_and_chunk_docs(docs_dir: Path):
    docs = []
    for p in docs_dir.glob("*"):
        try:
            if p.suffix == ".csv":
                text = pd.read_csv(p).to_string()
            else:
                text = p.read_text(encoding="utf-8")
            for i, chunk in enumerate(split_text(text)):
                if chunk.strip():
                    docs.append({"id": f"{p.name}__chunk{i}", "source": p.name, "text": chunk})
        except Exception:
            continue
    return docs


def build_tfidf_index(docs):
    N = len(docs)
    index, df, doc_lengths = {}, {}, {}
    for d in docs:
        tokens = tokenize(d["text"])
        doc_lengths[d["id"]] = len(tokens) or 1
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            tf[t] = 1 + math.log(tf[t])
            df[t] = df.get(t, 0) + 1
        index[d["id"]] = tf
    idf = {t: math.log((N + 1) / (freq + 1)) + 1 for t, freq in df.items()}
    return {
        "index": index,
        "idf": idf,
        "doc_lengths": doc_lengths,
        "docs_meta": {d["id"]: d for d in docs},
    }


def score_query(query, index_struct, top_k=TOP_K):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    q_tf = {t: 1 + math.log(q_tokens.count(t)) for t in set(q_tokens)}
    idf, index, docs_meta = index_struct["idf"], index_struct["index"], index_struct["docs_meta"]
    q_vec = {t: tf * idf[t] for t, tf in q_tf.items() if t in idf}
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
    scores = []
    for doc_id, doc_tf in index.items():
        dot = sum(qv * (doc_tf.get(t, 0) * idf.get(t, 0)) for t, qv in q_vec.items())
        doc_norm = math.sqrt(sum((doc_tf[t] * idf.get(t, 0)) ** 2 for t in doc_tf)) or 1.0
        score = dot / (q_norm * doc_norm)
        if score > 0:
            meta = docs_meta[doc_id]
            scores.append((score, meta["source"], meta["text"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


def doc_relevance_score(query: str) -> float:
    docs = load_and_chunk_docs(DOCS_DIR)
    if not docs:
        return 0.0
    index_struct = build_tfidf_index(docs)
    results = score_query(query, index_struct, top_k=1)
    return results[0][0] if results else 0.0


# ──────────────────────────────────────────────
# USER DISSATISFACTION DETECTOR
# ──────────────────────────────────────────────
DISSATISFACTION_TRIGGERS = re.compile(
    r"\b("
    r"wrong|incorrect|not right|that.s (wrong|incorrect|not right|off)"
    r"|not helpful|doesn.t help|didn.t help"
    r"|try (again|the web|searching|online)"
    r"|search (the web|online|internet|for it)"
    r"|look it up|google it|find it online"
    r"|outdated|old (info|information|answer|data)"
    r"|you.re (wrong|incorrect|mistaken|off)"
    r"|that.s (not|wrong|incorrect|off|outdated)"
    r"|i don.t think (that.s|you.re) right"
    r"|are you sure|double.?check"
    r"|not (accurate|correct|right)"
    r")\b",
    re.IGNORECASE,
)

def user_wants_web_search(query: str) -> bool:
    return bool(DISSATISFACTION_TRIGGERS.search(query))


# ──────────────────────────────────────────────
# LLM CONFIDENCE CHECKER (two-layer)
# ──────────────────────────────────────────────
UNCERTAINTY_PHRASES = re.compile(
    r"("
    r"i.m not sure|i don.t know|i.m unable|i cannot (confirm|verify|say|tell)"
    r"|as of my (knowledge|training|last update|cutoff)"
    r"|my (knowledge|training) (cutoff|ends|only goes)"
    r"|i don.t have (access|information|data)"
    r"|i cannot (access|browse|search|look up)"
    r"|this (may|might|could) (have changed|be outdated|be different)"
    r"|not in my (training|knowledge|database)"
    r"|beyond my (knowledge|training|cutoff|scope)"
    r"|i.m not aware|no (reliable|confirmed|verified) (information|data|details)"
    r"|speculating|this is a guess|i.m guessing"
    r"|i do not have real.?time|i lack (access|real.?time)"
    r"|i was (trained|last updated) (on|in|with)"
    r")",
    re.IGNORECASE,
)

def llm_answer_is_confident(question: str, answer: str) -> bool:
    # Layer 1: fast heuristic scan
    if UNCERTAINTY_PHRASES.search(answer):
        return False

    # Layer 2: strict LLM self-eval
    eval_prompt = (
        "You are a strict fact-checking evaluator. Decide if an AI answer is genuinely "
        "reliable or needs a web search to verify.\n\n"
        "Mark UNCERTAIN if ANY of these are true:\n"
        "- Question is about events, releases, results, or news from 2024 onwards\n"
        "- Question mentions a specific person, film, show, product, or event that may be post-2023\n"
        "- Answer contains hedging: maybe, probably, might, could, I think, I believe\n"
        "- Question is about sports results, scores, winners, rankings, or standings\n"
        "- Question is about reviews, ratings, or reception of a recently released work\n"
        "- The answer could have changed in the last 12 months\n"
        "- The answer is vague, generic, or avoids giving specific facts\n"
        "- The answer says it cannot find or does not have information\n\n"
        "Mark CONFIDENT only for stable, well-established facts: historical events before 2023, "
        "science, math, definitions, or general concepts that do not change.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        "Reply with exactly one word — CONFIDENT or UNCERTAIN. No explanation."
    )
    try:
        response = model.generate_content(eval_prompt)
        verdict = response.text.strip().upper()
        return verdict.startswith("CONFIDENT")
    except Exception:
        return False  # fail safe: go to web if evaluator errors


# ──────────────────────────────────────────────
# ANSWER STRATEGIES
# ──────────────────────────────────────────────
def answer_with_rag(query: str):
    docs = load_and_chunk_docs(DOCS_DIR)
    if not docs:
        return None
    index_struct = build_tfidf_index(docs)
    retrieved = score_query(query, index_struct)
    if not retrieved:
        return None
    context_text = "\n\n---\n\n".join(
        [f"Source: {src}\n{text[:800]}" for _, src, text in retrieved]
    )
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"CONTEXT FROM UPLOADED DOCUMENTS:\n{context_text}\n\n"
        f"USER QUESTION: {query}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.replace("*", "").strip() or None
    except Exception:
        return None


def answer_with_llm(query: str):
    prompt = f"{SYSTEM_INSTRUCTION}\n\nUSER QUESTION: {query}"
    try:
        response = model.generate_content(prompt)
        return response.text.replace("*", "").strip() or None
    except Exception:
        return None


def answer_with_tavily(query: str):
    """
    Uses Tavily Search API — purpose-built for AI retrieval.
    Returns a Gemini-synthesised answer from live web results.
    Get a free key at https://tavily.com
    """
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_API_KEY,
                "query": query,
                "search_depth": "advanced",   # deeper crawl for better results
                "include_answer": True,        # Tavily's own quick answer
                "include_raw_content": False,
                "max_results": 8,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # Build context from Tavily results
        snippets = []

        # Tavily's own synthesised answer (very useful)
        if data.get("answer"):
            snippets.append(f"Summary: {data['answer']}")

        # Individual result snippets
        for r in data.get("results", []):
            title   = r.get("title", "")
            content = r.get("content", "")
            url     = r.get("url", "")
            if content:
                snippets.append(f"[{title}] ({url})\n{content}")

        if not snippets:
            return None

        context_text = "\n\n---\n\n".join(snippets)
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"The following are live web search results for the user's question. "
            f"Use them to give an accurate, up-to-date answer. "
            f"If the results contain a direct answer, state it clearly.\n\n"
            f"WEB RESULTS:\n{context_text}\n\n"
            f"USER QUESTION: {query}"
        )
        response = model.generate_content(prompt)
        return response.text.replace("*", "").strip() or None

    except requests.exceptions.Timeout:
        st.warning("Web search timed out. Try again.")
        return None
    except Exception as e:
        st.warning(f"Web search failed: {e}")
        return None


# ──────────────────────────────────────────────
# MAIN ROUTING LOGIC
# Workflow: RAG -> LLM + confidence check -> Tavily web search
# ──────────────────────────────────────────────
def get_reply(query: str) -> tuple:
    """Returns (reply_text, source_label)"""
    has_docs = any(DOCS_DIR.glob("*"))

    # Force web search if flagged from previous turn
    if st.session_state.force_web_search:
        st.session_state.force_web_search = False
        reply = answer_with_tavily(query)
        return (reply or "Couldn't find anything on the web either. 😕", "web")

    # User is complaining about previous answer -> search web for the original topic
    if user_wants_web_search(query):
        last_user_q = next(
            (m["parts"][0] for m in reversed(st.session_state.conversation[:-1])
             if m["role"] == "user"),
            query,
        )
        reply = answer_with_tavily(last_user_q)
        return (reply or "Couldn't find anything on the web either. 😕", "web")

    # STEP 1: RAG (if docs uploaded and relevant)
    if has_docs:
        relevance = doc_relevance_score(query)
        if relevance > 0.05:
            reply = answer_with_rag(query)
            if reply:
                return (reply, "docs")

    # STEP 2: LLM + strict confidence check
    llm_reply = answer_with_llm(query)
    if llm_reply:
        if llm_answer_is_confident(query, llm_reply):
            return (llm_reply, "llm")
        # LLM not confident -> fall to web
        web_reply = answer_with_tavily(query)
        if web_reply:
            return (web_reply, "web")
        # Web also failed, return LLM answer with caveat
        return (
            llm_reply + "\n\n_(Note: I'm not fully certain — web search returned no results either.)_",
            "llm",
        )

    # STEP 3: Web search as final fallback
    reply = answer_with_tavily(query)
    return (reply or "Sorry, I couldn't find a good answer. Could you rephrase? 🤔", "web")


# ──────────────────────────────────────────────
# CHAT UI
# ──────────────────────────────────────────────
SOURCE_LABELS = {
    "docs": "📄 Answered from your documents",
    "llm":  "🧠 Answered from AI knowledge",
    "web":  "🌐 Answered from web search (Tavily)",
}

user_input = st.chat_input("Ask AI Assistant")

if user_input:
    st.session_state.conversation.append({"role": "user", "parts": [user_input]})

    with st.spinner(":blue[AI Assistant] is thinking..."):
        reply, source = get_reply(user_input)

    st.session_state.conversation.append({
        "role": "model",
        "parts": [reply],
        "source": source,
    })

for message in st.session_state.conversation[-6:]:
    if message["role"] == "user":
        st.chat_message("user").write(message["parts"][0])
    else:
        with st.chat_message("ai"):
            st.write(message["parts"][0])
            src = message.get("source")
            if src:
                st.caption(SOURCE_LABELS.get(src, ""))
