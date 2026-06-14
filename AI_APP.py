import streamlit as st
import google.generativeai as genai
from pathlib import Path
import re
import math
import pandas as pd
from duckduckgo_search import DDGS

gemini_API_KEY = st.secrets["gemini_API"]
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

st.set_page_config(page_title="AI Assistant")
st.title("Aarin's :blue[AI Assistant]")

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "force_web_search" not in st.session_state:
    st.session_state.force_web_search = False

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
    q_tf  = {t: 1 + math.log(q_tokens.count(t)) for t in set(q_tokens)}
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

DISSATISFACTION_TRIGGERS = re.compile(
    r"\b("
    r"wrong|incorrect|not right|that('?s| is) (wrong|incorrect|not right|off)"
    r"|not helpful|doesn'?t help|didn'?t help"
    r"|try (again|the web|searching|online)"
    r"|search (the web|online|internet|for it)"
    r"|look it up|google it|find it online"
    r"|outdated|old (info|information|answer|data)"
    r"|you'?re (wrong|incorrect|mistaken|off)"
    r"|that'?s (not|wrong|incorrect|off|outdated)"
    r"|i don'?t think (that'?s|you'?re) right"
    r"|are you sure|double[- ]?check"
    r"|not (accurate|correct|right)"
    r")\b",
    re.IGNORECASE,
)

def user_wants_web_search(query: str) -> bool:
    return bool(DISSATISFACTION_TRIGGERS.search(query))

def llm_answer_is_confident(question: str, answer: str) -> bool:
    """
    Ask Gemini to self-evaluate: is this answer reliable or a guess?
    Returns True if confident, False if it should fall back to web search.
    """
    eval_prompt = (
        "You are an honest self-evaluation assistant.\n"
        "Given the question and an answer below, decide if the answer is:\n"
        "- CONFIDENT: factually reliable, complete, and not a guess\n"
        "- UNCERTAIN: potentially outdated, a guess, incomplete, or about something you don't clearly know\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        "Reply with exactly one word: CONFIDENT or UNCERTAIN."
    )
    try:
        response = model.generate_content(eval_prompt)
        verdict = response.text.strip().upper()
        return "CONFIDENT" in verdict
    except Exception:
        return True

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


def answer_with_llm(query: str) -> str | None:
    prompt = f"{SYSTEM_INSTRUCTION}\n\nUSER QUESTION: {query}"
    try:
        response = model.generate_content(prompt)
        return response.text.replace("*", "").strip() or None
    except Exception:
        return None


def answer_with_duckduckgo(query: str) -> str | None:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=10))
        if not results:
            return None
        extracts = [r.get("body", "") for r in results if r.get("body")]
        context_text = "\n\n---\n\n".join(extracts[:8])
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"The following are live web search results. "
            f"Use them to give an accurate, up-to-date answer.\n\n"
            f"WEB RESULTS:\n{context_text}\n\n"
            f"USER QUESTION: {query}"
        )
        response = model.generate_content(prompt)
        return response.text.replace("*", "").strip() or None
    except Exception as e:
        st.warning(f"Web search failed: {e}")
        return None

def get_reply(query: str) -> tuple[str, str]:
    """
    Returns (reply_text, source_label)
    source_label is one of: "docs", "llm", "web"
    """
    has_docs = any(DOCS_DIR.glob("*"))

    if st.session_state.force_web_search:
        st.session_state.force_web_search = False
        reply = answer_with_duckduckgo(query)
        return (reply or "Couldn't find anything on the web either. 😕", "web")

    if user_wants_web_search(query):
        last_question = next(
            (m["parts"][0] for m in reversed(st.session_state.conversation) if m["role"] == "user"),
            query
        )
        reply = answer_with_duckduckgo(last_question)
        return (reply or "Couldn't find anything on the web either. 😕", "web")
        
    if has_docs:
        relevance = doc_relevance_score(query)
        if relevance > 0.05:
            reply = answer_with_rag(query)
            if reply:
                return (reply, "docs")
                
    llm_reply = answer_with_llm(query)
    if llm_reply:
        if llm_answer_is_confident(query, llm_reply):
            return (llm_reply, "llm")
        else:
            web_reply = answer_with_duckduckgo(query)
            if web_reply:
                return (web_reply, "web")
            return (
                f"{llm_reply}\n\n_(Note: I'm not fully certain about this — "
                f"web search didn't return results either.)_",
                "llm"
            )

    reply = answer_with_duckduckgo(query)
    return (reply or "Sorry, I couldn't find a good answer. Could you rephrase? 🤔", "web")

SOURCE_LABELS = {
    "docs": "📄 Answered from your documents",
    "llm":  "🧠 Answered from AI knowledge",
    "web":  "🌐 Answered from web search",
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
