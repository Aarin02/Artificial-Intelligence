import streamlit as st
import google.generativeai as genai
from pathlib import Path
import re
import math
import pandas as pd
import wikipediaapi

gemini_API_KEY = st.secrets["gemini_API"]
genai.configure(api_key=gemini_API_KEY)
model = genai.GenerativeModel("gemini-3.5-flash")

ins = [
    "INSTRUCTION= You are a supportive,enthusiastic, straight to the point AI Assistant, have some personality and responses must sound human, use emojis but very so lightly "
]

st.set_page_config(page_title="AI Assistant")
st.title("Aarin's :blue[AI Assistant]")

if "instruction" not in st.session_state:
    st.session_state.instruction = [ins]
if "conversation" not in st.session_state:
    st.session_state.conversation = []

uploaded_files = st.file_uploader("Upload your text or CSV files", type=["txt", "csv"], accept_multiple_files=True)

DOCS_DIR = Path("uploaded_docs")
DOCS_DIR.mkdir(exist_ok=True)

if uploaded_files:
    for f in uploaded_files:
        file_path = DOCS_DIR / f.name
        with open(file_path, "wb") as out:
            out.write(f.read())

CHUNK_SIZE = 550
CHUNK_OVERLAP = 100
TOP_K = 4

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

# ✅ Custom splitter replacing langchain
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_and_chunk_docs(docs_dir: Path):
    docs = []
    for p in docs_dir.glob("*"):
        try:
            if p.suffix == ".csv":
                df = pd.read_csv(p)
                text = df.to_string()
            else:
                text = p.read_text(encoding="utf-8")
            chunks = split_text(text)
            for i, c in enumerate(chunks):
                if c.strip():
                    docs.append({"id": f"{p.name}__chunk{i}", "source": p.name, "text": c})
        except Exception:
            continue
    return docs

def build_tfidf_index(docs):
    N = len(docs)
    index, df, doc_lengths = {}, {}, {}
    for d in docs:
        tokens = tokenize(d["text"])
        doc_len = len(tokens)
        doc_lengths[d["id"]] = doc_len if doc_len > 0 else 1
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            tf[t] = 1 + math.log(tf[t])
            df[t] = df.get(t, 0) + 1
        index[d["id"]] = tf
    idf = {t: math.log((N + 1) / (freq + 1)) + 1 for t, freq in df.items()}
    return {"index": index, "idf": idf, "doc_lengths": doc_lengths, "docs_meta": {d["id"]: d for d in docs}}

def score_query(query, index_struct, top_k=TOP_K):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    q_tf = {t: 1 + math.log(q_tokens.count(t)) for t in set(q_tokens)}
    idf, index, doc_lengths, docs_meta = index_struct["idf"], index_struct["index"], index_struct["doc_lengths"], index_struct["docs_meta"]
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

def answer_with_rag(query: str):
    docs = load_and_chunk_docs(DOCS_DIR)
    if not docs:
        return None
    index_struct = build_tfidf_index(docs)
    retrieved = score_query(query, index_struct)
    if not retrieved:
        return None
    context_text = "\n\n---\n\n".join([f"Source: {src}\n{text[:800]}" for _, src, text in retrieved])
    prompt = f"{st.session_state.instruction[-1]}\n\nCONTEXT:\n{context_text}\n\nUSER QUESTION: {query}"
    try:
        response = model.generate_content(prompt)
        return response.text.replace("*", "").strip()
    except Exception:
        return None

wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

def answer_with_wiki(query: str):
    page = wiki_wiki.page(query)
    if not page.exists():
        return None
    summary = page.summary
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    extract = " ".join(sentences[:3])
    prompt = f"{st.session_state.instruction[-1]}\n\nWikipedia extract:\n{extract}\n\nUSER QUESTION: {query}"
    try:
        response = model.generate_content(prompt)
        return response.text.replace("*", "").strip()
    except Exception:
        return None

user_input = st.chat_input("Ask AI Assistant")

if user_input:
    st.session_state.conversation.append({"role": "user", "parts": [user_input]})
    reply = None

    with st.spinner(":blue[AI Assistant] is thinking..."):
        if uploaded_files:
            reply = answer_with_rag(user_input)
        if not reply:
            try:
                prompt = str(st.session_state.instruction[-1] + str(user_input))
                response = model.generate_content(prompt)
                reply = response.text.replace("*", "").strip()
            except Exception:
                reply = None
        if not reply or reply.strip() == "":
            reply = answer_with_wiki(user_input) or "Sorry, I couldn't find relevant info."

    st.session_state.conversation.append({"role": "model", "parts": [reply]})

for message in st.session_state.conversation[-6]:
    if message["role"] == "user":
        st.chat_message("user").write(message["parts"][0])
    elif message["role"] == "model":
        st.chat_message("ai").write(message["parts"][0])
