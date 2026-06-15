# Aarin's AI Assistant

A Streamlit chatbot that answers questions from your uploaded documents first, then falls back to AI knowledge or live web search when needed.

---

## Features

- **Document Q&A** — Upload `.txt` or `.csv` files and the assistant searches them using a built-in TF-IDF retrieval engine before going anywhere else.
- **Smart fallback** — When docs don't have the answer, a popup lets you choose between AI knowledge or a live web search. Your preference is remembered for the session.
- **Web search** — Powered by the [Tavily API](https://tavily.com), triggered automatically for uncertain or outdated answers, or on demand.
- **Confidence checking** — The AI self-evaluates its answers and escalates to web search if it isn't sure.
- **Source labels** — Every reply is tagged: 📄 Documents, 🧠 AI Knowledge, or 🌐 Web Search.

---

## How it works

```
User question
      │
      ├─ Docs uploaded? ──Yes──► Relevant chunk found? ──Yes──► Answer from docs 📄
      │                                    │
      │                                   No
      │                                    │
      │                         Ask user: AI or Web? (once per session)
      │                                    │
      │                          ┌─────────┴──────────┐
      │                        "AI 🧠"             "Web 🌐"
      │                          │                    │
      │                    LLM answer          Tavily search
      │                    confident? ──No──► Web search
      │
      └─ No docs ──────────────────────────────► LLM → Web if uncertain
```

---
