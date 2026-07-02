# RAG Practice on PDF Documents

> **English** · [한국어](README.ko.md)

Given the employment-rules PDF of a fictional company, **Gureumtech**, we build an internal Q&A that answers employees' questions so they don't have to dig through the rulebook themselves.
No framework (LangChain, etc.) — just `pypdf` + `openai` + `chromadb`, so you watch text turn into vectors and get retrieved firsthand.

```
question → embed → vector DB search → relevant chunks + question as prompt → LLM answer
```

## Order to Follow

There are **two notebooks**, done in order.

```
1) exercise/en/rag_pdf.ipynb       — Basic RAG. Set up the environment/kernel here and see the RAG flow from scratch.
2) exercise/en/advanced_rag.ipynb  — Advanced RAG. Reuses the same kernel from 1) and stacks improvements one by one.
```

**Do `rag_pdf.ipynb` first, without exception** — the environment setup and kernel (**RAG PDF Practice**) are created there, and the advanced notebook uses that same kernel.

## Basic RAG - `rag_pdf.ipynb`

This is the hands-on for the blog post **[So, What is RAG](https://seojuny.dev/en/what-is-rag)**. With a single Gureumtech employment-rules PDF, and no framework — just `pypdf` + `openai` + `chromadb` — you check for yourself, through the hands-on practice, how text becomes vectors, gets retrieved, and produces an answer.

The flow has two stages.

```
Indexing  (1–3)  extract text from the doc → chunk → embed and store in the vector DB (Chroma)
Answering (4–5)  embed the question → similarity search (top-k) → relevant chunks + question as prompt → LLM answer
```

Open `exercise/en/rag_pdf.ipynb` and run the cells top to bottom.
**Everything — environment setup (Python · pipenv · kernel registration · model selection) and how to experiment by changing values — is written right in the notebook's top `0. Environment Setup` cell and the `CONFIG` cell.**

- If `OPENAI_API_KEY` is set it uses OpenAI; otherwise it falls back to a local Ollama (free) automatically.
- What it covers: text extraction (`pypdf`) · character-count chunking (`CHUNK_SIZE` · `OVERLAP`) · embedding and Chroma storage · similarity search (top-k, with cosine-similarity scores shown) · answer generation (`SYSTEM_PROMPT` to suppress hallucination) · a **with-RAG vs. without-RAG** comparison · hands-on experiments by changing `CONFIG` values.
- To use your own PDF, drop the file into `data/en/` and change only `PDF_PATH` in `CONFIG`.
- When you change values: `CHUNK_SIZE` · `OVERLAP` change the vector DB itself, so re-run the **2 (chunking) → 3 (indexing)** cells; `N_RESULTS` · `SYSTEM_PROMPT` · `TEMPERATURE` are used only at query time, so re-run just the **question cell**.

## Advanced Practice — `advanced_rag.ipynb`

This is the hands-on for the blog post **[Taking RAG Further](https://seojuny.dev/en/advanced-rag)**. Start from the simplest **basic RAG (STAGE 0)**, find what's lacking, apply improvements **one at a time**, and **compare each answer with the previous stage**. You implement each improvement yourself and see firsthand what gets better, and by how much.

```
STAGE 0 basic → +table preservation → +structural chunking → +Contextual → +query rewriting → +hybrid search → +reranking → +source citation = advanced
```

- **You must complete `exercise/en/rag_pdf.ipynb` first** — the environment setup and kernel live there, and this reuses the same **RAG PDF Practice** kernel.
- The dataset is **separate** from `rag_pdf` — `data_advanced/en/` holds **7 PDFs** of internal documents from the fictional company **Novatech** (employment rules, benefits, pay policy, security policy, HR policy, travel-expense policy, internal FAQ). Many pages contain multiple tables, and the formats vary — regulations, notices, FAQs, and more. Tables are extracted as Markdown tables via `pymupdf4llm` so they survive.
- Techniques covered: table-preserving extraction, structural chunking, Contextual Retrieval, query rewriting (decomposition implemented from scratch), hybrid search, metadata filtering, reranking, source citation, agentic RAG, and evaluation. At the end, you look over code that implements the same pipeline with LangChain.
- It uses proven libraries: `rank-bm25` (BM25), `sentence-transformers` (reranking, `bge-reranker-v2-m3`), `langchain` (the LangChain wrap-up implementation), and `ragas` (evaluation). The notebook's top **"install libraries" cell** installs them all **at compatible versions in one go** — `langchain` is **pinned to 0.3** so it meshes with `ragas` and the retriever components. Both the RAGAS and LangChain wrap-ups run directly in the notebook.
- It uses an in-memory vector DB (`EphemeralClient`), so experiments stay isolated from one another (nothing is left on disk).

## Project Layout

```
rag-pdf-practice/
├── exercise/
│   ├── rag_pdf.ipynb              # Korean — Basic RAG
│   ├── advanced_rag.ipynb         # Korean — Advanced
│   └── en/                        # English versions ← this README points here
│       ├── rag_pdf.ipynb          # 1) Basic RAG (0. setup → CONFIG → 5 steps → experiment)
│       └── advanced_rag.ipynb     # 2) Advanced (STAGE 0 basic → stack improvements → advanced)
├── Pipfile                        # dependency definitions (read by pipenv install)
├── data/
│   ├── sample-취업규칙.pdf         # Korean employment rules (for exercise/rag_pdf.ipynb)
│   └── en/
│       └── employment-rules.pdf   # English — Gureumtech employment rules (for exercise/en/rag_pdf.ipynb)
├── data_advanced/                 # Novatech internal docs — Korean (for exercise/advanced_rag.ipynb)
│   ├── 취업규칙.pdf … 사내FAQ.pdf  # 7 docs: rules, benefits, pay, security, HR, travel, FAQ
│   └── en/                        # English — 7 docs (for exercise/en/advanced_rag.ipynb)
│       ├── employment-rules.pdf       # rules (many chapters/articles), [appendix] leave summary table
│       ├── benefits.pdf               # notice + tables for events, welfare points, checkups
│       ├── pay-policy.pdf             # several tables: base pay, seniority add-ons, allowances
│       ├── security-policy.pdf        # info-grade table, equipment-code table (SEC-VPN-01, etc.)
│       ├── hr-policy.pdf              # job-grade and rating tables
│       ├── travel-expense-policy.pdf  # domestic/overseas travel-expense tables
│       └── internal-faq.pdf           # Q&A format
└── .env.example
```

> The documents in `data/` and `data_advanced/` are not real regulations — they're fictional documents made up for practice. The `en/` folders hold English translations of the same documents.
> `pipenv install` installs the exact versions pinned in `Pipfile.lock`, so everyone gets the same environment. The heavier libraries for the advanced practice (`langchain`, `ragas`, `sentence-transformers`, etc.) are installed at compatible versions by the "install libraries" cell at the top of `advanced_rag.ipynb`.

---

This practice code and the notebooks were written together with **[Claude](https://claude.ai)** (Anthropic).
