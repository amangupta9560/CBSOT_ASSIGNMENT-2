<div align="center">

# 🧠 ResearchMind AI 📚

### Smart Research Paper Assistant — RAG, Deep Learning & AI Agents

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&pause=1000&color=4C72B0&center=true&vCenter=true&width=650&lines=Reading+Research+Papers+at+Scale...;RAG-Powered+Q%26A+over+PDFs...;Auto-Generating+Literature+Reviews...;Knowledge+Graphs+%2B+AI+Agents+%F0%9F%A4%96" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-4C72B0?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM_Inference-F55036?style=for-the-badge)
![Transformers](https://img.shields.io/badge/🤗_Transformers-DistilBERT-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

![Status](https://img.shields.io/badge/Status-Portfolio--Ready-brightgreen?style=flat-square)
![Notebook](https://img.shields.io/badge/Runs%20On-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Cells](https://img.shields.io/badge/Notebook%20Cells-47-blue?style=flat-square)
![Languages](https://img.shields.io/badge/Language-Python%20100%25-3776AB?style=flat-square&logo=python)

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14EnOLUIVr9Xyq9_Ar9Q7ucbu2PiyL_Ui?usp=sharing)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Live Demo Preview](#-live-demo-preview)
- [What Makes This Project Unique](#-what-makes-this-project-unique)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#-tech-stack)
- [Notebook Walkthrough](#-notebook-walkthrough)
  - [PDF Processing Pipeline](#1-pdf-processing-pipeline)
  - [RAG Pipeline](#2-rag-pipeline-retrieval-augmented-generation)
  - [Deep Learning Classifier](#3-deep-learning-classification-module)
  - [Knowledge Graph](#4-knowledge-graph-generation)
  - [AI Agent System](#5-ai-agent-system)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Sample Interaction](#-sample-interaction)
- [Performance Comparison](#-performance-comparison)
- [Evaluation Metrics](#-evaluation-metrics)
- [Engineering Practices](#-engineering-practices)
- [Roadmap](#-roadmap)
- [Key Learnings](#-key-educational-takeaways--learned-skills)
- [Internship & Author Details](#-internship--author-details)
- [Contributing](#-contributing)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Overview

> **Researchers spend enormous amounts of time reading, comparing, and summarizing papers.** ResearchMind AI automates the heavy lifting of literature work — from raw PDF to grounded, cited answers — in a single, end-to-end Colab notebook.

This project demonstrates a **full-stack AI engineering pipeline** spanning:

| Component | Type | Purpose |
|---|---|---|
| 📄 **PDF Extraction Pipeline** | Classical NLP + Regex Heuristics | Pulls title, authors, abstract, body, references from raw PDFs |
| 🔎 **RAG Engine** | Sentence-Transformers + FAISS + Groq LLM | Grounded Q&A, summarization, comparison, literature review |
| 🧬 **Deep Learning Classifier** | Fine-tuned DistilBERT | Categorizes papers by ML sub-field |
| 🕸️ **Knowledge Graph** | NetworkX | Maps papers ↔ authors ↔ concepts |
| 🤖 **AI Agent System** | Lightweight Orchestrator | Routes tasks to specialized agents |
| 🖥️ **Interactive UI** | ipywidgets | Upload, ask, search, summarize, compare — all in-notebook |

<div align="center">

```
📄 PDF Upload ──▶ 🧹 Extraction ──▶ ✂️ Chunking ──▶ 🧠 Embeddings ──▶ 🗂️ FAISS
                                                                          │
        ┌─────────────────────────────────────────────────────────────┘
        ▼
🤖 Groq LLM (RAG) ──▶ 📝 Summaries / 📊 Comparison / 📚 Lit Review / 🕳️ Gap Detection
        │
        ▼
🕸️ Knowledge Graph  +  🎛️ Agent Orchestrator  +  🖥️ ipywidgets UI
```

</div>

---

## 🌟 What Makes This Project Unique

Unlike most "PDF chatbot" tutorials that stop at basic Q&A, ResearchMind AI combines **six distinct AI disciplines** into one coherent, portfolio-grade system:

| # | Uniqueness Factor | Why It Matters |
|---|---|---|
| 1 | 🔗 **Full-stack integration** — RAG, fine-tuning, graphs, and agents in *one* notebook | Most projects show only one of these in isolation |
| 2 | 🧠 **Model-choice flexibility** — swap between 3 Groq LLMs per task | Lets you trade off speed vs. reasoning depth per use case |
| 3 | 🕵️ **Cross-paper reasoning** — research gap detection analyzes *all* papers jointly, not one at a time | Surfaces gaps a single-paper summarizer would miss |
| 4 | 🤖 **Agent-based task routing** — a lightweight orchestrator dispatches work to specialized agents | Mirrors real production multi-agent architecture patterns |
| 5 | 🕸️ **Auto-built knowledge graph** — papers, authors, and concepts linked automatically from extracted text | No manual annotation required |
| 6 | 🖥️ **Zero-install interactivity** — a full ipywidgets control panel runs directly inside Colab | No separate frontend/backend deployment needed to demo it |
| 7 | 📊 **Built-in evaluation loop** — retrieval Precision/Recall/F1 *and* LLM-as-judge scoring | Most RAG demos skip evaluation entirely |

---

## 🖥️ Live Demo Preview

<div align="center">

| 📊 Chunk & Embedding Analytics | 🕸️ Knowledge Graph | 🖥️ Interactive UI |
|:---:|:---:|:---:|
| Histograms of chunk length & embedding norms | Papers ↔ Authors ↔ Concepts | Ask / Search / Summarize / Compare buttons |

</div>

> 💡 Open the notebook in Google Colab, paste your free [Groq API key](https://console.groq.com/keys), upload a few papers, and run top-to-bottom — every section streams live output, plots, and tables.

---

## ✨ Key Features

- 📤 **Multi-PDF upload** directly inside Colab (`google.colab.files`)
- 🧹 **Structured extraction** — title, authors, abstract, body, references via PyMuPDF + regex heuristics
- ✂️ **Recursive overlapping chunking** with metadata preservation
- 🧠 **Sentence-Transformer embeddings** (`all-MiniLM-L6-v2`)
- 🗂️ **FAISS vector index** with save/load persistence
- 🤖 **Full RAG pipeline** — Document → Chunk → Embed → FAISS → Retriever → Groq LLM → Answer
- 📝 **4-level summarization** — short, detailed, beginner-friendly, bullet-point
- 📊 **Multi-paper comparison engine** with structured Pandas output
- 📚 **Literature review generator** from a single topic prompt
- 🕳️ **Cross-paper research gap detection**
- 🔖 **Citation generator** — APA / MLA / IEEE / Chicago
- 🔎 **Semantic search** across all uploaded papers
- 🧬 **Fine-tuned DistilBERT classifier** (6 ML sub-fields) with confusion matrix
- 🌟 **Recommendation engine** via cosine similarity + KMeans clustering
- 🕸️ **Auto-built knowledge graph** (papers, authors, concepts)
- 🤖 **5-agent orchestration system** (Search, Summarize, Cite, Review, Gap agents)
- 🖥️ **ipywidgets control panel** for point-and-click interaction
- 📈 **Retrieval evaluation** — Precision, Recall, F1 + LLM-as-judge scoring

---

## 🏗️ Project Architecture

```text
ResearchMind_AI.ipynb
│
├── 1️⃣  Introduction & Architecture Overview
├── 2️⃣  Install Dependencies
├── 3️⃣  Import Libraries
├── 4️⃣  Groq API Configuration
├── 5️⃣  Upload Research Papers (Colab uploader)
├── 6️⃣  PDF Processing Pipeline           → title / authors / abstract / body / refs
├── 7️⃣  Text Chunking                     → recursive, overlapping, metadata-tagged
├── 8️⃣  Embedding Generation               → all-MiniLM-L6-v2
├── 9️⃣  FAISS Vector Database              → indexing, search, persistence
├── 🔟  RAG Pipeline                       → ask_question()
├── 1️⃣1️⃣ Summarization Engine              → short / detailed / beginner / bullets
├── 1️⃣2️⃣ Multi-Paper Comparison            → structured Pandas table
├── 1️⃣3️⃣ Literature Review Generator
├── 1️⃣4️⃣ Research Gap Detection
├── 1️⃣5️⃣ Citation Generator                → APA / MLA / IEEE / Chicago
├── 1️⃣6️⃣ Semantic Search Engine            → search_papers()
├── 1️⃣7️⃣ Deep Learning Classifier          → fine-tuned DistilBERT
├── 1️⃣8️⃣ Recommendation System              → cosine similarity + KMeans
├── 1️⃣9️⃣ Knowledge Graph Generation         → NetworkX + Matplotlib
├── 2️⃣0️⃣ AI Agent System                    → Orchestrator + 5 agents
├── 2️⃣1️⃣ Interactive UI                     → ipywidgets dashboard
├── 2️⃣2️⃣ Additional Visualizations
├── 2️⃣3️⃣ Evaluation                         → Precision / Recall / F1
└── 2️⃣4️⃣ Future Improvements
```

---

## 🧰 Tech Stack

<div align="center">

| Category | Tools |
|---|---|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **LLM Inference** | ![Groq](https://img.shields.io/badge/-Groq-F55036?style=flat-square) `llama-3.3-70b-versatile` · `deepseek-r1-distill-llama-70b` · `gemma2-9b-it` |
| **Orchestration** | ![LangChain](https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square) |
| **Embeddings** | ![Sentence Transformers](https://img.shields.io/badge/-Sentence--Transformers-4C72B0?style=flat-square) `all-MiniLM-L6-v2` |
| **Vector Store** | ![FAISS](https://img.shields.io/badge/-FAISS-0467DF?style=flat-square) |
| **PDF Parsing** | ![PyMuPDF](https://img.shields.io/badge/-PyMuPDF-DD8452?style=flat-square) |
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![Transformers](https://img.shields.io/badge/-🤗_Transformers-FFD21E?style=flat-square) |
| **Graph & Viz** | ![NetworkX](https://img.shields.io/badge/-NetworkX-55A868?style=flat-square) ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) |
| **UI** | ![ipywidgets](https://img.shields.io/badge/-ipywidgets-F37626?style=flat-square) |
| **Data** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) |

</div>

---

## 🔬 Notebook Walkthrough

### 1. PDF Processing Pipeline

Extracts structured fields from raw PDFs using **PyMuPDF** + heuristic regex parsing:

| Field | Extraction Method |
|---|---|
| 📌 Title | First reasonably-sized, non-empty line on page 1 |
| ✍️ Authors | Line(s) immediately following the detected title |
| 📝 Abstract | Regex span between `"abstract"` and `"introduction"`/`"keywords"` |
| 📖 Body Text | Everything from `"introduction"` onward |
| 🔗 References | Regex span between `"references"`/`"bibliography"` and `"appendix"` |

Includes automatic **noise cleanup**: de-hyphenation of line-broken words, whitespace collapsing, and broken-line merging.

<details>
<summary>📂 <b>Click to expand — chunking configuration</b></summary>

<br/>

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

Each chunk retains `source` (filename) and `chunk_id` metadata for traceable retrieval.

</details>

### 2. RAG Pipeline (Retrieval-Augmented Generation)

```
Document → Chunking → Embedding → FAISS → Retriever → Groq LLM → Response
```

```python
def ask_question(question: str, k: int = 5, model_key: str = "fast") -> str:
    retrieved = similarity_search(question, k=k)
    context = build_context(retrieved)
    ...
```

Answers are **grounded strictly in retrieved context** — the system is instructed to say so honestly if the answer isn't present, rather than hallucinate.

### 3. Deep Learning Classification Module

Fine-tunes **DistilBERT** across 6 categories:

<div align="center">

`Machine Learning` · `Deep Learning` · `NLP` · `Computer Vision` · `Cyber Security` · `Generative AI`

</div>

Includes a full **Trainer**-based fine-tuning loop, accuracy/precision/recall/F1 reporting, and a rendered **confusion matrix**.

### 4. Knowledge Graph Generation

```
Paper ──authored_by──▶ Author
Paper ──discusses────▶ Concept (e.g. Transformer, BERT, GPT)
```

Built with **NetworkX**, visualized with **Matplotlib** using color-coded node types (papers / authors / concepts).

### 5. AI Agent System

| Agent | Responsibility |
|---|---|
| 🔎 `SearchAgent` | Semantic search across chunks |
| 📝 `SummarizationAgent` | Multi-style paper summarization |
| 🔖 `CitationAgent` | APA/MLA/IEEE/Chicago citation generation |
| 📚 `LiteratureReviewAgent` | Structured literature review drafting |
| 🕳️ `ResearchGapAgent` | Cross-paper gap analysis |

An `Orchestrator` class routes natural-language task strings to the correct agent.

---

## 🚀 Installation & Setup

### 1️⃣ Open in Google Colab

Upload `ResearchMind_AI.ipynb` to [Google Colab](https://colab.research.google.com/).

### 2️⃣ Install Dependencies

```bash
pip install langchain langchain-community langchain-groq \
    sentence-transformers faiss-cpu transformers torch accelerate \
    pymupdf networkx matplotlib scikit-learn pandas numpy ipywidgets groq
```

*(Already included as the first executable cell in the notebook.)*

### 3️⃣ Configure Your Groq API Key

```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY"  # https://console.groq.com/keys
```

### 4️⃣ Upload Papers & Run

Run all cells top-to-bottom. Upload PDFs when prompted in **Section 4**.

<div align="center">

✅ **That's it — from raw PDFs to RAG-powered answers in one run.**

</div>

---

## 📘 Usage Guide

| Task | Function | Output |
|---|---|---|
| Ask a question | `ask_question("What is the methodology?")` | Grounded LLM answer |
| Summarize a paper | `summarize_paper(doc, style="bullets")` | Bullet-point summary |
| Compare papers | `comparison_df` | Pandas comparison table |
| Generate lit review | `generate_literature_review("Large Language Models")` | Structured markdown review |
| Detect research gaps | `detect_research_gaps()` | Cross-paper gap analysis |
| Generate citations | `generate_citations(doc)` | APA/MLA/IEEE/Chicago dict |
| Semantic search | `search_papers("transformers")` | Ranked results DataFrame |
| Classify a paper | `classify_paper(doc)` | Predicted ML sub-field |
| Get recommendations | `recommend_similar(doc_index=0)` | Similar chunks DataFrame |
| Route via agent | `orchestrator.route("summarize", doc=doc)` | Delegated agent output |

---

## 🔬 Sample Interaction

```
❓ Question: "What are the limitations of this paper?"

🤖 ResearchMind AI:
The paper acknowledges limited evaluation on low-resource languages
and notes that retrieval quality degrades on very long documents due
to fixed-size chunking. [Source: paper1.pdf]
──────────────────────────────────────────────
📊 Retrieval metrics: {'precision': 0.8, 'recall': 1.0, 'f1': 0.889}
```

---

## 🏎️ Performance Comparison

### LLM Backend Comparison (Groq Models)

<div align="center">

| Model | Best For | Relative Speed | Reasoning Depth | Notebook Sections Used |
|:---:|:---|:---:|:---:|:---|
| `llama-3.3-70b-versatile` | General Q&A, summarization, citations | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | RAG, Summarization, Concept Extraction |
| `deepseek-r1-distill-llama-70b` | Literature review & research gap reasoning | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | Literature Review, Gap Detection |
| `gemma2-9b-it` | Lightweight, low-latency tasks | ⚡⚡⚡⚡ Very Fast | ⭐⭐ Basic | Optional fallback for simple prompts |

</div>

### Embedding Model Comparison

<div align="center">

| Model | Dimensions | Speed | Retrieval Quality | Chosen? |
|:---:|:---:|:---:|:---:|:---:|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡⚡ Very Fast | ⭐⭐⭐⭐ Strong | ✅ Used in this project |
| `all-mpnet-base-v2` | 768 | ⚡⚡ Slower | ⭐⭐⭐⭐⭐ Best-in-class | ❌ Higher latency, overkill for demo scale |
| `SciBERT` / `SPECTER` | 768 | ⚡⚡ Slower | ⭐⭐⭐⭐⭐ Best for scientific text | 🔜 Suggested future upgrade (see Roadmap) |

</div>

### Custom CNN vs. Fine-Tuned DistilBERT (Classification Module)

<div align="center">

| Approach | Training Time | Accuracy on Demo Set | Notes |
|:---:|:---:|:---:|:---|
| **DistilBERT (fine-tuned)** | Fast (3 epochs, small dataset) | High | Used in this notebook — transfer learning from pretrained language understanding |
| Training a transformer **from scratch** | Very slow | Low on small datasets | Not viable without large labelled corpora |
| **Classical ML** (TF-IDF + Logistic Regression) | Very fast | Lower on nuanced text | Good baseline, weaker on semantic nuance |

</div>

---

## 📈 Evaluation Metrics

<div align="center">

| Metric | Purpose |
|:---:|:---|
| **Precision** | Fraction of retrieved chunks that are actually relevant |
| **Recall** | Fraction of relevant chunks successfully retrieved |
| **F1 Score** | Harmonic mean of precision & recall |
| **LLM-as-Judge** | 1–5 groundedness/relevance rating of generated answers |

</div>

*(See Section 22 of the notebook for live computed values on your uploaded papers.)*

---

## 🧹 Engineering Practices

- 🧱 **Dataclasses** (`PaperDocument`, `Chunk`) for clean, typed data structures
- 🛡️ **Exception handling** around every external API call (Groq, FAISS)
- ♻️ **Reusable functions** — every capability is a standalone, testable function
- 🧩 **Object-oriented agent layer** for extensibility (`BaseAgent` → specialized agents)
- 💾 **FAISS index persistence** (`faiss.write_index` / `read_index`)
- 📊 **Progressive visualization** — every major stage renders its own diagnostic plot

---

## 🗺️ Roadmap

- [ ] 🕸️ Multi-agent planning/critique loops (e.g. LangGraph)
- [ ] 🗣️ Voice assistant integration (speech-to-text / text-to-speech)
- [ ] 🌐 Arxiv API integration for automatic related-paper fetching
- [ ] 🔮 Citation prediction via citation-graph embeddings
- [ ] 🛣️ Research roadmap generation from detected gaps
- [ ] 🗄️ Persistent vector store (Chroma / Pinecone) across sessions
- [ ] 🧬 Domain-specific embeddings (SciBERT / SPECTER)

---

## 🎓 Key Educational Takeaways & Learned Skills

- 🧠 **RAG System Design** — chunking strategy, embedding choice, and retriever tuning
- 🏗️ **Transformer Fine-Tuning** — end-to-end DistilBERT training with Hugging Face `Trainer`
- 🕸️ **Knowledge Graph Construction** — entity/relation modeling with NetworkX
- 🤖 **Multi-Agent Orchestration** — task routing across specialized capability agents
- 🖥️ **In-Notebook UX Design** — building a usable control panel with ipywidgets

---

## 🎓 Internship & Author Details

<div align="center">

| Field                   | Detail                                                                 |
| ----------------------- | ---------------------------------------------------------------------- |
| 👤 **Author / Intern**  | **Aman Gupta**                                                         |
| 🎓 **Institution**      | *Hi-Tech Institute Of Engineering and Technology Ghaziabad*            |
| 📚 **Program / Branch** | *B.Tech Computer Science*                                              |
| 🏢 **Internship Role**  | **AI/ML Engineering Intern**                                           |
| 🧭 **Core Domain**      | **Generative AI, RAG, NLP, Deep Learning & Full-Stack AI Engineering** |
| ⏳ **Duration**          | **6 Weeks — Summer Session**                                           |
| 📅 **Year**             | **2026**                                                               |
| 📌 **Project Status**   | **Portfolio-Ready & Fully Executable in Google Colab**                 |
| 📧 **Contact**          | *[Email](mailto:ag0567688@gmail.com)*              |
| 🔗 **LinkedIn**         | *https://www.linkedin.com/in/amangupta9454/*                                         |

</div>


---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔁 Open a Pull Request

---

## 👥 Contributors

<div align="center">

| Avatar | Name | Role |
|:---:|:---:|:---:|
| 🧑‍💻 | **Aman Gupta** | AI/ML Engineer — Author & Maintainer of ResearchMind AI |

</div>

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">

### 📚 Built to make research reading smarter, faster, and grounded in evidence 🧠

⭐ **If this project helped you, consider giving it a star!** ⭐

![Visitor Count](https://komarev.com/ghpvc/?username=researchmind-ai&label=Repo%20Views&color=4C72B0&style=flat-square)

</div>
