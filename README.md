<div align="center">

# 🎫 Support Ticket Assistant with RAG

**Stop triaging tickets manually. Let the machine do it.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20UI-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange)](https://groq.com)
[![Chroma](https://img.shields.io/badge/VectorDB-Chroma-6A5ACD)](https://www.trychroma.com)

> A production-ready AI pipeline that **classifies** support tickets in milliseconds, **retrieves** semantically similar past tickets, and **generates** plain-English explanations — all in one shot.

</div>

---

## 🧠 What Does It Actually Do?

When a support ticket arrives, this system does three things automatically:

```
📩 Incoming ticket
      │
      ▼
┌─────────────────────────────────┐
│  1. ML Classification           │  → Category + Confidence score
│     (TF-IDF / DistilBERT /      │
│      Ensemble)                  │
└──────────────┬──────────────────┘
               │  if confidence ≥ threshold
               ▼
┌─────────────────────────────────┐
│  2. Semantic Retrieval (RAG)    │  → Top-k similar historical tickets
│     Sentence Embeddings +       │
│     Chroma Vector Search        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  3. LLM Explanation             │  → Human-readable reasoning
│     Groq Llama 3.1 (8B)         │
└─────────────────────────────────┘
               │
               ▼
      📋 Final Output:
      Category · Confidence · Similar tickets · Explanation
```

---

## 🗂️ Ticket Categories

The system routes tickets into **5 departments** with >99% accuracy:

| 🏷️ Category | 📋 Covers |
|---|---|
| **Account** | Login failures, 2FA issues, password resets, access control |
| **Billing** | Invoices, refunds, subscription changes, payment failures |
| **Fraud** | Stolen cards, unauthorized transactions, hacked accounts |
| **General Inquiry** | How-to questions, feature requests, general help |
| **Technical** | App crashes, bugs, slow performance, API errors |

---

## ⚡ Model Benchmarks

Three classification models are available — pick based on your hardware:

| Model | Accuracy | F1 Score | Speed | GPU? |
|---|---|---|---|---|
| Baseline (TF-IDF + Logistic Regression) | 97.24% | 97.25% | 2–5 ms | ❌ |
| Transformer (DistilBERT fine-tuned) | 99.80% | 99.80% | 20–50 ms | ✅ |
| **Ensemble (recommended)** | **~99.90%** | **~99.90%** | 30–60 ms | ✅ |

---

## 🏗️ Project Layout

```
Support-Ticket-Assist-With-RAG/
│
├── 📁 data/
│   ├── raw/              ← Synthetic + Twitter ticket datasets
│   ├── processed/        ← Cleaned, merged, balanced data
│   └── embeddings/       ← Chroma vector DB persisted here
│
├── 📁 models/
│   └── saved/
│       ├── baseline/     ← TF-IDF + LR (pickle)
│       └── transformer/  ← Fine-tuned DistilBERT (HF format)
│
├── 📁 src/
│   ├── preprocessing/    ← Text cleaning, deduplication, merging
│   ├── models/           ← Training scripts for all 3 models
│   ├── rag/              ← Retriever · Indexer · LLM generator
│   ├── api/              ← FastAPI routes, schemas, classifier
│   ├── ui/               ← Streamlit (simple + agent views)
│   └── utils/            ← Config loader, structured logger
│
├── 📁 flask_app/         ← Flask production frontend (Bootstrap)
├── 📁 docs/              ← Deployment guide, MLOps pipeline
├── 📁 tests/             ← Unit tests (API, models, preprocessing)
├── requirements.txt
└── setup.py
```

---

## 🛠️ Tech Stack

```
┌─────────────────┬────────────────────────────────────────┐
│ Layer           │ Technology                             │
├─────────────────┼────────────────────────────────────────┤
│ Classification  │ TF-IDF + LR · DistilBERT · Ensemble   │
│ Embeddings      │ all-MiniLM-L6-v2 (Sentence-Transformers│
│ Vector Search   │ Chroma DB                              │
│ LLM             │ Groq API → Llama 3.1 8B Instruct       │
│ Backend API     │ FastAPI + Uvicorn                      │
│ Demo Frontend   │ Streamlit                              │
│ Prod Frontend   │ Flask + Bootstrap 5                    │
│ Monitoring      │ Prometheus metrics + JSON logs         │
│ Experiment Mgmt │ MLflow                                 │
│ Containerization│ Docker (optional)                      │
└─────────────────┴────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Step 1 — Clone & enter the project

```bash
git clone https://github.com/nikhil9verma/Support-Ticket-Assist-With-RAG.git
cd Support-Ticket-Assist-With-RAG
```

### Step 2 — Create a virtual environment

```bash
# macOS / Linux
python -m venv venv && source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set your Groq API key

```bash
# Copy the example env file
cp .env.example .env
# Then open .env and fill in:
# GROQ_API_KEY=your_key_here
```

### Step 5 — Preprocess data & build the vector index

```bash
python src/preprocessing/preprocessing_run.py
# Interactive menu: clean → twitter → merge → embed
```

### Step 6 — Train a model

```bash
# Fast CPU-only baseline
python src/models/run_models.py --model baseline --balance

# Fine-tuned transformer (needs GPU)
python src/models/run_models.py --model transformer --balance --auto-limit --transformer-max 10000
```

---

## ▶️ Running the System

### FastAPI Backend (required for all frontends)

```bash
python -m src.api.main
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

### Streamlit (demo / development UI)

```bash
# Simple one-click interface
streamlit run src/ui/streamlit_simple.py

# Advanced interface with analytics
streamlit run src/ui/streamlit_agent.py
```

### Flask (production interface)

```bash
cd flask_app && python app.py
# → http://localhost:5000
```

---

## 📡 API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/classify` | Classify a single ticket |
| `POST` | `/api/v1/classify/batch` | Classify a CSV of tickets |
| `POST` | `/api/v1/rag/explain` | Classify + retrieve similar + explain |
| `GET` | `/api/v1/health` | Liveness check (models + vector DB) |
| `GET` | `/api/v1/metrics` | Prometheus metrics scrape endpoint |

#### Quick example

```bash
curl -s -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Someone stole my credit card and made purchases", "model_type": "ensemble", "return_details": true}'
```

```json
{
  "category": "Fraud",
  "confidence": 0.995,
  "needs_review": false,
  "model_used": "ensemble"
}
```

---

## 📊 Monitoring & Observability

Every prediction emits a structured JSON log entry and a Prometheus counter. Key fields tracked:

- `request_id` — unique trace per ticket
- `latency_ms` — end-to-end processing time
- `confidence` — model certainty score
- `fraud_flag` — binary fraud alert
- `llm_fallback` — whether Groq was invoked

Metrics are scraped from `/api/v1/metrics`.

---

## ⚠️ Known Limitations

- **Groq dependency** — LLM explanation requires an internet connection and a valid API key.
- **Token cap** — Tickets longer than 512 tokens are truncated before classification.
- **Synthetic training data** — May not perfectly reflect all real-world ticket distributions.
- **Fraud edge cases** — Adversarial or obfuscated fraud attempts may slip through.

---

## 🔭 Roadmap

- [ ] Cloud deployment (AWS ECS / Google Cloud Run)
- [ ] Local LLM support via Ollama (offline mode)
- [ ] Human feedback loop for continuous fine-tuning
- [ ] Grafana dashboards over Prometheus
- [ ] Auth & rate limiting on the API layer

---

## 👥 Project Team

| Name |
|---|
| Hasham Abdelrahman |
| Enas Essam |
| Hossam Ashraf |
| Ahmed Magdy |
| Randa Hamada |

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — blazing-fast LLM inference API
- [Chroma](https://www.trychroma.com) — open-source vector database
- [Hugging Face](https://huggingface.co) — transformers, sentence-embeddings, model hub
- [FastAPI](https://fastapi.tiangolo.com) & [Streamlit](https://streamlit.io) — modern Python web frameworks

---

<div align="center">
<sub>Built for educational and research purposes &nbsp;·&nbsp; MIT License</sub>
</div>
