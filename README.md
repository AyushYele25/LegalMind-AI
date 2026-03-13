# ⚖️ LegalMind AI — India's Intelligent Legal Assistant

An AI-powered legal assistant that helps common Indians understand their legal rights in simple language. Built with RAG (Retrieval-Augmented Generation) using LLaMA 3.3 70B via Groq.

---

## 🏛️ Laws Covered (14 Laws)

| Category | Laws |
|---|---|
| Consumer | Consumer Protection Act, 2019 |
| Criminal | Indian Penal Code, 1860 · IT Act, 2000 · POCSO Act, 2012 |
| Labour | Code on Wages, 2019 · Industrial Disputes Act, 1947 · EPF Act, 1952 |
| Civil | Indian Contract Act, 1872 · RTI Act, 2005 |
| Women & Children | Domestic Violence Act, 2005 · POCSO Act, 2012 |
| Housing | Model Tenancy Act, 2021 · Maharashtra Rent Control Act, 1999 |
| Safety | Food Safety Act, 2006 · Motor Vehicles Act, 1988 |

---

## ✨ Features

- 🔍 **RAG Pipeline** — Retrieves relevant legal sections before answering
- 📖 **Source Citations** — Every answer cites the exact law and page number
- 🔄 **Streaming Responses** — Real-time token-by-token streaming
- 💬 **Conversation Memory** — Remembers context within a session
- ⚠️ **Error Handling** — Graceful handling of API failures
- 🇮🇳 **Built for India** — Simple English explanations for common people

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector DB | ChromaDB (persistent) |
| UI | Gradio 6.9 |
| PDF Parsing | PyMuPDF (fitz) |

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/LegalMind-AI.git
cd LegalMind-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at: https://console.groq.com

### 4. Add PDF laws to `data/` folder
Download PDFs from https://indiacode.nic.in and place them in the `data/` folder.

### 5. Ingest the PDFs
```bash
python ingest.py
```

### 6. Run the app
```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## 📁 Project Structure

```
LegalMind-AI/
├── app.py          # Gradio UI with streaming
├── chatbot.py      # RAG + Groq streaming logic
├── ingest.py       # PDF ingestion pipeline
├── data/           # PDF law files (not tracked in git)
├── embeddings/     # ChromaDB vector store (not tracked in git)
├── requirements.txt
├── .env            # API keys (not tracked in git)
└── README.md
```

---

## 💡 Example Questions

- *"I bought a phone that stopped working after 2 days. What are my rights?"*
- *"My employer hasn't paid salary for 2 months. What can I do?"*
- *"Someone hacked my bank account. What can I do under IT Act?"*
- *"How do I file an RTI application?"*
- *"My landlord is illegally evicting me. What are my rights?"*

---

## ⚠️ Disclaimer

LegalMind AI provides general legal information only. It is not a substitute for professional legal advice. Always consult a qualified lawyer for serious legal matters.

---

## 👨‍💻 Built by

Ayush — Built with ❤️ for every Indian who deserves to know their rights.
