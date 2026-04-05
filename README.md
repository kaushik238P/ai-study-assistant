## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| LangChain (LCEL) | RAG pipeline |
| ChromaDB | Vector storage |
| Ollama (llama3.2) | Local LLM + Embeddings |
| PyMuPDF | PDF text extraction |
| Streamlit | Chat UI |

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/kaushik238P/ai-study-assistant.git
cd ai-study-assistant
```

**2. Install dependencies**
```bash
pip install langchain langchain-ollama langchain-chroma chromadb streamlit pymupdf pdfplumber
```

**3. Start Ollama**
```bash
ollama pull llama3.2
ollama serve
```

**4. Run the app**
```bash
streamlit run app.py
```

## ✨ Features
- 📄 Upload PDF or TXT files
- 💬 Chat with your document
- 🧠 Conversation memory across questions
- 🔍 Top-5 chunk retrieval for accurate answers
- 🗑️ Clear conversation and load new files

## 👨‍💻 Author
Kaushik — Mechanical Engineering student at SVNIT Surat, transitioning into AI Engineering.