# 🧠 Digital Second Brain: Handwritten Notes to Knowledge Base

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![AI](https://img.shields.io/badge/AI-Ollama%20%7C%20DeepSeek%20%7C%20Gemma-purple.svg)
![Stack](https://img.shields.io/badge/Stack-Gradio%20%7C%20ChromaDB-orange.svg)

**Digital Second Brain** is an AI-powered knowledge management system that bridges the gap between analog handwriting and digital intelligence. It ingests handwritten notes, converts them into searchable text using Optical Character Recognition (OCR), and organizes them into a **Semantic Knowledge Graph**.

> **Portfolio Highlight:** This project demonstrates the integration of **Vision Models** (DeepSeek-OCR), **Vector Databases** (ChromaDB), and **Interactive Graph Theory** (NetworkX) in a local-first architecture.

---

## 🚀 Key Features

### 1. 📸 Handwriting-to-Knowledge Pipeline
- **Input:** Upload images or snap photos of handwritten journal entries/meeting notes.
- **Process:** Utilizes **DeepSeek-OCR** to transcribe complex handwriting with high fidelity.
- **Refinement:** Uses **GLM-4** to fix OCR typos, auto-tag content, and generate concise summaries.

### 2. 🧠 RAG (Retrieval Augmented Generation)
- **Chat with your Notebook:** Ask questions like *"What were the action items from the budget meeting?"*
- **Vector Search:** Powered by **EmbeddingGemma** and **ChromaDB** to perform semantic searches (finding concepts, not just keywords).

### 3. 🕸️ Interactive Knowledge Graph
- **Visual Thinking:** Automatically builds a dynamic network graph connecting notes via shared tags.
- **Discovery:** Reveals hidden connections between disparate ideas (e.g., linking "Project Alpha" notes with "Q3 Financials").

### 4. 🔌 Hybrid AI Architecture (BYOK)
- **Local First:** Runs 100% locally using Ollama for privacy.
- **Cloud Ready:** "Bring Your Own Key" support for **OpenAI (GPT-4o)**, **Anthropic (Claude 3.5)**, and **Google (Gemini 1.5)**.

---

## 🛠️ Tech Stack

- **Frontend:** Gradio (Web UI, Image Editing, Chat Interface)
- **Backend:** Python
- **AI Engine:** Ollama (Local LLMs)
- **Vector DB:** ChromaDB (Persistent local storage)
- **Graphing:** NetworkX + PyVis (HTML5 Visualization)

---

## 📦 Installation

### Prerequisites
1. **Python 3.10+**
2. **Ollama** installed and running.

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pypi-ahmad/digital-second-brain.git
   cd digital-second-brain
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull Required Local Models**
   Ensure you have these models loaded in Ollama:
   ```bash
   ollama pull deepseek-ocr
   ollama pull glm-4.7-flash
   ollama pull embeddinggemma
   ```

---

## 🏃‍♂️ Usage

1. **Start the App**
   ```bash
   python app.py
   ```

2. **Open Browser:** Go to `http://127.0.0.1:7860`
3. **Workflow:**
   * **Tab 1 (Scan):** Upload a photo of your notes. Click "Convert & Save".
   * **Tab 2 (Search):** Type a concept to find relevant notes.
   * **Tab 3 (Chat):** Ask the AI questions about your stored knowledge.
   * **Tab 4 (Graph):** Click "Refresh Graph" to see your knowledge network grow.

---

## 📂 Project Structure

```text
digital-second-brain/
├── app.py               # Gradio Frontend UI
├── backend.py           # Logic, DB, & AI Wrapper
├── chroma_db/           # Local Vector Storage (Auto-generated)
├── requirements.txt     # Python Dependencies
└── README.md            # Documentation
```
