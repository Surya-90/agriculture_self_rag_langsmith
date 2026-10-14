# Agriculture Schemes Self-RAG (LangSmith traced)

A beginner-friendly Streamlit application for asking questions about agriculture scheme documents.

## Features

- OpenAI `gpt-4o-mini` for answers and Self-RAG-style checks
- Local Hugging Face embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector index stored in memory only
- PDF and TXT document loading
- `RecursiveCharacterTextSplitter` with chunk size 800 and overlap 200
- Retrieves `k=2` chunks at a time
- Relevance grading, one query rewrite/retrieval retry, and answer grounding grading
- Word-by-word answer display
- LangSmith tracing for pipeline steps and wrapped OpenAI API calls (optional)

> This is a **Self-RAG-inspired** application implemented with evaluator prompts. It is not the original research model trained specifically for Self-RAG.

## 1. Install prerequisites

Install Python 3.10 or 3.11 from https://www.python.org/downloads/ and make sure Python is available in your terminal. VS Code should have the Python extension installed.

## 2. Open the project in VS Code

Extract the ZIP file, then in VS Code choose **File → Open Folder** and select `agriculture_self_rag_langsmith`.

Open the VS Code terminal (**Terminal → New Terminal**) and run these commands on Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first installation downloads Python packages and the embedding model; it may take a few minutes.

## 3. Add your OpenAI key

Open the `.env` file in VS Code and replace:

```text
OPENAI_API_KEY=paste_your_openai_api_key_here
```

with your actual key. Do not share the key or commit the `.env` file.

## 4. Enable LangSmith tracing (optional but recommended)

1. Sign in at https://smith.langchain.com/
2. Create/copy a LangSmith API key.
3. In `.env`, set:

```text
LANGSMITH_API_KEY=your_actual_langsmith_api_key
LANGSMITH_PROJECT=agriculture-self-rag
```

If `LANGSMITH_API_KEY` is blank, the app runs without sending traces to LangSmith. The OpenAI key and LangSmith key are separate keys.

## 5. Add agriculture scheme documents

Place PDF or UTF-8 TXT files into the `data` folder. A small sample text file is included so you can test the app immediately. Replace/add official scheme documents for your intended use. For important eligibility or deadline decisions, verify against the official government source.

## 6. Run the app

In the VS Code terminal:

```bat
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## 7. Refresh the index after adding documents

Use **Rebuild document index** in the left sidebar after adding, removing, or changing files in `data`. The FAISS index is held in memory; it is not saved to disk. When the app process ends, the index is discarded and rebuilt on the next run.

## Privacy and tracing notes

- Embedding generation happens on your computer after the model package is downloaded.
- Your source documents remain in the local `data` folder.
- Retrieved text is sent to OpenAI as context when you ask a question, because the hosted GPT-4o-mini API generates the answer.
- If LangSmith tracing is enabled, trace inputs/outputs and relevant run metadata may be sent to your LangSmith workspace. Depending on the traced call, traces can include the user question, retrieved document text, and model response. Avoid indexing confidential data or disable tracing by blanking `LANGSMITH_API_KEY` if you do not want traces sent to LangSmith.
- Keep both API keys private.

## Troubleshooting

**`OPENAI_API_KEY` missing:** edit `.env`, save it, and restart Streamlit.

**No documents found:** add at least one `.pdf` or `.txt` file to `data`.

**FAISS or package install error:** confirm the virtual environment is activated and Python version is 3.10 or 3.11, then retry `pip install -r requirements.txt`.

**LangSmith traces do not appear:** confirm `LANGSMITH_API_KEY` is correct, `LANGSMITH_PROJECT` is set, the computer has internet access, and you restarted the app after editing `.env`.

**Want to turn tracing off:** remove the LangSmith key from `.env`, save, and restart the app.
