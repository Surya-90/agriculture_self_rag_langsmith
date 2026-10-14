import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# LangSmith tracing is optional. It activates when LANGSMITH_API_KEY is configured.
tracing_enabled = bool(os.getenv("LANGSMITH_API_KEY", "").strip())
if tracing_enabled:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "agriculture-self-rag")
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

from ingest import build_vectorstore
from rag_pipeline import answer_question

st.set_page_config(page_title="Agriculture Schemes RAG", page_icon="🌾", layout="wide")
st.title("🌾 Agriculture Schemes Assistant")
st.caption("Self-RAG-inspired question answering with local FAISS retrieval and optional LangSmith tracing.")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

@st.cache_resource(show_spinner=False)
def load_index(data_dir: str):
    return build_vectorstore(data_dir)

with st.sidebar:
    st.header("Configuration")
    st.write("**LLM:** GPT-4o-mini")
    st.write("**Embeddings:** all-MiniLM-L6-v2 (local)")
    st.write("**Vector store:** FAISS (in memory)")
    st.write("**Retrieval k:** 2")
    st.write("**Chunk size / overlap:** 800 / 200")
    if tracing_enabled:
        st.success("LangSmith tracing is enabled")
        st.caption(f"Project: {os.getenv('LANGSMITH_PROJECT', 'agriculture-self-rag')}")
    else:
        st.info("LangSmith tracing is off. Add LANGSMITH_API_KEY to .env to enable it.")
    if st.button("Rebuild document index", use_container_width=True):
        load_index.clear()
        st.rerun()
    st.divider()
    st.markdown("**Add your documents**")
    st.caption("Place .txt or .pdf agriculture-scheme documents in the `data` folder, then click Rebuild document index.")

api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key or api_key == "paste_your_openai_api_key_here":
    st.error("Add your OpenAI API key to the .env file, then restart the app.")
    st.code("OPENAI_API_KEY=your_actual_openai_api_key")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    with st.spinner("Loading documents and building the local FAISS index (first run can take a few minutes)..."):
        vectorstore, source_documents_count, chunk_count = load_index(DATA_DIR)
except Exception as exc:
    st.error(f"Could not build the document index: {exc}")
    st.info("Check that the `data` folder contains at least one readable .txt or .pdf file.")
    st.stop()

with st.sidebar:
    st.metric("Loaded document records", source_documents_count)
    st.metric("Text chunks", chunk_count)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved sources"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}: {source['label']}**")
                    st.caption(source["excerpt"])

def word_stream(text: str):
    # Streamlit displays the answer a word at a time.
    for word in text.split():
        yield word + " "

prompt = st.chat_input("Ask about an agriculture scheme, eligibility, benefits, or application steps...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching scheme documents and checking the answer..."):
                result = answer_question(prompt, vectorstore, api_key)
            answer = result["answer"]
            st.write_stream(word_stream(answer))
            sources = result.get("sources", [])
            if sources:
                with st.expander("Retrieved sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}: {source['label']}**")
                        st.caption(source["excerpt"])
            if result.get("notes"):
                with st.expander("Self-RAG checks"):
                    for note in result["notes"]:
                        st.write(f"- {note}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.caption("Check your OpenAI key, internet connection, installed dependencies, and the terminal traceback.")
