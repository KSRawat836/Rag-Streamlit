import streamlit as st

from core.ingest import load_file, chunk_text, embed_chunks, upsert_to_vectordb
from core.retrieve import retrieve
from core.generate import generate_answer

st.set_page_config(page_title="Ask your document", layout="centered")
st.title("Ask your document")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    key = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.get("processed_key") != key:
        with st.spinner("Indexing..."):
            raw_text = load_file(uploaded_file)
            chunks = chunk_text(raw_text)
            embeddings = embed_chunks(chunks)
            upsert_to_vectordb(embeddings.tolist(), chunks)
        st.session_state.processed_key = key
        st.session_state.messages = []
        st.success(f"Indexed {uploaded_file.name}")

st.session_state.setdefault("messages", [])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

query = st.chat_input(
    "Ask a question about your document",
    disabled=st.session_state.get("processed_key") is None,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(retrieve(query), query)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})