import streamlit as st
import os
from openai import OpenAI
import tiktoken
import time

st.set_page_config(page_title="Roberta AI", page_icon=":robot_face:")

st.markdown("# 🤖 Roberta AI")
st.markdown("**by Andrea Barilà per IDRA srl**")

with st.sidebar:
    api_key = st.secrets["OPENAI_API_KEY"]
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        st.stop()

system_prompt = """
Il file .srt contiene una serie di snippet numerati, ciascuno con timestamp e testo.

Il tuo compito è individuare **gli snippet sospetti** che potrebbero contenere errori ortografici, grammaticali, sintattici o incongruenze logiche, dovuti a trascrizioni imperfette da audio di scarsa qualità.

Per ogni snippet sospetto:
- restituisci **numero**, **timestamp**, **testo originale**, **motivazione** (massimo 200 caratteri).

**Anche solo il sospetto è sufficiente** per includerlo: meglio un falso positivo che un errore ignorato.

### Formato:
Rispondi **solo** con una tabella in Markdown come questa:

| Numero | Timestamp | Testo | Motivazione |
|--------|-----------|-------|-------------|
| 437 | 00:27:33,718 --> 00:27:38,490 | in questa lezione terciaa del secondo modulo | “terciaa” sembra errore di battitura, dovrebbe essere “terza” |
"""

if "chat_records" not in st.session_state:
    st.session_state.chat_records = []
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = []
if "active_chat_index" not in st.session_state:
    st.session_state.active_chat_index = None
if "delete_index" not in st.session_state:
    st.session_state.delete_index = None

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

def split_srt_into_chunks(srt_text, max_tokens):
    blocks = [b for b in srt_text.strip().split("\n\n") if len(b.strip().splitlines()) >= 3 and len(b.strip()) > 10]
    chunks, current_chunk, current_token_count = [], [], 0
    for block in blocks:
        token_count = count_tokens(block)
        if current_token_count + token_count > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [block]
            current_token_count = token_count
        else:
            current_chunk.append(block)
            current_token_count += token_count
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

@st.cache_data(show_spinner=False)
def generate_response_from_chunk(chunk, retries=1):
    if not chunk.strip():
        return ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0.2,
            top_p=1.0,
            max_tokens=st.session_state.get("max_tokens", 1500)

        )
        return response.choices[0].message.content
    except Exception as e:
        if "token" in str(e).lower() and retries > 0:
            mid = len(chunk) // 2
            return generate_response_from_chunk(chunk[:mid], retries=retries-1)
        return f"⚠️ Errore durante l'elaborazione: {str(e)}"

with st.sidebar:
    st.title("📜 Storico Analisi File")
    if st.button("➕ Nuova analisi"):
        st.session_state.active_chat_index = None
        st.rerun()
    for i, title in enumerate(st.session_state.chat_titles):
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(f"📄 {title}", key=f"select_{i}"):
                st.session_state.active_chat_index = i
                st.rerun()
        with col2:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.delete_index = i
    if st.session_state.delete_index is not None:
        del st.session_state.chat_titles[st.session_state.delete_index]
        del st.session_state.chat_records[st.session_state.delete_index]
        st.session_state.delete_index = None
        st.session_state.active_chat_index = None
        st.rerun()
    st.session_state["max_tokens"] = st.slider('Max Tokens per blocco', 256, 4096, 1500, 128)


uploaded_file = st.file_uploader("📁 Carica file .srt da analizzare (uno per volta)", type=["srt"])

if (
    st.session_state.active_chat_index is not None
    and 0 <= st.session_state.active_chat_index < len(st.session_state.chat_titles)
):
    st.subheader(f"🧠 Risultato per: {st.session_state.chat_titles[st.session_state.active_chat_index]}")
    st.markdown("\n\n" + st.session_state.chat_records[st.session_state.active_chat_index], unsafe_allow_html=True)
elif uploaded_file is not None:
    file_name = uploaded_file.name
    file_content = uploaded_file.read().decode('utf-8')
    with st.spinner("Analisi in corso..."):
        chunks = split_srt_into_chunks(file_content, st.session_state["max_tokens"])
        all_responses = []
        progress = st.progress(0)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                resp = generate_response_from_chunk(chunk)
                if resp.strip():
                    all_responses.append(resp)
            progress.progress((i + 1) / len(chunks))
            time.sleep(1.5)
    final_output = "\n\n" + "\n\n".join(all_responses)
    st.subheader(f"🧠 Risultato per: {file_name}")
    st.markdown(final_output, unsafe_allow_html=True)
    st.session_state.chat_records.append(final_output)
    st.session_state.chat_titles.append(file_name)
    st.session_state.active_chat_index = len(st.session_state.chat_titles) - 1
    st.download_button(
        label="💾 Scarica correzioni in .txt",
        data=final_output,
        file_name=f"correzione_{file_name}.txt",
        mime="text/plain"
    )
