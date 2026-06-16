import streamlit as st

from modules.rag import retrieve_context
from modules.llm import ask_llm
from modules.text_to_speech import speak

st.set_page_config(
    page_title="Voice AI Meeting Agent",
    layout="wide"
)

st.title("🎙️ Voice AI Meeting Agent")

question = st.text_input(
    "Ask your question"
)

if st.button("Ask"):

    context = retrieve_context(
        question
    )

    answer = ask_llm(
        question,
        context
    )

    st.subheader("Answer")

    st.write(answer)

    speak(answer)