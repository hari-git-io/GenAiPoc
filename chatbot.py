# Need to install following pip install pypdf2 langchain faiss-cpu openai tiktoken streamlit
# to run locally : streamlit run C:\Users\harih\PycharmProjects\GenAiExample\chatbot.py


import streamlit as st

st.header("My First Chatbot")

with st.sidebar:
        st.title("My First Chatbot")
        file = st.file_uploader("Upload a file", type = "pdf")
