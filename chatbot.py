# Need to install following pip install pypdf2 langchain faiss-cpu openai tiktoken streamlit langchain_community
# to run locally : streamlit run C:\Users\harih\PycharmProjects\GenAiExample\chatbot.py

# streamlit - lib for creating ui interfaces..no html, css needed..
# pypdf2 - allows to read pdf src files
# langchain - interface to use open ai service
# faiss - vector store to store embeddings


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

#OPENAI_API_KEY = ""#Pass your key here
#Upload PDF files
st.header("Hari GPT :P ")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)


    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user question
    user_question = st.text_input("Type Your question here")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            #llm generates content. not pulling. as they generate.
            # there can be randomness. its not wrong, just random. cud make answer lengthy.. so we define temperature.
            # lower the value of temp, lesser than randomness. there is no right or wrong,

            max_tokens = 1000,  ## defines the limit of the response... (eg: this cud be 750 words)
            model_name = "gpt-3.5-turbo"
        )

        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)

