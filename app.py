import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="Academic Policy RAG Chatbot")
st.title("📘 Academic Policy Manual Chatbot")

PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf"

# OpenAI API Key
openai_key = st.text_input("Enter OpenAI API Key", type="password")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Embeddings (FREE)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # LLM (OpenAI)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    # RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    question = st.text_input("Ask a question from Academic Policy Manual")

    if question:
        answer = qa.run(question)
        st.success(answer)

else:
    st.warning("Please enter OpenAI API Key to continue")
