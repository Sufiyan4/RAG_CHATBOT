import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="Academic Policy RAG Chatbot")
st.title("📘 Academic Policy Manual Chatbot")

PDF_PATH = "Academic-Policy-Manual-for-Students2.pdf"

openai_key = st.text_input("Enter OpenAI API Key", type="password")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = OpenAI(temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    question = st.text_input("Ask a question from Academic Policy Manual")

    if question:
        answer = qa.run(question)
        st.success(answer)

else:
    st.warning("Please enter OpenAI API Key")
