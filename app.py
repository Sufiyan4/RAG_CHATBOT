import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain

st.title("📄 PDF RAG Chatbot with Groq LLM")

# LLM Initialization
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama3-8b-8192"
)

# Embeddings Initialization
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # lightweight, cloud-friendly
)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_chunks = text_splitter.split_documents(pages)
    
    # Convert text to vectors & store in FAISS
    vectorstore = FAISS.from_documents(final_chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    st.success("PDF Loaded & Vectorstore Ready!")
    
    # Chat input
    query = st.text_input("Ask something about your PDF:")
    
    if query:
        chain = create_retrieval_chain(llm=llm, retriever=retriever)
        response = chain.run(query)
        st.write(response)
