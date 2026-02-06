# app.py
import streamlit as st
from io import BytesIO
import re

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# -------------------------------
# Helper functions
# -------------------------------

def clean_text(text):
    """Clean text for better readability."""
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove simple names (Firstname Lastname)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '', text)
    # Collapse multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Strip leading/trailing spaces
    return text.strip()

import tempfile

def load_pdfs(uploaded_files):
    """Load PDF files from Streamlit upload (works with PyPDFLoader)."""
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents += loader.load()
    return documents


def split_documents(documents, chunk_size=400, chunk_overlap=50):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    """Create FAISS vector store from chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_llm():
    """Create HuggingFace LLM pipeline."""
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_answer(vector_store, llm, question, top_k=2):
    """Retrieve relevant documents and generate answer."""
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    
    context = "\n\n".join([clean_text(d.page_content) for d in docs])
    context = context[:1000]  # truncate if too long

    final_prompt = f"""
You are a research assistant. Answer the question **clearly and concisely** using **only the context below**. 
Do not add information not present in the context. 
If the answer is not in the context, say "I don't know".
Context:
{context}

Question:
{question}
"""
    return llm.invoke(final_prompt)

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="RAG PDF Research Assistant", layout="wide")
st.title("RAG PDF Research Assistant")

# PDF Upload
uploaded_files = st.file_uploader(
    "Upload PDFs (multiple allowed)", 
    type="pdf", 
    accept_multiple_files=True
)

# Question input
question = st.text_input("Ask your question:")

if uploaded_files:
    with st.spinner("Loading PDFs..."):
        documents = load_pdfs(uploaded_files)
    st.success(f"{len(documents)} document pages loaded!")

    # Split and create vector store
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    llm = create_llm()
    st.success("Vector store and LLM loaded!")

    if question:
        with st.spinner("Generating answer..."):
            answer = get_answer(vector_store, llm, question)
        st.subheader("Answer:")
        st.write(answer)
