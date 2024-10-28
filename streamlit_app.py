import streamlit as st
import pathlib
import shutil
import os
import tempfile
from pathlib import Path
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state variables
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .source-box {
        background-color: #e8eaf6;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def setup_google_ai():
    """Configure Google Generative AI with API key"""
    try:
        genai.configure(api_key=st.session_state.api_key)
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=st.session_state.api_key,
            temperature=0.3
        )
    except Exception as e:
        st.error(f"Error setting up Google AI: {str(e)}")
        return None

def process_pdf(uploaded_file):
    """Process the uploaded PDF file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load and split the PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        texts = text_splitter.split_documents(pages)

        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.session_state.api_key
        )
        
        # Use FAISS instead of Chroma
        st.session_state.vector_store = FAISS.from_documents(texts, embeddings)
        
        # Setup QA chain
        llm = setup_google_ai()
        if llm:
            prompt_template = """
            Use the following pieces of context to answer the question at the end. 
            If you cannot find the answer in the context, say "I cannot find the answer in the provided document."
            Please provide a clear and concise answer based only on the information given in the context.
            
            Context: {context}
            Question: {question}
            
            Answer: """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 4}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            # Set processing complete flag
            st.session_state.processing_complete = True
            st.success("PDF processed successfully!")
            
            # Force a rerun to update the UI
            st.rerun()
            
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.session_state.processing_complete = False

def ask_question(question):
    """Process a question and return the answer"""
    if not st.session_state.qa_chain:
        st.error("Please upload a PDF and wait for processing to complete first.")
        return None
    
    try:
        result = st.session_state.qa_chain({"query": question})
        return result
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return None

# Main app layout
st.title("📚 PDF Question Answering System")
st.markdown("---")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key
        
    # Add API key validation
    if st.session_state.api_key:
        try:
            genai.configure(api_key=st.session_state.api_key)
            st.success("API key configured successfully!")
        except Exception as e:
            st.error("Invalid API key. Please check and try again.")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        if not st.session_state.api_key:
            st.warning("Please enter your Google API key first.")
        elif uploaded_file != st.session_state.uploaded_file or not st.session_state.processing_complete:
            st.session_state.uploaded_file = uploaded_file
            with st.spinner("Processing PDF..."):
                process_pdf(uploaded_file)

with col2:
    st.header("Ask Questions")
    user_question = st.text_input("Enter your question about the document")
    
    # Add a check for the qa_chain
    if st.session_state.qa_chain is not None:
        submit_button = st.button("Ask")
    else:
        submit_button = st.button("Ask", disabled=True)

    # Question handling
    if submit_button and user_question and st.session_state.qa_chain:
        with st.spinner("Finding answer..."):
            result = ask_question(user_question)
            if result:
                st.markdown("### Answer")
                st.markdown(f'<div class="answer-box">{result["result"]}</div>', 
                          unsafe_allow_html=True)
                
                st.markdown("### Sources")
                for i, doc in enumerate(result["source_documents"], 1):
                    content = ' '.join(doc.page_content.split())
                    st.markdown(f'<div class="source-box">**Excerpt {i}**<br>{content[:300]}...</div>', 
                              unsafe_allow_html=True)
    elif submit_button and not user_question:
        st.warning("Please enter a question first.")
    elif not st.session_state.qa_chain:
        st.info("Please upload a PDF first to ask questions.")

# Progress indicator
if st.session_state.processing_complete:
    st.sidebar.success("System ready for questions!")
else:
    st.sidebar.info("Upload a PDF to begin")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and LangChain 🚀")
