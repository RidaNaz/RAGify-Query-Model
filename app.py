import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="RAG Query Model",
    page_icon="favicon.ico",
    layout="centered",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .custom-title {
        padding-bottom: 10px;
        font-size: 2.5em;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .custom-subtitle {
        font-size: 1.2em;
        color: #5D6D7E;
        text-align: center;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1em;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the title
st.markdown("<h1 class='custom-title'>RAGify Query Model</h1>", unsafe_allow_html=True)

# Sidebar for file upload and model selection
with st.sidebar:
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
      st.image("favicon.ico", width=25)

    with col2:
      st.markdown("<h2 style='margin-top: -15px;'>Rida Naz</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
    model_name = st.selectbox(
        "Select Model",
        options=["llama-3.3-70b-versatile", "llama-3.3-70b-specdec", "llama3-70b-8192"],
        index=0,
        help="Select the model for generating embeddings and answering queries.",
    )
    if uploaded_files and st.button("Generate Embeddings"):
        with st.spinner("Generating embeddings..."):
            # Clear existing embeddings and documents
            if "vectors" in st.session_state:
                del st.session_state.vectors
            if "docs" in st.session_state:
                del st.session_state.docs

            # Save uploaded files to a temporary directory
            temp_dir = "./temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Load and process documents
            st.session_state.loader = PyPDFDirectoryLoader(temp_dir)
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

            # Clean up temporary directory
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

            st.success("Embeddings generated successfully!")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

# Dynamic system prompt
def generate_dynamic_prompt(doc_titles):
    return f"""
    You are an AI assistant specializing in the content provided within the following documents: {', '.join(doc_titles)}.
    Answer user queries strictly based on the context from these documents. Provide clear, concise, and accurate responses.
    """

# User query input
user_query = st.text_input("What would you like to know?", placeholder="Enter your query here...")

# Process user query
if user_query:
    if "vectors" not in st.session_state:
        st.warning("Please generate embeddings first.")
    else:
        with st.spinner("Processing your query..."):
            # Extract document titles
            doc_titles = [doc.metadata['source'].split('/')[-1] for doc in st.session_state.docs]
            system_prompt = generate_dynamic_prompt(doc_titles)

            # Create prompt template
            prompt = ChatPromptTemplate.from_template(f"""
            {system_prompt}
            <context>
            {{context}}
            <context>
            Questions:{{input}}
            """)

            # Create document chain and retrieval chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Invoke retrieval chain
            response = retrieval_chain.invoke({'input': user_query})
            st.write(response['answer'])

# Reset session state
if st.button("Reset App"):
    st.session_state.clear()
    st.rerun()