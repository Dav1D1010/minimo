import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
import chromadb
from markitdown import MarkItDown
import subprocess

def process_pdf_with_ocr(pdf_path, output_pdf_path):
    """
    Automates local document ingestion by processing PDFs with OCRMyPDF.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return None
    
    print(f"Running OCR on {pdf_path}...")
    try:
        # OCRMyPDF is a CLI tool, we run it via subprocess
        # Using fast optimization and skipping text if already present
        subprocess.run(["ocrmypdf", "--skip-text", "--optimize", "1", pdf_path, output_pdf_path], check=True)
        print(f"OCR complete. Saved to {output_pdf_path}")
        return output_pdf_path
    except subprocess.CalledProcessError as e:
        print(f"OCR failed: {e}")
        return None
    except FileNotFoundError:
        print("Error: ocrmypdf not installed or not in PATH.")
        return None

def extract_text_with_markitdown(pdf_path):
    """
    Extracts structured markdown text from the OCR'd PDF using MarkItDown.
    """
    print(f"Extracting markdown from {pdf_path} using MarkItDown...")
    md = MarkItDown()
    try:
        result = md.convert(pdf_path)
        print("Markdown extraction complete.")
        return result.text_content
    except Exception as e:
        print(f"MarkItDown extraction failed: {e}")
        return ""

def build_rag_pipeline(documents_text, collection_name="minimo_rag"):
    """
    Builds a local RAG pipeline using LlamaIndex and ChromaDB.
    Generates embeddings using all-MiniLM-L6-v2.
    """
    print("Initializing embedding model: all-MiniLM-L6-v2...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    
    # In a full setup, we would wrap our custom MinimoModel in a LlamaIndex LLM class
    # Settings.llm = MinimoLLM() 
    # For now, we will just use the default (None or OpenAI) or disable it for embedding-only indexing
    Settings.llm = None 
    
    print("Setting up ChromaDB vector store...")
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create LlamaIndex Document
    docs = [Document(text=documents_text)]
    
    print("Generating embeddings and building index...")
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    print("RAG Pipeline initialization complete. Index stored in ChromaDB.")
    
    return index

def query_rag(index, query_text):
    """
    Query the RAG pipeline.
    """
    print(f"Querying RAG system: '{query_text}'")
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    print(f"Response: {response}")
    return response

if __name__ == "__main__":
    # Example Pipeline
    sample_pdf = "sample.pdf"
    output_pdf = "sample_ocr.pdf"
    
    # 1. Provide a dummy PDF if none exists for demonstration
    if not os.path.exists(sample_pdf):
        print(f"Please provide a '{sample_pdf}' to test the pipeline.")
        print("Using dummy text instead of full PDF ingestion.")
        text_content = "Minimo is an engineered ~105.6M parameter autoregressive Causal Language Model optimized for an RTX 5060 8GB GPU. It features 16 layers, a hidden dimension of 768, Grouped-Query Attention (GQA) with 12 Q-heads and 4 KV-heads, and RMSNorm."
    else:
        # 2. Process with OCRMyPDF
        processed_pdf = process_pdf_with_ocr(sample_pdf, output_pdf)
        
        # 3. Extract text with MarkItDown
        if processed_pdf:
            text_content = extract_text_with_markitdown(processed_pdf)
        else:
            text_content = "Fallback text content."
    
    # 4. Build RAG Pipeline and embed with all-MiniLM-L6-v2
    if text_content:
        index = build_rag_pipeline(text_content)
        query_rag(index, "What is Minimo?")
