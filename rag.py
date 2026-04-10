import os
import subprocess

import chromadb
import torch
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from markitdown import MarkItDown


def process_pdf_with_ocr(pdf_path, output_pdf_path):
    """
    Run OCR on a PDF before text extraction.

    OCR is needed because many PDFs are effectively images of text rather than
    digitally selectable text. Without OCR, the retriever would often ingest an
    empty or badly fragmented document.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return None

    print(f"Running OCR on {pdf_path}...")
    try:
        subprocess.run(
            [
                "ocrmypdf",
                "--skip-text",
                "--optimize",
                "1",
                pdf_path,
                output_pdf_path,
            ],
            check=True,
        )
        print(f"OCR complete. Saved to {output_pdf_path}")
        return output_pdf_path
    except subprocess.CalledProcessError as exc:
        print(f"OCR failed: {exc}")
        return None
    except FileNotFoundError:
        print("Error: ocrmypdf is not installed or not available on PATH.")
        return None


def extract_text_with_markitdown(pdf_path):
    """
    Convert a processed document into text that is easy to chunk and embed.

    MarkItDown is useful here because it tries to preserve more document
    structure than a naive plain-text extractor.
    """
    print(f"Extracting markdown from {pdf_path} using MarkItDown...")
    markdown_converter = MarkItDown()
    try:
        result = markdown_converter.convert(pdf_path)
        print("Markdown extraction complete.")
        return result.text_content
    except Exception as exc:
        print(f"MarkItDown extraction failed: {exc}")
        return ""


def build_rag_pipeline(documents_text, collection_name="minimo_rag", use_minimo_llm=False):
    """
    Build a local retrieval pipeline with ChromaDB and LlamaIndex.

    The embedding model is kept separate from the text generator. That modular
    split is standard in RAG systems: the retriever specializes in semantic
    search, while the generator specializes in turning retrieved facts into
    fluent answers.
    """
    print("Initializing embedding model: sentence-transformers/all-MiniLM-L6-v2")

    # `all-MiniLM-L6-v2` is a common lightweight embedding model for local work.
    # It is small enough to run comfortably on consumer hardware while still
    # producing useful semantic search vectors.
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    if use_minimo_llm:
        print("Loading the local Minimo model into LlamaIndex...")
        llm = HuggingFaceLLM(
            model_name="checkpoints/hf_minimo_base",
            tokenizer_name="checkpoints/hf_minimo_base",
            # The larger context window here is an inference-side setting for
            # the wrapper, not proof that the model was fully trained at 2048
            # tokens. It leaves room for experimentation with retrieved context.
            context_window=2048,
            max_new_tokens=150,
            generate_kwargs={"temperature": 0.7, "do_sample": True},
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        Settings.llm = llm
    else:
        Settings.llm = None

    print("Setting up the ChromaDB vector store...")
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = [Document(text=documents_text)]

    print("Generating embeddings and building the index...")
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    print("RAG pipeline initialization complete.")
    return index


def query_rag(index, query_text):
    """
    Run a query through the RAG index and print the answer.
    """
    print(f"Querying RAG system: {query_text!r}")
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    print(f"Response: {response}")
    return response


if __name__ == "__main__":
    sample_pdf = "sample.pdf"
    output_pdf = "sample_ocr.pdf"

    if not os.path.exists(sample_pdf):
        print(f"Please provide a '{sample_pdf}' file to test the PDF pipeline.")
        print("Using fallback text instead.")
        text_content = (
            "Minimo is an engineered ~217M parameter autoregressive causal language "
            "model optimized for an RTX 5060 8GB GPU. It features 18 layers, a "
            "hidden size of 896, grouped-query attention with 14 query heads and "
            "2 key/value heads, and RMSNorm."
        )
    else:
        processed_pdf = process_pdf_with_ocr(sample_pdf, output_pdf)
        if processed_pdf:
            text_content = extract_text_with_markitdown(processed_pdf)
        else:
            text_content = "Fallback text content."

    if text_content:
        index = build_rag_pipeline(text_content, use_minimo_llm=False)
        query_rag(index, "What is Minimo?")
