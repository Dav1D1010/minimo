# Minimo: Minimal Multi-Model for Learning

Minimo is a custom, from-scratch Causal Language Model (LLM) pipeline built for local experimentation, education, and hardware-constrained environments. It is specifically engineered to maximize the capabilities of an **NVIDIA RTX 5060 (8GB VRAM)**.

## 🚀 Architecture
*   **Parameters:** ~217.5 Million
*   **Layers:** 18 (Deeper for better reasoning)
*   **Hidden Dimension:** 896
*   **Attention Heads:** 14 Query Heads
*   **Grouped-Query Attention (GQA):** 2 KV Heads (Extreme 7:1 ratio to conserve intermediate memory states)
*   **Max Sequence Length:** 2048 (256 used during Pretraining/SFT)
*   **Hardware Footprint:** ~6.8GB VRAM utilized

## 🛠️ Pipeline Features
The Minimo repository contains an end-to-end local AI pipeline controlled via `main.py`:

### 1. Tokenizer Training (`--mode tokenize`)
Trains a custom Byte-Level BPE tokenizer optimized for your domain.
*   Supports extracting text from local datasets (PDFs, Markdown, Word docs, TXT) and merging it with Hugging Face datasets (e.g., `TinyStories`).
*   **Usage:** `python main.py --mode tokenize --docs_dir "/path/to/your/documents"`

### 2. Full Training Pipeline (`--mode train`)
Runs the complete 3-stage training process sequentially, caching datasets to an external `../Data` folder to save project space.
*   **Stage 1: Pretraining (Base Model)**
    *   Trains on `roneneldan/TinyStories`.
    *   **Scale:** 100,000 steps (reading nearly the entire 2.11M example dataset).
    *   **Time:** ~20 hours on an RTX 5060.
*   **Stage 2: Supervised Fine-Tuning (SFT)**
    *   Trains on `ise-uiuc/Magicoder-OSS-Instruct-75K` using LoRA adapters.
    *   **Scale:** 1 Epoch (~4,700 steps) to prevent overfitting.
*   **Stage 3: Direct Preference Optimization (DPO)**
    *   Aligns model to human preferences using `argilla/dpo-mix-7k`.
    *   **Scale:** 1 Epoch (~422 steps) for final polish.

**Usage:** `python main.py --mode train`

### 3. Vision-Language Model (`--mode vlm`)
Connects the trained base LLM with Google's SigLIP (`google/siglip-base-patch16-224`) to process image and text inputs natively.
*   **Usage:** `python main.py --mode vlm`

### 4. Local RAG Pipeline (`--mode rag`)
Demonstrates Retrieval-Augmented Generation using LlamaIndex, ChromaDB, and `all-MiniLM-L6-v2` embeddings, allowing the model to answer questions based on ingested local documents.
*   **Usage:** `python main.py --mode rag`
