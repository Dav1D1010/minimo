# Minimo

Minimo is a small local language-model project built to be understandable, hackable, and realistic on consumer hardware. The repository covers the whole loop: train a tokenizer, pretrain a compact decoder-only transformer, adapt it with LoRA, align it with DPO, merge the final weights, and run chat, RAG, or a simple vision-language bridge.

The current setup is shaped around an RTX 5060 with 8 GB of VRAM, so a lot of the design choices trade raw scale for practicality. That is the point of the project. The code tries to stay close enough to modern LLM ideas to be educational, while still being small enough to read in one sitting and run locally without a data center.

## What is in this repo

- `main.py` is the entrypoint. It can launch training, tokenizer building, RAG, VLM initialization, or interactive chat.
- `model.py` defines the Minimo transformer in Hugging Face style so checkpoints can be saved and loaded with `save_pretrained()` and `from_pretrained()`.
- `train.py` runs the three-stage training pipeline: pretraining, supervised fine-tuning, and DPO alignment.
- `tokenizer_builder.py` builds the byte-level BPE tokenizer from local documents and optionally a Hugging Face dataset.
- `chat.py` runs an interactive terminal chat interface that can also read local documents and images.
- `rag.py` sets up a small local retrieval pipeline with ChromaDB and LlamaIndex.
- `vlm.py` connects the text model to a SigLIP vision encoder through a learned projector.
- `export_merged.py` merges the final LoRA adapter into the base model for easier inference.
- `test_speed.py` gives a quick local generation-speed benchmark.

## Model shape

The text model is intentionally small by modern standards, but still large enough to demonstrate the main transformer ideas:

- About 217 million parameters
- 18 transformer layers
- Hidden size of 896
- 14 attention query heads
- 2 key/value heads for grouped-query attention
- RMSNorm
- SwiGLU feed-forward blocks
- Rotary position embeddings

Some of those numbers look unusual at first, so it helps to know why they exist:

- `hidden_size=896` divides cleanly across 14 heads, which gives 64 dimensions per head.
- `num_key_value_heads=2` keeps memory use lower than full multi-head attention during generation.
- `intermediate_size=3584` is 4 times the hidden size, which is a common transformer feed-forward ratio.
- `max_position_embeddings=2048` gives room for longer-context experiments, even though training uses shorter sequences for cost reasons.

## Training pipeline

The project uses three stages.

### 1. Tokenizer training

The tokenizer is a byte-level BPE tokenizer with a default vocabulary size of `6400`.

That vocabulary is smaller than what larger production models usually use, but it is a reasonable tradeoff for a compact local model:

- a smaller vocabulary keeps the embedding table and language-model head lighter
- byte-level tokenization stays robust to punctuation, odd Unicode text, and domain-specific strings
- BPE still learns useful subword chunks instead of forcing everything into raw bytes

Run it with:

```bash
python main.py --mode tokenize --docs_dir /path/to/documents
```

Or train it from a single text file:

```bash
python main.py --mode tokenize --data my_corpus.txt
```

If no mode is supplied at all, `main.py` now opens a small interactive mode picker in the terminal.

### 2. Pretraining

Pretraining uses `roneneldan/TinyStories`.

That dataset is a practical choice for a project like this because it is clean, large, and simple enough for a smaller model to learn meaningful language patterns without needing an enormous training budget. The default training configuration in `train.py` uses:

- `BATCH_SIZE=1`
- `GRAD_ACCUM_STEPS=16`
- `PRETRAIN_STEPS=100000`
- `MAX_SEQ_LEN=256`
- `LEARNING_RATE=5e-4`

Those numbers are chosen for local feasibility more than theoretical perfection. Sequence length is kept short because attention cost grows quickly, and gradient accumulation is used to simulate a larger batch without needing the VRAM for one.

Run it with:

```bash
python main.py --mode train
```

If a Hugging Face-format base checkpoint already exists in `hf_minimo`, the entrypoint skips the pretraining stage and continues from there.

### 3. Supervised fine-tuning

SFT uses `ise-uiuc/Magicoder-OSS-Instruct-75K` and LoRA adapters.

LoRA is a good fit here because it avoids updating the full model. Instead, it learns small low-rank matrices inside the attention projections. That makes adaptation much cheaper in memory and storage, which matters a lot on an 8 GB GPU.

The default LoRA settings are:

- `r=8`
- `lora_alpha=32`
- `lora_dropout=0.05`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`

### 4. DPO alignment

The final stage uses `argilla/dpo-mix-7k`.

DPO is useful because it pushes the model toward preferred answers without needing a separate reward model. The code compares the current policy against a reference version of the model and nudges it toward the chosen response in each preference pair.

The DPO stage uses a lower learning rate than pretraining and SFT, because alignment is meant to gently shape behavior rather than rewrite the model’s core language knowledge.

## Running the project

### Interactive mode picker

```bash
python main.py
```

If `--mode` is missing, the script asks which mode to run.

### Chat

```bash
python main.py --mode chat
```

The chat interface can:

- answer normal text prompts
- read `.txt` and `.md` files
- OCR and search `.pdf` files
- accept one image path per turn and route the request through the VLM path

Example:

```text
Summarize this file /home/david/notes.txt
```

### RAG demo

```bash
python main.py --mode rag
```

This runs a small local retrieval example with:

- ChromaDB for vector storage
- LlamaIndex for indexing and retrieval
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings

### Vision-language model

```bash
python main.py --mode vlm
```

The VLM path uses:

- the local Minimo text model for generation
- `google/siglip-base-patch16-224` as the image encoder
- a small learned MLP projector to map image features into the text-model hidden space

This is a practical multimodal architecture for a local project because it reuses a strong pretrained vision backbone instead of trying to train a full multimodal stack from scratch.

### Merge the final model

```bash
python export_merged.py
```

This folds the DPO LoRA adapter into the base model and writes a standalone inference checkpoint to `hf_minimo_merged/`.

### Quick speed test

```bash
python test_speed.py
```

This is only a rough benchmark, but it is useful for checking whether generation speed is in the expected range after exporting the merged model.

## Dependencies

The project expects Python 3.13+ and uses libraries such as:

- PyTorch
- Transformers
- PEFT
- Datasets
- Tokenizers
- ChromaDB
- LlamaIndex
- MarkItDown
- OCRmyPDF

Dependencies are listed in [`pyproject.toml`](/home/david/Projects/minimo/pyproject.toml).

## A quick note on expectations

Minimo is a learning project, not a claim that a 217M local model will outperform modern production LLMs. The value here is in seeing how the pieces fit together:

- how tokenizer size affects model footprint
- why grouped-query attention matters for local inference
- why LoRA is so useful for limited hardware
- how retrieval can supplement a small model
- how a vision encoder can be attached to a text model with a projector

That makes the repository a good sandbox for experimentation, debugging, and understanding the mechanics behind larger systems.
