import argparse
from model import MinimoModel
from vlm import MinimoVLM
from rag import build_rag_pipeline, query_rag
from tokenizer_builder import train_tokenizer
import os

def main():
    parser = argparse.ArgumentParser(description="Minimo Local LLM Pipeline")
    parser.add_argument("--mode", choices=["train", "vlm", "rag", "tokenize"], required=True, 
                        help="Select the pipeline mode to run.")
    args = parser.parse_args()

    print(f"=== Starting Minimo in {args.mode.upper()} mode ===")

    if args.mode == "tokenize":
        print("Training custom BPE tokenizer...")
        dummy_file = "dummy_data.txt"
        with open(dummy_file, "w") as f:
            f.write("A small test dataset for Minimo's BPE Tokenizer. " * 100)
        train_tokenizer(dummy_file)
        os.remove(dummy_file)

    elif args.mode == "train":
        import train
        print("Running full training pipeline...")
        base_model = train.pretrain_model()
        sft_model = train.fine_tune_sft(base_model)
        aligned_model = train.align_dpo(sft_model)

    elif args.mode == "vlm":
        print("Initializing Minimo Vision-Language Model...")
        # Base LLM with dummy 6400 vocab size
        llm = MinimoModel(vocab_size=6400)
        vlm = MinimoVLM(llm_model=llm, vision_model_name="google/siglip-base-patch16-224")
        print("VLM Initialized. Ready to process image and text inputs.")

    elif args.mode == "rag":
        print("Setting up local RAG deployment...")
        # Since vLLM has CUDA compilation issues in this environment,
        # we demonstrate the LlamaIndex retrieval setup locally.
        print("Deploying RAG pipeline (simulated via LlamaIndex + ChromaDB)...")
        text_content = "Minimo is an engineered ~105.6M parameter autoregressive Causal Language Model optimized for an RTX 5060 8GB GPU. It features 16 layers, a hidden dimension of 768, Grouped-Query Attention (GQA) with 12 Q-heads and 4 KV-heads, and RMSNorm."
        print(f"Ingesting default text: {text_content}")
        index = build_rag_pipeline(text_content)
        
        print("\n--- Testing RAG Pipeline ---")
        query_rag(index, "What features does Minimo have?")

if __name__ == "__main__":
    main()
