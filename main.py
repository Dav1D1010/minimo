import argparse
import os

from rag import build_rag_pipeline, query_rag
from tokenizer_builder import prepare_corpus, train_tokenizer
from vlm import MinimoVLM


AVAILABLE_MODES = ("train", "vlm", "rag", "tokenize", "chat")


def prompt_for_mode():
    """
    Ask for a mode only when the command line did not provide one.

    Keeping this prompt separate from the rest of `main()` makes the startup
    flow easier to read and keeps the non-interactive command line path exactly
    the same as before.
    """
    print("No mode was provided on the command line.")
    print("Select a mode to run:")
    for index, mode in enumerate(AVAILABLE_MODES, start=1):
        print(f"  {index}. {mode}")

    while True:
        choice = input("Enter a number or mode name: ").strip().lower()
        if choice in AVAILABLE_MODES:
            return choice
        if choice.isdigit():
            numeric_choice = int(choice)
            if 1 <= numeric_choice <= len(AVAILABLE_MODES):
                return AVAILABLE_MODES[numeric_choice - 1]
        print("Invalid selection. Choose one of the listed modes.")


def build_parser():
    """
    Build the command line parser in one place.

    A tiny helper like this keeps `main()` focused on orchestration instead of
    argument setup, which makes the top-level program flow easier to follow.
    """
    parser = argparse.ArgumentParser(description="Minimo local model pipeline")
    parser.add_argument(
        "--mode",
        choices=AVAILABLE_MODES,
        default=None,
        help="Pipeline mode to run. When omitted, an interactive selector appears.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a text file used when training the tokenizer from a custom corpus.",
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=None,
        help=(
            "Path to a directory of local documents to mix into the tokenizer corpus "
            "alongside TinyStories."
        ),
    )
    return parser


def run_tokenizer_mode(args):
    """
    Train the tokenizer from either a prepared dataset, a document folder, or a
    tiny fallback file.

    The fallback branch exists only so the tokenizer code can still be exercised
    on a completely fresh checkout. It is not a good training corpus, but it is
    enough to verify that the pipeline is wired correctly.
    """
    print("Training custom BPE tokenizer...")

    if args.docs_dir:
        print(f"Building a mixed corpus from {args.docs_dir} and TinyStories...")
        corpus_file = prepare_corpus(docs_dir=args.docs_dir)
        train_tokenizer(corpus_file)
        return

    if args.data:
        print(f"Using the provided text dataset: {args.data}")
        train_tokenizer(args.data)
        return

    print("No corpus path was provided. Using a tiny temporary fallback dataset...")
    dummy_file = "dummy_data.txt"
    with open(dummy_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("A small test dataset for Minimo's tokenizer. " * 100)
    try:
        train_tokenizer(dummy_file)
    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)


def run_training_mode():
    """
    Run the three-stage text-model pipeline.

    The training script already contains the stage-specific logic, so this
    branch keeps the entrypoint intentionally thin and acts more like a launcher
    than a second copy of the training implementation.
    """
    import torch

    import train
    from model import MinimoForCausalLM

    print("Running the Hugging Face training pipeline...")

    if os.path.exists("hf_minimo"):
        print("Found an existing base model in 'hf_minimo'. Skipping pretraining.")
        base_model = MinimoForCausalLM.from_pretrained("hf_minimo")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model.to(device)
    else:
        base_model = train.pretrain_model()

    sft_model = train.fine_tune_sft(base_model)
    train.align_dpo(sft_model)


def run_vlm_mode():
    """
    Initialize the vision-language stack.

    This mode currently focuses on bringing the model components into memory and
    verifying that the text model and vision encoder can be stitched together.
    """
    print("Initializing Minimo vision-language model...")
    MinimoVLM(
        hf_model_path="hf_minimo",
        vision_model_name="google/siglip-base-patch16-224",
    )
    print("VLM initialized and ready for image-text experiments.")


def run_rag_mode():
    """
    Build a small local retrieval demo.

    The example text is intentionally short and self-contained so the retrieval
    flow can be demonstrated without requiring any external files.
    """
    print("Setting up the local RAG demo...")
    text_content = (
        "Minimo is an engineered ~217M parameter autoregressive causal language "
        "model optimized for an RTX 5060 8GB GPU. It uses 18 layers, a hidden "
        "size of 896, grouped-query attention with 14 query heads and 2 key/value "
        "heads, and RMSNorm."
    )
    print(f"Ingesting demo text:\n{text_content}")
    index = build_rag_pipeline(text_content, use_minimo_llm=False)
    print("\n--- Testing the retrieval pipeline ---")
    query_rag(index, "What features does Minimo have?")


def run_chat_mode():
    """
    Hand off to the dedicated chat module.

    Keeping chat in its own module helps the entrypoint stay small while still
    allowing interactive behavior to grow without turning `main.py` into a very
    large script.
    """
    print("Launching Minimo interactive chat...")
    from chat import main as chat_main

    chat_main()


def main():
    parser = build_parser()
    args = parser.parse_args()
    mode = args.mode or prompt_for_mode()

    print(f"=== Starting Minimo in {mode.upper()} mode ===")

    if mode == "tokenize":
        run_tokenizer_mode(args)
    elif mode == "train":
        run_training_mode()
    elif mode == "vlm":
        run_vlm_mode()
    elif mode == "rag":
        run_rag_mode()
    elif mode == "chat":
        run_chat_mode()


if __name__ == "__main__":
    main()
