from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import os

def prepare_corpus(docs_dir=None, hf_dataset="roneneldan/TinyStories", output_txt="tokenizer_corpus.txt"):
    """Extracts text from a directory of documents and a HuggingFace dataset, saving to a text file."""
    import tqdm
    from datasets import load_dataset
    
    with open(output_txt, "w", encoding="utf-8") as out_f:
        if docs_dir and os.path.exists(docs_dir):
            if os.path.isdir(docs_dir):
                print(f"Processing documents in directory: {docs_dir}")
                from markitdown import MarkItDown
                md = MarkItDown()
                
                for filename in os.listdir(docs_dir):
                    filepath = os.path.join(docs_dir, filename)
                    # Skip subdirectories or non-document files if necessary
                    if os.path.isfile(filepath):
                        print(f"Extracting text from: {filename}...")
                        try:
                            # MarkItDown handles pdf, docx, pptx, md, txt, etc.
                            result = md.convert(filepath)
                            out_f.write(result.text_content + "\n\n")
                            print(f"Added {filename} to corpus.")
                        except Exception as e:
                            print(f"Failed to extract from {filename}: {e}")
            else:
                print(f"Warning: {docs_dir} is not a directory. Please provide a folder path.")
        elif docs_dir:
            print(f"Warning: Directory not found: {docs_dir}.")
            print("If you are using WSL, ensure your Windows path is converted to a Linux path (e.g., /mnt/d/...).")
                
        if hf_dataset:
            print(f"Downloading/Loading HF Dataset: {hf_dataset}")
            # We stream a subset (e.g., 200,000 stories) so the text file doesn't get too large (BPE training loads it all into RAM)
            ds = load_dataset(hf_dataset, split="train", streaming=True)
            print("Writing dataset to corpus file...")
            count = 0
            for item in ds:
                text = item.get("text", "")
                if text:
                    out_f.write(text + "\n\n")
                count += 1
                if count >= 200000: # Limit to 200k stories
                    break
            print(f"Added {count} examples from {hf_dataset} to corpus.")
            
    return output_txt

def train_tokenizer(data_path, vocab_size=6400, output_dir="."):
    """Trains a custom BPE tokenizer optimized for personal usage."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Normalization: NFKC (standard unicode normalization) and strip accents if needed
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Strip(left=True, right=True)
    ])
    
    # Pre-processing: ByteLevel (standard for BPE like GPT-2/Llama)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<user>", "<bot>"]
    )

    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Please provide a valid text dataset for tokenizer training.")
        return None

    # Assuming the data path is a single text file or a list of files
    files = [data_path] if isinstance(data_path, str) else data_path
    
    print("Training tokenizer...")
    tokenizer.train(files, trainer)
    
    output_path = os.path.join(output_dir, "minimo_tokenizer.json")
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    
    return tokenizer

if __name__ == "__main__":
    # Example usage: create a dummy file and train
    dummy_file = "dummy_dataset.txt"
    with open(dummy_file, "w") as f:
        f.write("Hello, world! This is a dummy dataset for training the custom tokenizer. " * 100)
    
    train_tokenizer(dummy_file)
    os.remove(dummy_file)
