from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import os

def train_tokenizer(data_path, vocab_size=6400, output_dir="."):
    """Trains a custom BPE tokenizer optimized for personal usage."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
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
