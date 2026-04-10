import os

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def prepare_corpus(docs_dir=None, hf_dataset="roneneldan/TinyStories", output_txt="tokenizer_corpus.txt"):
    """
    Build one plain-text corpus file for tokenizer training.

    Tokenizer training is much simpler when all text lives in one place. This
    function optionally mixes local documents with a streamed Hugging Face
    dataset so the tokenizer can learn both project-specific words and more
    general language patterns.
    """
    from datasets import load_dataset

    with open(output_txt, "w", encoding="utf-8") as output_file:
        if docs_dir and os.path.exists(docs_dir):
            if os.path.isdir(docs_dir):
                print(f"Processing documents in directory: {docs_dir}")
                from markitdown import MarkItDown

                markdown_converter = MarkItDown()

                for filename in os.listdir(docs_dir):
                    filepath = os.path.join(docs_dir, filename)

                    # Only files are processed here. Subdirectories are skipped
                    # because recursive ingestion would make it easier to pull in
                    # unrelated files by accident.
                    if os.path.isfile(filepath):
                        print(f"Extracting text from: {filename}")
                        try:
                            result = markdown_converter.convert(filepath)
                            output_file.write(result.text_content + "\n\n")
                            print(f"Added {filename} to the tokenizer corpus.")
                        except Exception as exc:
                            print(f"Failed to extract from {filename}: {exc}")
            else:
                print(f"Warning: {docs_dir} is not a directory.")
        elif docs_dir:
            print(f"Warning: Directory not found: {docs_dir}")
            print("If the path comes from Windows inside WSL, convert it to a Linux path first.")

        if hf_dataset:
            print(f"Downloading or streaming dataset: {hf_dataset}")

            # Streaming is important here because tokenizer training only needs
            # raw text, not random indexing. Streaming avoids downloading the
            # full dataset before the corpus can be assembled.
            dataset = load_dataset(hf_dataset, split="train", streaming=True)
            print("Writing dataset text into the corpus file...")

            count = 0
            for item in dataset:
                text = item.get("text", "")
                if text:
                    output_file.write(text + "\n\n")
                count += 1

                # The cap keeps the corpus at a size that is still comfortable
                # for local BPE training. Larger corpora can improve coverage,
                # but they also raise RAM use and training time noticeably.
                if count >= 200000:
                    break

            print(f"Added {count} examples from {hf_dataset} to the corpus.")

    return output_txt


def train_tokenizer(data_path, vocab_size=6400, output_dir="."):
    """
    Train a byte-level BPE tokenizer for Minimo.

    Byte-level BPE is a practical choice for small local language models because
    it can represent any text without an unknown-character failure mode, while
    still learning useful subword pieces such as common prefixes, suffixes, and
    frequent whole words.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Unicode normalization helps merge visually equivalent text into a more
    # consistent form before token frequencies are counted. That makes the
    # limited vocabulary budget less wasteful.
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.Strip(left=True, right=True),
        ]
    )

    # Byte-level pre-tokenization is the same broad family used by GPT-style
    # tokenizers. It keeps the tokenizer robust to punctuation, whitespace, and
    # unusual Unicode input.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        # `vocab_size=6400` is intentionally compact. A smaller vocabulary keeps
        # the embedding tables light, which matters a lot in a modest-size model.
        vocab_size=vocab_size,
        # The special tokens reserve IDs for padding and the tiny chat template
        # used elsewhere in the project.
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<user>", "<bot>"],
    )

    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Provide a valid text dataset first.")
        return None

    files = [data_path] if isinstance(data_path, str) else data_path

    print("Training tokenizer...")
    tokenizer.train(files, trainer)

    output_path = os.path.join(output_dir, "minimo_tokenizer.json")
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    return tokenizer


if __name__ == "__main__":
    dummy_file = "dummy_dataset.txt"
    with open(dummy_file, "w", encoding="utf-8") as file_handle:
        file_handle.write(
            "Hello, world! This is a dummy dataset for training the custom tokenizer. " * 100
        )

    train_tokenizer(dummy_file)
    os.remove(dummy_file)
