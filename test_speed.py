import time

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


def test_pytorch():
    """
    Measure simple generation throughput for the merged model checkpoint.

    This is a rough local benchmark, not a rigorous profiling suite. The goal is
    to answer a practical question: how quickly can the model produce tokens on
    the current machine under a straightforward generation setting?
    """
    print("Loading PyTorch model...")
    model = AutoModelForCausalLM.from_pretrained(
        "./hf_minimo_merged",
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    tokenizer_obj = Tokenizer.from_file("minimo_tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.pad_token = "<pad>"

    prompt = "<user> Write a very long story about a brave knight and a dragon. <bot>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    print("Generating 100 tokens with PyTorch...")
    start_time = time.time()

    with torch.no_grad():
        model.generate(
            **inputs,
            # A fixed-length decode makes tokens-per-second easy to interpret.
            max_new_tokens=100,
            # Greedy decoding removes sampling randomness from the speed test.
            do_sample=False,
            # KV caching is crucial for decoder performance because it avoids
            # recomputing attention history from scratch at every new token.
            use_cache=True,
        )

    duration = time.time() - start_time
    tokens_per_second = 100 / duration

    print(f"PyTorch Time: {duration:.2f} seconds")
    print(f"PyTorch Speed: {tokens_per_second:.2f} tokens/sec")


if __name__ == "__main__":
    test_pytorch()
