import os

import torch
from peft import PeftModel
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from model import MinimoForCausalLM


print("Loading base model...")

# The base model is loaded in float16 because the goal of this export step is to
# prepare an inference-friendly checkpoint rather than continue high-precision
# training.
base_model = MinimoForCausalLM.from_pretrained("hf_minimo", torch_dtype=torch.float16)

print("Loading DPO adapter...")
model = PeftModel.from_pretrained(base_model, "checkpoints/hf_dpo_adapter")

# `merge_and_unload()` folds the low-rank adapter weights into the base model so
# inference no longer depends on PEFT wrappers or separate adapter files.
merged_model = model.merge_and_unload()

tokenizer = Tokenizer.from_file("minimo_tokenizer.json")
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
hf_tokenizer.pad_token = "<pad>"

output_dir = "hf_minimo_merged"
os.makedirs(output_dir, exist_ok=True)
merged_model.save_pretrained(output_dir)
hf_tokenizer.save_pretrained(output_dir)

print(f"Successfully saved merged model and tokenizer to {output_dir}/")
