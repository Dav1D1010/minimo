import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MinimoConfig, MinimoForCausalLM


# The dataset cache is redirected outside the repository so repeated runs do not
# keep bloating the project folder with downloaded training data.
HF_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Data"))
os.environ["HF_DATASETS_CACHE"] = HF_CACHE
os.makedirs(HF_CACHE, exist_ok=True)


# The training loop is sized around an RTX 5060 with 8 GB of VRAM.
# `BATCH_SIZE=1` is the physically safe micro-batch size that fits in memory.
# `GRAD_ACCUM_STEPS=16` lets the optimizer see 16 examples per update anyway,
# which produces an effective batch size of 16 without needing 16 examples in
# memory at the same time.
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16


# The project uses a three-stage training schedule.
# `PRETRAIN_STEPS=100000` is long enough to expose the model to a substantial
# slice of TinyStories while still being achievable on the target hardware.
PRETRAIN_STEPS = 100000

# `SFT_STEPS=4700` is roughly one pass over the chosen instruction dataset at
# the configured effective batch size. The intent is adaptation, not memorizing
# the dataset word-for-word through many repeated epochs.
SFT_STEPS = 4700

# `DPO_STEPS=422` is a short alignment pass. Preference tuning usually needs a
# gentler touch than pretraining because it is shaping behavior, not teaching
# language from scratch.
DPO_STEPS = 422


# `MAX_SEQ_LEN=256` keeps the training examples short enough to be affordable.
# Sequence length has a large cost in transformers because attention scales with
# roughly the square of the context length.
MAX_SEQ_LEN = 256

# `LEARNING_RATE=5e-4` is a practical small-model baseline for AdamW. It is
# high enough to make progress on a compact network, but not so large that early
# training becomes wildly unstable.
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def autocast_context():
    """
    Pick a mixed-precision mode that matches the current hardware.

    CUDA GPUs benefit substantially from float16 or bfloat16 during training
    because the model fits more comfortably in memory and matrix math usually
    runs faster. CPU execution falls back to a no-op context manager.
    """
    if DEVICE == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def load_custom_tokenizer():
    """
    Load the locally trained tokenizer and guarantee a padding token exists.

    Padding is necessary because PyTorch batches need rectangular tensors, but
    natural language examples rarely have the same length. Adding `<pad>` keeps
    the shape regular while letting the loss function ignore the filler tokens.
    """
    tokenizer_path = "minimo_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Tokenizer not found. Run 'python main.py --mode tokenize' first.")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")

    if pad_id is None:
        tokenizer.add_special_tokens(["<pad>"])
        pad_id = tokenizer.token_to_id("<pad>")

    return tokenizer, pad_id


def pad_or_truncate(token_ids, pad_id):
    """
    Force every token sequence to exactly `MAX_SEQ_LEN`.

    Truncation keeps compute bounded, while padding keeps the batch shape legal.
    This is a common compromise in small educational projects where full dynamic
    packing would add a lot of extra code complexity.
    """
    token_ids = token_ids[:MAX_SEQ_LEN]
    return token_ids + [pad_id] * (MAX_SEQ_LEN - len(token_ids))


def collate_fn_pretrain(batch, tokenizer, pad_id):
    """
    Convert raw TinyStories text into fixed-length token tensors.

    For plain causal language-model pretraining, the input tokens and target
    tokens start out identical. The model code shifts them internally so each
    position learns to predict the next token.
    """
    texts = [item["text"] for item in batch]
    encodings = tokenizer.encode_batch(texts)
    input_ids = [pad_or_truncate(encoding.ids, pad_id) for encoding in encodings]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = input_ids.clone()
    return input_ids, targets


def pretrain_model():
    """
    Train the base language model on TinyStories.

    TinyStories is a good starter corpus for a compact model because the text is
    clean, abundant, and simple enough for a smaller network to model without
    requiring giant compute budgets.
    """
    print(f"Using HF cache directory: {HF_CACHE}")

    tokenizer, pad_id = load_custom_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with vocabulary size: {vocab_size}")

    print("Initializing the base Minimo model...")
    config = MinimoConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_id,
        max_position_embeddings=MAX_SEQ_LEN,
    )
    model = MinimoForCausalLM(config)
    model.to(DEVICE)

    print("Loading dataset: roneneldan/TinyStories")
    dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=HF_CACHE)

    # A capped subset keeps the dataloader from walking far beyond the number of
    # items needed for the requested number of optimizer steps.
    usable_examples = BATCH_SIZE * GRAD_ACCUM_STEPS * PRETRAIN_STEPS * 2
    dataset = dataset.select(range(min(len(dataset), usable_examples)))

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_pretrain(batch, tokenizer, pad_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting pretraining for {PRETRAIN_STEPS} optimizer steps...")
    model.train()
    step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad()
    progress = tqdm(total=PRETRAIN_STEPS, desc="Pretraining")

    for batch_index, (input_ids, targets) in enumerate(dataloader):
        if step >= PRETRAIN_STEPS:
            break

        input_ids = input_ids.to(DEVICE)
        targets = targets.to(DEVICE)

        with autocast_context():
            outputs = model(input_ids=input_ids, labels=targets)
            loss = outputs["loss"] / GRAD_ACCUM_STEPS

        loss.backward()
        accumulated_loss += loss.item()

        if (batch_index + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            progress.update(1)
            progress.set_postfix({"loss": accumulated_loss})
            step += 1
            accumulated_loss = 0.0

    progress.close()

    os.makedirs("checkpoints/hf_minimo_base", exist_ok=True)
    model.save_pretrained("checkpoints/hf_minimo_base")
    print("Pretraining complete. Saved the base model to checkpoints/hf_minimo_base")
    return model


def collate_fn_sft(batch, tokenizer, pad_id):
    """
    Format instruction-following examples into a simple chat template.

    The explicit `<user>` and `<bot>` markers teach the model where the prompt
    ends and the expected answer begins. Even a tiny chat format like this gives
    the model clearer structure than concatenating the fields without markers.
    """
    prompts = [f"<user> {item['problem']} <bot> {item['solution']}" for item in batch]
    encodings = tokenizer.encode_batch(prompts)
    input_ids = [pad_or_truncate(encoding.ids, pad_id) for encoding in encodings]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = input_ids.clone()
    return input_ids, targets


def fine_tune_sft(model):
    """
    Run supervised fine-tuning with LoRA adapters.

    LoRA is used because it drastically reduces the number of trainable weights.
    Instead of updating the full base model, small low-rank matrices are learned
    inside selected layers. That makes instruction tuning much cheaper in both
    memory and storage.
    """
    print("\nPreparing LoRA adapters for supervised fine-tuning...")

    lora_config = LoraConfig(
        # `r=8` keeps the adapter compact. Higher rank increases capacity but
        # also raises memory use and the risk of overfitting small SFT datasets.
        r=8,
        # `lora_alpha=32` scales the LoRA update. A value several times larger
        # than the rank is a common choice that keeps the adapter expressive.
        lora_alpha=32,
        # Attention projections are the most influential and cost-effective
        # places to adapt behavior in many decoder-only models.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # `0.05` adds a little regularization so the adapter does not cling too
        # hard to quirks of the instruction dataset.
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    tokenizer, pad_id = load_custom_tokenizer()

    print("Loading dataset: ise-uiuc/Magicoder-OSS-Instruct-75K")
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", cache_dir=HF_CACHE)
    usable_examples = BATCH_SIZE * GRAD_ACCUM_STEPS * SFT_STEPS * 2
    dataset = dataset.select(range(min(len(dataset), usable_examples)))

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_sft(batch, tokenizer, pad_id),
    )

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=LEARNING_RATE)

    print(f"Starting supervised fine-tuning for {SFT_STEPS} optimizer steps...")
    peft_model.train()
    step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad()
    progress = tqdm(total=SFT_STEPS, desc="SFT")

    for batch_index, (input_ids, targets) in enumerate(dataloader):
        if step >= SFT_STEPS:
            break

        input_ids = input_ids.to(DEVICE)
        targets = targets.to(DEVICE)

        with autocast_context():
            outputs = peft_model(input_ids=input_ids, labels=targets)
            loss = outputs["loss"] / GRAD_ACCUM_STEPS

        loss.backward()
        accumulated_loss += loss.item()

        if (batch_index + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            progress.update(1)
            progress.set_postfix({"loss": accumulated_loss})
            step += 1
            accumulated_loss = 0.0

    progress.close()

    os.makedirs("checkpoints/hf_sft_adapter", exist_ok=True)
    peft_model.save_pretrained("checkpoints/hf_sft_adapter")
    print("SFT complete. Saved the adapter to checkpoints/hf_sft_adapter")
    return peft_model


def get_batch_logprobs(logits, labels, pad_id):
    """
    Compute sequence log-probabilities for DPO.

    DPO compares how strongly the model prefers a chosen response over a
    rejected one. Summing token log-probabilities gives one score per sequence,
    which is what the preference objective needs.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    label_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

    loss_mask = shift_labels != pad_id
    label_log_probs = label_log_probs * loss_mask
    return label_log_probs.sum(dim=-1)


def collate_fn_dpo(batch, tokenizer, pad_id):
    """
    Build paired chosen and rejected sequences for preference learning.

    The dataset schema can vary a little across examples, so the function is
    intentionally defensive when extracting the text fields.
    """
    try:
        chosen_texts = [
            f"<user> {item['prompt']} <bot> "
            f"{item['chosen'][0]['content'] if isinstance(item['chosen'], list) else item['chosen']}"
            for item in batch
        ]
        rejected_texts = [
            f"<user> {item['prompt']} <bot> "
            f"{item['rejected'][0]['content'] if isinstance(item['rejected'], list) else item['rejected']}"
            for item in batch
        ]
    except KeyError:
        chosen_texts = [f"<user> {item} <bot> chosen" for item in batch]
        rejected_texts = [f"<user> {item} <bot> rejected" for item in batch]

    chosen_encodings = tokenizer.encode_batch(chosen_texts)
    rejected_encodings = tokenizer.encode_batch(rejected_texts)

    chosen_ids = [pad_or_truncate(encoding.ids, pad_id) for encoding in chosen_encodings]
    rejected_ids = [pad_or_truncate(encoding.ids, pad_id) for encoding in rejected_encodings]

    return (
        torch.tensor(chosen_ids, dtype=torch.long),
        torch.tensor(rejected_ids, dtype=torch.long),
    )


def align_dpo(model):
    """
    Run Direct Preference Optimization on top of the SFT adapter.

    DPO does not require a reward model. Instead, it compares the policy model
    against a frozen reference version and nudges the policy toward answers that
    humans preferred in the dataset.
    """
    print("\nLoading dataset: argilla/dpo-mix-7k")
    dataset = load_dataset("argilla/dpo-mix-7k", split="train", cache_dir=HF_CACHE)
    usable_examples = BATCH_SIZE * GRAD_ACCUM_STEPS * DPO_STEPS * 2
    dataset = dataset.select(range(min(len(dataset), usable_examples)))

    tokenizer, pad_id = load_custom_tokenizer()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_dpo(batch, tokenizer, pad_id),
    )

    # DPO is deliberately run at a smaller learning rate than pretraining and
    # SFT because alignment should gently reshape behavior rather than rewrite
    # the model's language knowledge.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1)

    # `beta=0.1` controls how strongly the policy is pushed away from the
    # reference model. A modest beta keeps the preference update from becoming
    # too aggressive on a relatively small alignment set.
    beta = 0.1

    print(f"Starting DPO for {DPO_STEPS} optimizer steps...")
    step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad()
    progress = tqdm(total=DPO_STEPS, desc="DPO")

    for batch_index, (chosen_ids, rejected_ids) in enumerate(dataloader):
        if step >= DPO_STEPS:
            break

        chosen_ids = chosen_ids.to(DEVICE)
        rejected_ids = rejected_ids.to(DEVICE)

        with autocast_context():
            chosen_logits = model(input_ids=chosen_ids)["logits"]
            rejected_logits = model(input_ids=rejected_ids)["logits"]

            policy_chosen_logps = get_batch_logprobs(chosen_logits, chosen_ids, pad_id)
            policy_rejected_logps = get_batch_logprobs(rejected_logits, rejected_ids, pad_id)

            # The adapter is disabled temporarily to recover the reference model.
            # That gives DPO a stable baseline to compare against without having
            # to keep a second fully separate model in memory.
            with torch.no_grad():
                with model.disable_adapter():
                    ref_chosen_logits = model(input_ids=chosen_ids)["logits"]
                    ref_rejected_logits = model(input_ids=rejected_ids)["logits"]

                ref_chosen_logps = get_batch_logprobs(ref_chosen_logits, chosen_ids, pad_id)
                ref_rejected_logps = get_batch_logprobs(ref_rejected_logits, rejected_ids, pad_id)

            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps

            logits_diff = pi_logratios - ref_logratios
            loss = -F.logsigmoid(beta * logits_diff).mean() / GRAD_ACCUM_STEPS

        loss.backward()
        accumulated_loss += loss.item()

        if (batch_index + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            progress.update(1)
            progress.set_postfix({"loss": accumulated_loss})
            step += 1
            accumulated_loss = 0.0

    progress.close()

    os.makedirs("checkpoints/hf_dpo_adapter", exist_ok=True)
    model.save_pretrained("checkpoints/hf_dpo_adapter")
    print("DPO complete. Saved the aligned adapter to checkpoints/hf_dpo_adapter")
    return model


if __name__ == "__main__":
    print("=== Minimo Hugging Face Training Pipeline ===")

    if os.path.exists("hf_minimo"):
        print("Found an existing base model in 'hf_minimo'. Skipping pretraining.")
        base_model = MinimoForCausalLM.from_pretrained("hf_minimo")
        base_model.to(DEVICE)
    else:
        base_model = pretrain_model()

    sft_model = fine_tune_sft(base_model)
    align_dpo(sft_model)
    print("=== Training Pipeline Fully Completed ===")
