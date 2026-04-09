import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from model import MinimoModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Setup cache dir to external Projects/Data as requested
HF_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Data"))
os.environ["HF_DATASETS_CACHE"] = HF_CACHE
os.makedirs(HF_CACHE, exist_ok=True)

# Hyperparameters for RTX 5060 (8GB VRAM) -> ~15+ Hours Total Training Time
BATCH_SIZE = 1 # Hardware limit for per-device batch
GRAD_ACCUM_STEPS = 16 # Effective Batch Size = 16. Averages gradients over 16 passes before updating.

# --- Step Calculations for Effective Batch Size 16 ---
# 1. Pretraining (~2.11M examples in TinyStories):
#    1 Epoch = 2,119,719 / 16 = 132,482 steps. 
#    At ~1 second per step, 50,000 steps takes ~14 hours and covers a solid chunk of the dataset.
PRETRAIN_STEPS = 50000

# 2. SFT (~75K examples in Magicoder):
#    1 Epoch = 75,197 / 16 = ~4,700 steps.
#    Training for exactly 1 epoch (~1.3 hours) prevents overfitting on instruction formats.
SFT_STEPS = 4700

# 3. DPO (~6.7K examples in dpo-mix-7k):
#    1 Epoch = 6,750 / 16 = ~422 steps.
#    Alignment needs very few steps. 1 epoch (~15-20 minutes) is perfect.
DPO_STEPS = 422

MAX_SEQ_LEN = 256
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_custom_tokenizer():
    tokenizer_path = "minimo_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Tokenizer not found. Run 'python main.py --mode tokenize' first.")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # The padding id might be missing, assume 3 (or <pad>)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        tokenizer.add_special_tokens(["<pad>"])
        pad_id = tokenizer.token_to_id("<pad>")
    return tokenizer, pad_id

def collate_fn_pretrain(batch, tokenizer, pad_id):
    """
    Collate function for pretraining: pads sequences to max length.
    """
    texts = [item["text"] for item in batch]
    encodings = tokenizer.encode_batch(texts)
    
    input_ids = []
    for enc in encodings:
        ids = enc.ids[:MAX_SEQ_LEN]
        # Pad if necessary
        ids = ids + [pad_id] * (MAX_SEQ_LEN - len(ids))
        input_ids.append(ids)
        
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    # Targets for causal LM are shifted inputs
    targets = input_ids.clone()
    return input_ids, targets

def pretrain_model():
    """
    Pretrains the ~105M parameter base model on TinyStories.
    Executes a custom PyTorch training loop.
    """
    print(f"Using HF Cache directory: {HF_CACHE}")
    print("Initializing Minimo Causal LM (~105.6M params)...")
    
    tokenizer, pad_id = load_custom_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with vocab size: {vocab_size}")

    # Initialize model optimized for 8GB VRAM
    model = MinimoModel(vocab_size=vocab_size, dim=768, n_layers=16, n_heads=12, n_kv_heads=4, max_seq_len=MAX_SEQ_LEN)
    model.to(DEVICE)
    
    print("Loading dataset: roneneldan/TinyStories (This may download data to ../Data)")
    dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=HF_CACHE)
    
    # Take a small subset to match steps
    dataset = dataset.select(range(BATCH_SIZE * GRAD_ACCUM_STEPS * PRETRAIN_STEPS * 2))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn_pretrain(b, tokenizer, pad_id)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting Pretraining for {PRETRAIN_STEPS} optimization steps...")
    model.train()
    
    step = 0
    accum_loss = 0
    optimizer.zero_grad()
    
    progress = tqdm(total=PRETRAIN_STEPS, desc="Pretraining")
    
    for i, (input_ids, targets) in enumerate(dataloader):
        if step >= PRETRAIN_STEPS:
            break
            
        input_ids, targets = input_ids.to(DEVICE), targets.to(DEVICE)
        
        # Mixed precision (BF16 or FP16) to save VRAM
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            # Shift targets for causal LM: model output[:-1] predicts target[1:]
            logits, _ = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=pad_id)
            loss = loss / GRAD_ACCUM_STEPS
            
        loss.backward()
        accum_loss += loss.item()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            progress.update(1)
            progress.set_postfix({"loss": accum_loss})
            
            step += 1
            accum_loss = 0
            
    progress.close()
    
    # Save the base model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/minimo_base.pt")
    print("Pretraining complete. Saved base model to checkpoints/minimo_base.pt")
    return model

def collate_fn_sft(batch, tokenizer, pad_id):
    """
    Format Magicoder dataset for SFT.
    """
    # Magicoder contains 'problem' and 'solution'
    prompts = [f"<user> {item['problem']} <bot> {item['solution']}" for item in batch]
    encodings = tokenizer.encode_batch(prompts)
    
    input_ids = []
    for enc in encodings:
        ids = enc.ids[:MAX_SEQ_LEN]
        ids = ids + [pad_id] * (MAX_SEQ_LEN - len(ids))
        input_ids.append(ids)
        
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = input_ids.clone()
    return input_ids, targets

def fine_tune_sft(model):
    """
    Executes Supervised Fine-Tuning (SFT) using Magicoder-OSS-Instruct-75K.
    Applies LoRA via PEFT.
    """
    print("\nPreparing model for LoRA (Parameter-Efficient Fine-Tuning)...")
    
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["wq", "wk", "wv", "wo"], # Target attention projections
        lora_dropout=0.05,
        bias="none"
    )
    
    # Wrap custom PyTorch module with PEFT
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    tokenizer, pad_id = load_custom_tokenizer()
    
    print("Loading dataset: ise-uiuc/Magicoder-OSS-Instruct-75K")
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", cache_dir=HF_CACHE)
    
    # Take enough examples to cover the exact steps requested (padding slightly for shuffle)
    dataset = dataset.select(range(min(len(dataset), BATCH_SIZE * GRAD_ACCUM_STEPS * SFT_STEPS * 2)))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn_sft(b, tokenizer, pad_id)
    )
    
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting SFT for {SFT_STEPS} optimization steps (~1 Epoch)...")
    peft_model.train()
    
    step = 0
    accum_loss = 0
    optimizer.zero_grad()
    
    progress = tqdm(total=SFT_STEPS, desc="SFT")
    
    for i, (input_ids, targets) in enumerate(dataloader):
        if step >= SFT_STEPS:
            break
            
        input_ids, targets = input_ids.to(DEVICE), targets.to(DEVICE)
        
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            # The custom model forward takes input_ids
            # peft_model wraps the base model, so its forward expects what base expects
            # For causal LM, we calculate loss same as pretrain
            logits, _ = peft_model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=pad_id)
            loss = loss / GRAD_ACCUM_STEPS
            
        loss.backward()
        accum_loss += loss.item()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            progress.update(1)
            progress.set_postfix({"loss": accum_loss})
            
            step += 1
            accum_loss = 0
            
    progress.close()
    
    os.makedirs("checkpoints/sft_adapter", exist_ok=True)
    # Save the PEFT adapter
    peft_model.save_pretrained("checkpoints/sft_adapter")
    print("SFT complete. Saved instruction-tuned adapter.")
    return peft_model

def get_batch_logprobs(logits, labels, pad_id):
    """
    Computes log probabilities for DPO loss.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather the log probs of the actual labels
    label_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out padding
    loss_mask = (shift_labels != pad_id)
    label_log_probs = label_log_probs * loss_mask
    
    # Sum over sequence length
    return label_log_probs.sum(dim=-1)

def collate_fn_dpo(batch, tokenizer, pad_id):
    """
    Format DPO dataset containing chosen and rejected responses.
    """
    # dpo-mix-7k typically has 'prompt', 'chosen', 'rejected'
    # Fallback to general formatting if structure differs slightly
    try:
        chosen_texts = [f"<user> {item['prompt']} <bot> {item['chosen'][0]['content'] if isinstance(item['chosen'], list) else item['chosen']}" for item in batch]
        rejected_texts = [f"<user> {item['prompt']} <bot> {item['rejected'][0]['content'] if isinstance(item['rejected'], list) else item['rejected']}" for item in batch]
    except KeyError:
        # Generic fallback
        chosen_texts = ["<user> " + str(item) + " <bot> chosen" for item in batch]
        rejected_texts = ["<user> " + str(item) + " <bot> rejected" for item in batch]
        
    chosen_enc = tokenizer.encode_batch(chosen_texts)
    rejected_enc = tokenizer.encode_batch(rejected_texts)
    
    chosen_ids, rejected_ids = [], []
    for c_enc, r_enc in zip(chosen_enc, rejected_enc):
        c_ids = c_enc.ids[:MAX_SEQ_LEN]
        c_ids = c_ids + [pad_id] * (MAX_SEQ_LEN - len(c_ids))
        chosen_ids.append(c_ids)
        
        r_ids = r_enc.ids[:MAX_SEQ_LEN]
        r_ids = r_ids + [pad_id] * (MAX_SEQ_LEN - len(r_ids))
        rejected_ids.append(r_ids)
        
    return torch.tensor(chosen_ids, dtype=torch.long), torch.tensor(rejected_ids, dtype=torch.long)

def align_dpo(model):
    """
    Direct Preference Optimization (DPO) on dpo-mix-7k.
    """
    print("\nLoading dataset: argilla/dpo-mix-7k")
    dataset = load_dataset("argilla/dpo-mix-7k", split="train", cache_dir=HF_CACHE)
    
    # Limit max steps for DPO alignment
    dpo_steps = DPO_STEPS
    dataset = dataset.select(range(min(len(dataset), BATCH_SIZE * GRAD_ACCUM_STEPS * dpo_steps * 2)))
    
    tokenizer, pad_id = load_custom_tokenizer()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn_dpo(b, tokenizer, pad_id)
    )
    
    # DPO requires a reference model (frozen).
    # Since we are using PEFT, we can disable the adapter to act as the reference model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1) # Lower LR for DPO
    
    beta = 0.1 # DPO temperature
    print(f"Starting Direct Preference Optimization (DPO) for {dpo_steps} steps...")
    
    step = 0
    accum_loss = 0
    optimizer.zero_grad()
    
    progress = tqdm(total=dpo_steps, desc="DPO")
    
    for i, (chosen_ids, rejected_ids) in enumerate(dataloader):
        if step >= dpo_steps:
            break
            
        chosen_ids, rejected_ids = chosen_ids.to(DEVICE), rejected_ids.to(DEVICE)
        
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            # 1. Get policy log probs (adapter enabled)
            chosen_logits, _ = model(chosen_ids)
            rejected_logits, _ = model(rejected_ids)
            
            policy_chosen_logps = get_batch_logprobs(chosen_logits, chosen_ids, pad_id)
            policy_rejected_logps = get_batch_logprobs(rejected_logits, rejected_ids, pad_id)
            
            # 2. Get reference log probs (adapter disabled)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_chosen_logits, _ = model(chosen_ids)
                    ref_rejected_logits, _ = model(rejected_ids)
                
                ref_chosen_logps = get_batch_logprobs(ref_chosen_logits, chosen_ids, pad_id)
                ref_rejected_logps = get_batch_logprobs(ref_rejected_logits, rejected_ids, pad_id)
            
            # 3. Calculate DPO Loss
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            logits_diff = pi_logratios - ref_logratios
            loss = -F.logsigmoid(beta * logits_diff).mean()
            loss = loss / GRAD_ACCUM_STEPS
            
        loss.backward()
        accum_loss += loss.item()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            progress.update(1)
            progress.set_postfix({"loss": accum_loss})
            
            step += 1
            accum_loss = 0
            
    progress.close()
    
    os.makedirs("checkpoints/dpo_adapter", exist_ok=True)
    model.save_pretrained("checkpoints/dpo_adapter")
    print("DPO complete. Model aligned to human preferences. Saved to checkpoints/dpo_adapter.")
    return model

if __name__ == "__main__":
    print("=== Minimo Full Training Pipeline ===")
    print("Note: Running the full optimized training pipeline for RTX 5060 (>15 hours total estimated time).")
    
    # 1. Pretrain base model
    base_model = pretrain_model()
    
    # 2. SFT using LoRA
    sft_model = fine_tune_sft(base_model)
    
    # 3. DPO Alignment
    aligned_model = align_dpo(sft_model)
    
    print("=== Training Pipeline Fully Completed ===")
