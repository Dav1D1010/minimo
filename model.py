import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Used instead of LayerNorm because it does not center the mean, saving 
    computation time and simplifying the backward pass, while still maintaining 
    training stability. It's standard in modern architectures like Llama and Qwen.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate the root mean square over the last dimension (d_model).
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def precompute_rope_cis(dim, end, theta=10000.0):
    """
    Precomputes complex exponentials for Rotary Position Embedding (RoPE).
    RoPE provides relative positional information directly within the attention 
    calculation by rotating the Query and Key vectors.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Applies the precomputed Rotary Position Embeddings to Query and Key tensors.
    Transforms them into the complex domain for the rotation operation, 
    then converts them back to real numbers.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim/2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Instead of every Query head having its own Key/Value head (Multi-Head Attention),
    multiple Query heads share a single KV head. 
    
    Why GQA? 
    For an 8GB RTX 5060, KV-cache memory during inference and intermediate state memory 
    during training are critical bottlenecks. Reducing KV heads from 12 to 4 drops the 
    KV parameter count significantly, freeing up space to make the model deeper (16 layers).
    """
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        
        # How many Query heads share a single KV head (e.g., 12 / 4 = 3).
        self.n_rep = n_heads // n_kv_heads
        
        # The dimension of each individual head.
        # e.g., 768 / 12 = 64. A head_dim of 64 or 128 is crucial because hardware 
        # Tensor Cores are highly optimized for powers of 2.
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, T, C = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # GQA Repeat: Expand the KV heads to match the number of Query heads
        # This allows the standard dot-product attention calculation to proceed.
        xk = xk.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(B, T, self.n_heads, self.head_dim)
        xv = xv.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(B, T, self.n_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scale dot-product attention
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        
        # Recombine heads
        output = output.transpose(1, 2).reshape(B, T, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    """
    Standard FeedForward Network using SiLU (Swish) activation.
    The intermediate dimension is typically scaled to 4x the model dimension.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU-style gating mechanism
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    """
    A single block consisting of Pre-Norm -> Attention -> Add -> Pre-Norm -> FFN -> Add.
    Pre-normalization is more stable for deep networks than post-normalization.
    """
    def __init__(self, dim, n_heads, n_kv_heads, hidden_dim):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class MinimoModel(nn.Module):
    """
    The main Causal LM architecture.
    
    Optimized ~217M Parameter Setup for RTX 5060 (8GB VRAM):
    - `n_layers` = 18: Increased depth for better reasoning and abstraction.
    - `dim` (d_model) = 896: Increased width for a larger knowledge capacity (Divisible by 14 heads = 64 head_dim).
    - `n_heads` = 14: Query heads.
    - `n_kv_heads` = 2: Extreme Grouped-Query Attention (7:1 ratio) to keep memory spikes low.
    - `intermediate_dim` = 3584: Exact 4x scaling of d_model for the FFN layer.
    
    Total parameters: ~217.5 Million.
    Expected VRAM footprint for states (AdamW + BF16 Mixed Precision): ~3.0 GB.
    This leaves ~5 GB of VRAM completely free for activations (Batch Size and Sequence Length),
    which is still very safe for an 8GB GPU while significantly pushing model capability.
    """
    def __init__(self, vocab_size, dim=896, n_layers=18, n_heads=14, n_kv_heads=2, max_seq_len=2048):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # FFN intermediate dimension is 4x the base dimension
        intermediate_dim = dim * 4 
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, intermediate_dim) for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(dim)
        
        # The LM Output head. We don't tie embeddings by default here for flexibility, 
        # but tying them (self.output.weight = self.tok_embeddings.weight) 
        # could save ~4.9M parameters if needed.
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        self.freqs_cis = precompute_rope_cis(dim // n_heads, max_seq_len * 2)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:T].to(h.device)

        mask = None
        if T > 1:
            # Causal mask: prevent attending to future tokens
            mask = torch.full((T, T), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)

        # Pass through all 16 Transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # Final normalization before the prediction head
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
