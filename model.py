import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig


class MinimoConfig(PretrainedConfig):
    """
    Configuration object for the text model.

    Hugging Face stores this object in `config.json`, which means the numerical
    design of the model can travel with the weights. That is important because
    a transformer is not just "a bag of tensors"; the loader also needs to know
    the hidden size, number of layers, head layout, and special token ids before
    those tensors can be interpreted correctly.
    """

    model_type = "minimo"

    def __init__(
        self,
        vocab_size=6400,
        hidden_size=896,
        intermediate_size=3584,
        num_hidden_layers=18,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        **kwargs,
    ):
        # `vocab_size=6400` keeps the tokenizer compact. A smaller vocabulary
        # means the embedding table and output head stay much lighter, which is
        # useful when the whole model is being sized around a single 8 GB GPU.
        self.vocab_size = vocab_size

        # `hidden_size=896` sets the width of each token representation. The
        # number was chosen so it divides evenly across 14 attention heads:
        # 896 / 14 = 64 features per head, which is a common, stable head size.
        self.hidden_size = hidden_size

        # `intermediate_size=3584` is exactly 4x the hidden size. That is a very
        # common transformer rule of thumb because the feed-forward block needs
        # a wider internal space than attention in order to mix features richly.
        self.intermediate_size = intermediate_size

        # `num_hidden_layers=18` makes the network deeper than many tiny toy
        # models without pushing it into a size that would be uncomfortable on
        # the target hardware. More layers usually help the model build more
        # abstract features over multiple transformation steps.
        self.num_hidden_layers = num_hidden_layers

        # `num_attention_heads=14` controls how many separate attention patterns
        # the model can learn at each layer. Multiple heads let the model track
        # different relationships in parallel, such as local grammar and longer
        # context links.
        self.num_attention_heads = num_attention_heads

        # `num_key_value_heads=2` turns the attention module into grouped-query
        # attention. Fourteen query heads still read from attention normally,
        # but only two key/value head sets are stored and reused. This saves
        # memory bandwidth and cache space during generation.
        self.num_key_value_heads = num_key_value_heads

        # `max_position_embeddings=2048` sets the longest context the model is
        # architecturally prepared to represent. The training code may choose a
        # shorter sequence length for cost reasons, but the architecture itself
        # keeps room for longer inference experiments.
        self.max_position_embeddings = max_position_embeddings

        # RMSNorm needs a tiny epsilon so division stays numerically safe even
        # when the variance of a vector becomes extremely small.
        self.rms_norm_eps = rms_norm_eps

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    RMSNorm is often used in modern decoder-only transformers because it keeps
    the stabilizing effect of normalization while being slightly simpler than
    LayerNorm. It scales activations based on their magnitude but does not
    subtract the mean first.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_rope_cis(dim, end, theta=10000.0):
    """
    Precompute the complex rotation values used by RoPE.

    Rotary position embeddings encode position by rotating pairs of features in
    query and key vectors. Precomputing the sinusoidal rotations once is cheaper
    than rebuilding them at every layer and every forward pass.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    positions = torch.arange(end, device=freqs.device)
    freqs = torch.outer(positions, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary position embeddings to query and key tensors.

    The tensors are temporarily viewed as complex numbers so a rotation can be
    expressed as a simple multiplication in the complex plane. After the
    rotation, the values are converted back to regular real-valued tensors.
    """

    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    Grouped-query self-attention.

    Standard multi-head attention would give every query head its own key and
    value head. Grouped-query attention keeps many query heads but shares a
    smaller number of key/value heads across them. That tradeoff usually keeps
    quality close to full multi-head attention while reducing memory cost,
    especially during autoregressive generation.
    """

    def __init__(self, config: MinimoConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.hidden_size // self.n_heads

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        batch_size, sequence_length, _ = x.shape

        xq = self.q_proj(x).view(batch_size, sequence_length, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # The key/value heads are repeated so the tensor shapes line up with the
        # larger number of query heads. Conceptually, several query heads are
        # reading from the same stored key/value representation group.
        xk = (
            xk.unsqueeze(3)
            .expand(-1, -1, -1, self.n_rep, -1)
            .reshape(batch_size, sequence_length, self.n_heads, self.head_dim)
        )
        xv = (
            xv.unsqueeze(3)
            .expand(-1, -1, -1, self.n_rep, -1)
            .reshape(batch_size, sequence_length, self.n_heads, self.head_dim)
        )

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        attention_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(attention_weights, xv)
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    """
    SwiGLU feed-forward network.

    The feed-forward block is where each token representation is transformed
    independently after attention has mixed information across positions.
    SwiGLU uses a gated path, which tends to work better than a plain two-layer
    MLP in many modern transformer architectures.
    """

    def __init__(self, config: MinimoConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


class TransformerBlock(nn.Module):
    """
    One decoder block made of attention, feed-forward, and residual paths.

    Residual connections let each layer refine the representation rather than
    rebuild it from scratch. That makes deep transformers much easier to train.
    """

    def __init__(self, config: MinimoConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, freqs_cis, mask):
        attention_input = self.input_layernorm(x)
        x = x + self.self_attn(attention_input, freqs_cis, mask)

        mlp_input = self.post_attention_layernorm(x)
        x = x + self.mlp(mlp_input)
        return x


class MinimoPreTrainedModel(PreTrainedModel):
    """
    Shared Hugging Face base class for Minimo variants.

    Inheriting from `PreTrainedModel` unlocks the Hugging Face save/load
    interface, config integration, and generation utilities. That makes the
    custom model much easier to reuse across training, export, and inference.
    """

    config_class = MinimoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        # A small Gaussian initialization is the standard default for
        # transformer linear layers and embeddings. Starting too large would
        # make the early activations unstable.
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MinimoBaseModel(MinimoPreTrainedModel):
    """
    Transformer backbone without the final language-model head.

    This module turns token ids into contextual hidden states. The causal
    language-model wrapper will place one more linear layer on top so those
    hidden states can predict the next token.
    """

    def __init__(self, config: MinimoConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # The rotary cache is registered as a buffer rather than a parameter so
        # it moves with the model device automatically but is not updated by the
        # optimizer. Multiplying by 2 leaves room for somewhat longer inference
        # experiments than the nominal training sequence length.
        freqs_cis = precompute_rope_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings * 2,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        # Text-only inference passes token ids. Multimodal code can bypass the
        # embedding layer and provide `inputs_embeds` directly after stitching
        # image and text representations together.
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        _, sequence_length, _ = hidden_states.shape
        freqs_cis = self.freqs_cis[:sequence_length].to(hidden_states.device)

        # A causal mask blocks attention from reading future tokens. Without
        # this mask, the model could "peek ahead" during training and the loss
        # would no longer teach genuine next-token prediction.
        mask = None
        if sequence_length > 1:
            mask = torch.full((sequence_length, sequence_length), float("-inf"), device=hidden_states.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis, mask)

        return self.norm(hidden_states)


class MinimoForCausalLM(MinimoPreTrainedModel):
    """
    Full decoder-only language model.

    This wrapper adds the output projection that maps hidden states back into
    vocabulary logits. Those logits are the raw scores for every possible next
    token before sampling or argmax decoding is applied.
    """

    def __init__(self, config: MinimoConfig):
        super().__init__(config)
        self.model = MinimoBaseModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, attention_mask=None, **kwargs):
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # The output at position `t` should predict the token at `t + 1`, so
            # both tensors are shifted before cross-entropy is applied.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Padding tokens are ignored so the loss reflects real text rather
            # than the artificial zeros added to make batches rectangular.
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}


# Registering the architecture lets `AutoConfig` and `AutoModelForCausalLM`
# rebuild the custom model from a saved Hugging Face directory without any
# manual branching elsewhere in the codebase.
AutoConfig.register("minimo", MinimoConfig)
AutoModelForCausalLM.register(MinimoConfig, MinimoForCausalLM)
