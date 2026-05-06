"""
model.py

Part I only: GPT-style decoder-only Transformer from scratch.

This file is intentionally close to the uploaded recitation demo:
- make_padding_mask
- make_causal_mask
- scaled_dot_product_attention
- MultiHeadSelfAttention
- GPTDecoderBlock
- TinyGPT

The main difference is that the names and comments are polished for homework
submission, and the model returns logits for training/generation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_padding_mask(token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Create a padding mask.

    Args:
        token_ids: Tensor of shape (B, T).
        pad_id: Padding token ID.

    Returns:
        Boolean tensor of shape (B, 1, 1, T).
        True means BLOCK this key position.
    """
    return (token_ids == pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Create the causal GPT mask.

    The upper triangle is True, which means future tokens are blocked.

    Returns:
        Boolean tensor of shape (1, 1, T, T).
    """
    return torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.

    Shape convention:
        q, k, v: (B, H, T, Dh)
        scores:  (B, H, T, T)

    The softmax is applied over the key dimension, dim=-1.
    """
    d_head = q.size(-1)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)

    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, -1e9)

    attn = F.softmax(scores, dim=-1)

    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=training)

    out = attn @ v
    return out, attn


class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention for GPT decoder blocks."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        # One projection creates Q, K, and V together.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Final projection after heads are merged.
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run masked multi-head self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model).
            attn_mask: Boolean mask broadcastable to (B, H, T, T).

        Returns:
            out: Output tensor of shape (B, T, d_model).
            attn: Attention matrix of shape (B, H, T, T).
        """
        B, T, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        out, attn = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
        )

        # Merge heads back to (B, T, d_model).
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.proj(out)

        return out, attn


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embeddings for token positions 0..max_len-1."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pos(positions)


class FeedForward(nn.Module):
    """Position-wise MLP used after self-attention."""

    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPTDecoderBlock(nn.Module):
    """One GPT decoder block using Pre-LayerNorm.

    Formula:

        y = x + Dropout(SelfAttention(LN(x), mask))
        y = y + Dropout(FFN(LN(y)))
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.drop1(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.drop2(ffn_out)

        return x, attn


class TinyGPT(nn.Module):
    """GPT-style decoder-only language model."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 128,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.max_len = max_len

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = LearnablePositionalEmbedding(max_len=max_len, d_model=d_model)

        self.blocks = nn.ModuleList(
            [
                GPTDecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            token_ids: Tensor of token IDs with shape (B, T).

        Returns:
            logits: Tensor of shape (B, T, vocab_size).
            attn_last: Last layer attention tensor of shape (B, H, T, T).
        """
        B, T = token_ids.shape

        if T > self.max_len:
            raise ValueError(f"Input length {T} is larger than max_len {self.max_len}.")

        x = self.emb(token_ids)
        x = self.pos(x)

        causal_mask = make_causal_mask(T, token_ids.device)
        padding_mask = make_padding_mask(token_ids, self.pad_id)
        attn_mask = causal_mask | padding_mask

        attn_last = None

        for block in self.blocks:
            x, attn_last = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, attn_last


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
