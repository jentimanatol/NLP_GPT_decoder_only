"""
generate.py

Part I only: generate text from a trained GPT decoder-only model.

Supports:
- greedy decoding
- top-k sampling
- top-p sampling
- temperature
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from data import CharVocab
from model import TinyGPT


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from GPT decoder-only model.")

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--method", choices=["greedy", "sample"], default="sample")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.0)

    return parser.parse_args()


@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Sample the next token from logits."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k is not None and top_k > 0:
        values, indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        filtered = torch.zeros_like(probs).scatter_(-1, indices, values)
        probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative > top_p
        remove_mask[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: TinyGPT,
    vocab: CharVocab,
    prompt: str,
    method: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
) -> str:
    """Generate text from a prompt."""
    model.eval()

    ids = vocab.encode(prompt)
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = token_ids[:, -model.max_len :]

        logits, _ = model(context)
        last_logits = logits[:, -1, :]

        if method == "greedy":
            next_token = last_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token = sample_next_token(
                last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        token_ids = torch.cat([token_ids, next_token], dim=1)

    return vocab.decode(token_ids[0].tolist())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    config_path = output_dir / "config.json"
    vocab_path = output_dir / "tokenizer.json"
    checkpoint_path = output_dir / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError("Missing outputs/config.json. Train first.")
    if not vocab_path.exists():
        raise FileNotFoundError("Missing outputs/tokenizer.json. Train first.")
    if not checkpoint_path.exists():
        raise FileNotFoundError("Missing outputs/best_model.pt. Train first.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab = CharVocab.load(vocab_path)

    device = pick_device()

    model = TinyGPT(
        vocab_size=config["vocab_size"],
        pad_id=config["pad_id"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_len=config["block_size"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    result = generate(
        model=model,
        vocab=vocab,
        prompt=args.prompt,
        method=args.method,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print(result)


if __name__ == "__main__":
    main()
