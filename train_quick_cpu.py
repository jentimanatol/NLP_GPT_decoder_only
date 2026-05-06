"""
train_quick_cpu.py
Fast CPU-safe training for HW Part I: GPT decoder-only Transformer.

Why this file exists:
- The original train.py uses full epochs over Tiny Shakespeare, which can take a long time on CPU.
- This version trains by a fixed number of random mini-batch steps.
- It prints progress frequently, saves outputs compatible with generate.py, and finishes quickly.

Outputs:
- outputs/config.json
- outputs/tokenizer.json
- outputs/best_model.pt
- outputs/training_log.csv
- outputs/final_results.json
- outputs/loss_plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import build_char_vocab, load_text
from model import TinyGPT, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick CPU training for GPT decoder-only Transformer.")
    parser.add_argument("--data_path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    # Small CPU-safe defaults.
    parser.add_argument("--block_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--eval_batches", type=int, default=8)

    # Tiny model, still a real decoder-only Transformer.
    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10)
    return parser.parse_args()


def learning_rate_schedule(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_batch(token_ids: torch.Tensor, batch_size: int, block_size: int, device: str):
    """Create random x/y batches for next-token prediction."""
    max_start = token_ids.numel() - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for this block_size.")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([token_ids[i : i + block_size] for i in starts])
    y = torch.stack([token_ids[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


@torch.no_grad()
def evaluate(model: TinyGPT, val_ids: torch.Tensor, args: argparse.Namespace, vocab, device: str):
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = make_batch(val_ids, args.batch_size, args.block_size, device)
        logits, _ = model(x)
        loss = lm_loss(logits, y, vocab.pad_id)
        losses.append(float(loss.item()))
    nll = sum(losses) / max(1, len(losses))
    ppl = math.exp(min(20.0, nll))
    return nll, ppl


def save_loss_plot(rows, output_dir: Path) -> None:
    if not rows:
        return
    steps = [row["step"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    val_nll = [row["val_nll"] for row in rows]

    plt.figure()
    plt.plot(steps, train_loss, label="train loss")
    plt.plot(steps, val_nll, label="validation NLL")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Quick GPT Decoder Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_plot.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing dataset...", flush=True)
    text = load_text(args.data_path)
    vocab = build_char_vocab(text)
    all_ids = torch.tensor(vocab.encode(text) + [vocab.eos_id], dtype=torch.long)

    split_idx = int(0.9 * all_ids.numel())
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    device = pick_device()
    print(f"Using device: {device}", flush=True)

    model = TinyGPT(
        vocab_size=vocab.vocab_size,
        pad_id=vocab.pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.block_size,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    params = count_parameters(model)
    print(f"Vocabulary size: {vocab.vocab_size}", flush=True)
    print(f"Trainable parameters: {params:,}", flush=True)
    print(f"Quick training steps: {args.steps}", flush=True)
    print("Starting training now. You should see progress every few seconds.\n", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()
    config.update(
        {
            "assignment_part": "Part I NLP GPT decoder-only",
            "training_mode": "quick random-batch CPU training",
            "task": "unconditional character-level language modeling",
            "dataset": "Tiny Shakespeare",
            "dataset_split": "90% train, 10% validation",
            "objective": "causal next-token prediction",
            "tokenizer": "character-level",
            "pad_id": vocab.pad_id,
            "eos_id": vocab.eos_id,
            "vocab_size": vocab.vocab_size,
            "parameters": params,
            "device": device,
        }
    )

    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    vocab.save(output_dir / "tokenizer.json")

    rows = []
    best_val_nll = float("inf")
    running_loss = 0.0
    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        x, y = make_batch(train_ids, args.batch_size, args.block_size, device)

        lr = learning_rate_schedule(step - 1, args.steps, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        logits, _ = model(x)
        loss = lm_loss(logits, y, vocab.pad_id)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += float(loss.item())

        # Frequent heartbeat so the user knows it is not frozen.
        if step == 1 or step % 5 == 0:
            elapsed = time.time() - start_time
            print(
                f"heartbeat step {step:04d}/{args.steps} | "
                f"last loss = {loss.item():.4f} | "
                f"lr = {lr:.6f} | "
                f"elapsed = {elapsed:.1f}s",
                flush=True,
            )

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            denom = 1 if step == 1 else args.eval_every
            avg_train = running_loss / denom
            running_loss = 0.0

            val_nll, val_ppl = evaluate(model, val_ids, args, vocab, device)
            row = {
                "step": step,
                "train_loss": avg_train,
                "val_nll": val_nll,
                "val_ppl": val_ppl,
                "lr": lr,
            }
            rows.append(row)

            print(
                f"EVAL step {step:04d}/{args.steps} | "
                f"train loss = {avg_train:.4f} | "
                f"val NLL = {val_nll:.4f} | "
                f"val PPL = {val_ppl:.2f}",
                flush=True,
            )

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print("Saved new best model: outputs/best_model.pt", flush=True)

    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_nll", "val_ppl", "lr"])
        writer.writeheader()
        writer.writerows(rows)

    save_loss_plot(rows, output_dir)

    final_results = {
        "best_val_nll": best_val_nll,
        "best_val_ppl": math.exp(min(20.0, best_val_nll)),
        "final_step": rows[-1] if rows else None,
        "total_seconds": round(time.time() - start_time, 2),
    }
    (output_dir / "final_results.json").write_text(json.dumps(final_results, indent=2), encoding="utf-8")

    print("\nDone.", flush=True)
    print(f"Best validation NLL: {best_val_nll:.4f}", flush=True)
    print(f"Best validation PPL: {math.exp(min(20.0, best_val_nll)):.2f}", flush=True)
    print(f"Saved artifacts to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
