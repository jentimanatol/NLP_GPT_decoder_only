"""
train.py

Part I only: train a GPT decoder-only Transformer using next-token prediction.

Main assignment requirements covered:
- train/validation split
- cross-entropy shifted-token loss
- AdamW optimizer
- warmup + cosine LR schedule
- validation NLL and perplexity
- checkpointing
- saved config and results
- fixed random seed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import prepare_datasets
from model import TinyGPT, count_parameters


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    """Choose the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT decoder-only Transformer.")

    parser.add_argument("--data_path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)

    return parser.parse_args()


def learning_rate_schedule(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
) -> float:
    """Warmup followed by cosine decay."""
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * cosine_decay


def lm_shift_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction.

    Args:
        logits: (B, T, V)
        targets: (B, T)

    Returns:
        Scalar cross-entropy loss, ignoring PAD targets.
    """
    V = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


def run_train_epoch(
    model: TinyGPT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pad_id: int,
    global_step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        # GPT next-token shift:
        # input:  batch[:, :-1]
        # target: batch[:, 1:]
        x_in = batch[:, :-1]
        y_tgt = batch[:, 1:]

        lr = learning_rate_schedule(global_step, total_steps, base_lr, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        logits, _ = model(x_in)
        loss = lm_shift_loss(logits, y_tgt, pad_id)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        global_step += 1

    return total_loss / len(loader), global_step


@torch.no_grad()
def evaluate(model: TinyGPT, loader: DataLoader, device: str, pad_id: int):
    """Evaluate validation NLL and perplexity."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        x_in = batch[:, :-1]
        y_tgt = batch[:, 1:]

        logits, _ = model(x_in)
        loss = lm_shift_loss(logits, y_tgt, pad_id)

        total_loss += float(loss.item())

    nll = total_loss / len(loader)
    ppl = math.exp(nll)

    return nll, ppl


def save_loss_plot(rows, output_dir: Path) -> None:
    """Save a train/validation loss curve."""
    epochs = [row["epoch"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    val_nll = [row["val_nll"] for row in rows]

    plt.figure()
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_nll, label="validation NLL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GPT Decoder Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_plot.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing dataset...")
    train_dataset, val_dataset, vocab = prepare_datasets(
        data_path=args.data_path,
        block_size=args.block_size,
        train_fraction=0.9,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = pick_device()
    print(f"Using device: {device}")

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
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Trainable parameters: {params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    global_step = 0
    best_val_nll = float("inf")
    rows = []

    config = vars(args).copy()
    config.update(
        {
            "assignment_part": "Part I NLP GPT decoder-only",
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

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_id=vocab.pad_id,
            global_step=global_step,
            total_steps=total_steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
        )

        val_nll, val_ppl = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            pad_id=vocab.pad_id,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_nll": val_nll,
            "val_ppl": val_ppl,
        }
        rows.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss = {train_loss:.4f} | "
            f"val NLL = {val_nll:.4f} | "
            f"val PPL = {val_ppl:.2f}"
        )

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_nll", "val_ppl"])
        writer.writeheader()
        writer.writerows(rows)

    save_loss_plot(rows, output_dir)

    final_results = {
        "best_val_nll": best_val_nll,
        "best_val_ppl": math.exp(best_val_nll),
        "final_epoch": rows[-1],
    }

    (output_dir / "final_results.json").write_text(
        json.dumps(final_results, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Best validation NLL: {best_val_nll:.4f}")
    print(f"Best validation PPL: {math.exp(best_val_nll):.2f}")
    print(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
