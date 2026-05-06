"""
data.py

Part I only: character-level dataset utilities for GPT-style next-token prediction.

This file keeps the data pipeline simple and reproducible:
- load Tiny Shakespeare from a local text file
- build a character vocabulary
- reserve PAD=0 and EOS=1
- split the text into train/validation portions
- create fixed-length sequences for next-token prediction
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class CharVocab:
    """Character-level vocabulary.

    IDs:
        0 = PAD
        1 = EOS
        2..N = real characters from the dataset
    """

    stoi: dict
    itos: list
    pad_id: int = 0
    eos_id: int = 1

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        """Convert text into integer token IDs."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Convert integer token IDs back into text."""
        chars = []
        for idx in ids:
            idx = int(idx)
            if idx in (self.pad_id, self.eos_id):
                continue
            chars.append(self.itos[idx])
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON for reproducible generation."""
        payload = {
            "itos": self.itos,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        """Load vocabulary from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = payload["itos"]
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(
            stoi=stoi,
            itos=itos,
            pad_id=payload.get("pad_id", 0),
            eos_id=payload.get("eos_id", 1),
        )


def build_char_vocab(text: str) -> CharVocab:
    """Build a character vocabulary from raw text."""
    chars = sorted(set(text))
    itos = ["<PAD>", "<EOS>"] + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return CharVocab(stoi=stoi, itos=itos)


def load_text(path: str | Path) -> str:
    """Load the raw dataset text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run download_dataset.bat first."
        )
    return path.read_text(encoding="utf-8")


class GPTTextDataset(Dataset):
    """Fixed-length language modeling dataset.

    Each item returns a sequence of length block_size + 1.
    The training loop then shifts it:

        x_in  = batch[:, :-1]
        y_tgt = batch[:, 1:]

    This mirrors the recitation demo and makes the shift pattern explicit.
    """

    def __init__(self, token_ids: List[int], block_size: int):
        if len(token_ids) <= block_size + 1:
            raise ValueError("Dataset is too small for selected block_size.")

        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - (self.block_size + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        return torch.tensor(chunk, dtype=torch.long)


def prepare_datasets(
    data_path: str | Path,
    block_size: int,
    train_fraction: float = 0.9,
) -> Tuple[GPTTextDataset, GPTTextDataset, CharVocab]:
    """Load text, tokenize it, split it, and create train/validation datasets."""
    text = load_text(data_path)

    vocab = build_char_vocab(text)
    token_ids = vocab.encode(text) + [vocab.eos_id]

    split_idx = int(len(token_ids) * train_fraction)

    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_dataset = GPTTextDataset(train_ids, block_size)
    val_dataset = GPTTextDataset(val_ids, block_size)

    return train_dataset, val_dataset, vocab
