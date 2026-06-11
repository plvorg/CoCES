from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from coces.utils.io import read_jsonl

from .schema import KGQAExample


class CoCESDataset(Dataset[KGQAExample]):
    def __init__(self, path: str | Path) -> None:
        self.examples = [KGQAExample.from_dict(value) for value in read_jsonl(path)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> KGQAExample:
        return self.examples[index]


def collate_examples(examples: list[KGQAExample]) -> list[KGQAExample]:
    return examples


def encode_text_pairs(
    tokenizer: Any,
    left: list[str],
    right: list[str],
    max_length: int,
    device: Any,
) -> dict[str, Any]:
    encoded = tokenizer(
        left,
        right,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}

