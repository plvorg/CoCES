from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextCrossEncoder(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = int(self.encoder.config.hidden_size)

    def encode(
        self,
        left: list[str],
        right: list[str],
        max_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        tokens: dict[str, Any] = self.tokenizer(
            left,
            right,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        output = self.encoder(**tokens)
        if getattr(output, "pooler_output", None) is not None:
            pooled = output.pooler_output
        else:
            pooled = output.last_hidden_state[:, 0]
        return self.dropout(pooled)

