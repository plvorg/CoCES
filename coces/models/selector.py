from __future__ import annotations

import torch
from torch import nn

from coces.data.schema import KGQAExample

from .encoder import TextCrossEncoder


class PathSelector(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        max_length: int,
        feature_dim: int = 5,
        dropout: float = 0.1,
        context_aware: bool = False,
    ) -> None:
        super().__init__()
        self.text_encoder = TextCrossEncoder(encoder_name, dropout)
        self.max_length = max_length
        self.context_aware = context_aware
        attention_heads = (
            8 if self.text_encoder.hidden_size % 8 == 0 else 1
        )
        self.path_attention = nn.MultiheadAttention(
            self.text_encoder.hidden_size,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(self.text_encoder.hidden_size)
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self.text_encoder.hidden_size + 32, 1)

    def forward(self, example: KGQAExample, max_path_length: int) -> torch.Tensor:
        device = next(self.parameters()).device
        if not example.paths:
            return self.classifier.weight.sum().expand(0)
        question = [example.question] * len(example.paths)
        path_text = [path.verbalize() for path in example.paths]
        text_features = self.text_encoder.encode(
            question, path_text, self.max_length, device
        )
        if self.context_aware and len(example.paths) > 1:
            contextualized, _ = self.path_attention(
                text_features.unsqueeze(0),
                text_features.unsqueeze(0),
                text_features.unsqueeze(0),
                need_weights=False,
            )
            text_features = self.context_norm(
                text_features + contextualized.squeeze(0)
            )
        structural = torch.tensor(
            [
                path.structural_features(max_path_length=max_path_length)
                for path in example.paths
            ],
            dtype=text_features.dtype,
            device=device,
        )
        features = torch.cat(
            [text_features, self.feature_projection(structural)], dim=-1
        )
        return self.classifier(features).squeeze(-1)
