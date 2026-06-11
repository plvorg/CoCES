from __future__ import annotations

import math

import torch
from torch import nn

from coces.data.schema import KGQAExample

from .encoder import TextCrossEncoder


class AnswerSupportEvaluator(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        max_length: int,
        safe_feature_dim: int = 5,
        dropout: float = 0.1,
        eta: float = 1e-8,
    ) -> None:
        super().__init__()
        self.text_encoder = TextCrossEncoder(encoder_name, dropout)
        self.max_length = max_length
        self.path_scorer = nn.Linear(self.text_encoder.hidden_size, 1)
        self.safe_scorer = nn.Linear(safe_feature_dim, 1)
        self.eta = eta

    def encode_path_scores(self, example: KGQAExample) -> torch.Tensor:
        device = next(self.parameters()).device
        if not example.paths:
            return torch.empty(0, device=device)
        candidate_names = {
            candidate.entity_id: candidate.name or candidate.entity_id
            for candidate in example.candidates
        }
        left = [example.question] * len(example.paths)
        right = [
            f"{path.verbalize()} [ANSWER] {candidate_names.get(path.end, path.end)}"
            for path in example.paths
        ]
        encoded = self.text_encoder.encode(left, right, self.max_length, device)
        return self.path_scorer(encoded).squeeze(-1)

    def aggregate(
        self,
        example: KGQAExample,
        masks: torch.Tensor,
        path_scores: torch.Tensor,
        max_path_length: int,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        safe_features = torch.tensor(
            [
                candidate.safe_features(max_path_length=max_path_length)
                for candidate in example.candidates
            ],
            dtype=dtype,
            device=device,
        )
        safe_logits = self.safe_scorer(safe_features).squeeze(-1)
        answer_logits: list[torch.Tensor] = []
        for candidate_index, candidate in enumerate(example.candidates):
            indices = [
                index
                for index, path in enumerate(example.paths)
                if path.end == candidate.entity_id
            ]
            if indices:
                index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                selected_masks = masks[index_tensor]
                log_weights = torch.where(
                    selected_masks > 0,
                    torch.log(selected_masks),
                    torch.full_like(selected_masks, float("-inf")),
                )
                weighted_logsum = torch.logsumexp(
                    path_scores[index_tensor] + log_weights,
                    dim=0,
                )
                path_logit = torch.logaddexp(
                    torch.tensor(
                        math.log(self.eta),
                        dtype=dtype,
                        device=device,
                    ),
                    weighted_logsum,
                )
            else:
                path_logit = torch.log(
                    torch.tensor(self.eta, dtype=dtype, device=device)
                )
            answer_logits.append(path_logit + safe_logits[candidate_index])
        return torch.stack(answer_logits)

    def forward(
        self,
        example: KGQAExample,
        masks: torch.Tensor,
        max_path_length: int,
        path_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if path_scores is None:
            path_scores = self.encode_path_scores(example)
        return self.aggregate(example, masks, path_scores, max_path_length)
