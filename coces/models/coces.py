from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from coces.config import CoCESConfig
from coces.data.schema import KGQAExample

from .evaluator import AnswerSupportEvaluator
from .selector import PathSelector


class CoCESModel(nn.Module):
    def __init__(self, config: CoCESConfig) -> None:
        super().__init__()
        self.config = config
        model_config = config.model
        self.selector = PathSelector(
            encoder_name=model_config.encoder_name,
            max_length=model_config.max_selector_length,
            feature_dim=model_config.structural_feature_dim,
            dropout=model_config.dropout,
            context_aware=model_config.context_aware,
        )
        self.evaluator = AnswerSupportEvaluator(
            encoder_name=model_config.encoder_name,
            max_length=model_config.max_evaluator_length,
            safe_feature_dim=model_config.structural_feature_dim,
            dropout=model_config.dropout,
            eta=model_config.eta,
        )

    def selector_masks(self, example: KGQAExample) -> tuple[torch.Tensor, torch.Tensor]:
        selector_logits = self.selector(example, self.config.data.max_path_length)
        return selector_logits, torch.sigmoid(selector_logits)

    def answer_logits(
        self,
        example: KGQAExample,
        masks: torch.Tensor,
        path_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.evaluator(
            example,
            masks,
            self.config.data.max_path_length,
            path_scores=path_scores,
        )

    def forward(self, example: KGQAExample) -> dict[str, torch.Tensor]:
        selector_logits, masks = self.selector_masks(example)
        path_scores = self.evaluator.encode_path_scores(example)
        answer_logits = self.answer_logits(example, masks, path_scores)
        return {
            "selector_logits": selector_logits,
            "masks": masks,
            "path_scores": path_scores,
            "answer_logits": answer_logits,
        }

    def save(self, directory: str | Path, extra: dict[str, Any] | None = None) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "extra": extra or {},
        }
        torch.save(payload, target / "model.pt")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "CoCESModel":
        payload = torch.load(
            Path(directory) / "model.pt",
            map_location=map_location,
            weights_only=False,
        )
        config = CoCESConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        return model
