from __future__ import annotations

from dataclasses import dataclass

import torch

from coces.config import InferenceConfig
from coces.data.schema import KGQAExample
from coces.models.coces import CoCESModel


@dataclass
class PruningResult:
    predicted_answer: str | None
    predicted_name: str | None
    answer_logits: list[float]
    selector_scores: list[float]
    initial_path_indices: list[int]
    final_path_indices: list[int]
    stable: bool
    support: float
    margin: float
    lir: float


class ConservativePruner:
    def __init__(self, model: CoCESModel, config: InferenceConfig) -> None:
        self.model = model
        self.config = config

    @torch.no_grad()
    def predict(self, example: KGQAExample) -> PruningResult:
        self.model.eval()
        device = next(self.model.parameters()).device
        _, selector_scores = self.model.selector_masks(example)
        path_scores = self.model.evaluator.encode_path_scores(example)
        selected = [
            index
            for index, score in enumerate(selector_scores.tolist())
            if score >= self.config.selector_threshold
        ]
        if (
            len(selected) < self.config.minimum_paths
            and self.config.allow_threshold_relaxation
            and len(example.paths)
        ):
            count = min(self.config.minimum_paths, len(example.paths))
            selected = torch.topk(selector_scores, k=count).indices.tolist()
        masks = torch.zeros(len(example.paths), device=device)
        if selected:
            masks[selected] = 1.0
        logits = self.model.answer_logits(example, masks, path_scores)
        predicted_index, support, margin = score_state(logits)
        stable = self._is_stable(support, margin)
        initial = sorted(selected)
        if not stable:
            return self._result(
                example,
                logits,
                selector_scores,
                initial,
                initial,
                False,
                support,
                margin,
                path_scores,
            )
        if not self.config.enable_pruning:
            return self._result(
                example,
                logits,
                selector_scores,
                initial,
                initial,
                True,
                support,
                margin,
                path_scores,
            )

        contributions = []
        base_support = logits[predicted_index]
        for path_index in initial:
            deleted = masks.clone()
            deleted[path_index] = 0.0
            deleted_logits = self.model.answer_logits(example, deleted, path_scores)
            contributions.append(
                (float(base_support - deleted_logits[predicted_index]), path_index)
            )
        current_masks = masks.clone()
        for _, path_index in sorted(contributions):
            trial = current_masks.clone()
            trial[path_index] = 0.0
            trial_logits = self.model.answer_logits(example, trial, path_scores)
            trial_index, trial_support, trial_margin = score_state(trial_logits)
            if (
                trial_index == predicted_index
                and self._is_stable(trial_support, trial_margin)
            ):
                current_masks = trial

        final = torch.nonzero(current_masks > 0.5, as_tuple=False).flatten().tolist()
        final_logits = self.model.answer_logits(example, current_masks, path_scores)
        _, final_support, final_margin = score_state(final_logits)
        return self._result(
            example,
            final_logits,
            selector_scores,
            initial,
            final,
            True,
            final_support,
            final_margin,
            path_scores,
        )

    def _is_stable(self, support: float, margin: float) -> bool:
        return (
            support >= self.config.support_threshold
            and margin >= self.config.margin_threshold
        )

    def _result(
        self,
        example: KGQAExample,
        logits: torch.Tensor,
        selector_scores: torch.Tensor,
        initial: list[int],
        final: list[int],
        stable: bool,
        support: float,
        margin: float,
        path_scores: torch.Tensor,
    ) -> PruningResult:
        predicted_index = int(torch.argmax(logits)) if len(logits) else -1
        candidate = example.candidates[predicted_index] if predicted_index >= 0 else None
        lir = self._local_irreducibility(
            example, final, predicted_index, path_scores
        )
        return PruningResult(
            predicted_answer=candidate.entity_id if stable and candidate else None,
            predicted_name=(candidate.name or candidate.entity_id) if stable and candidate else None,
            answer_logits=[float(value) for value in logits.cpu()],
            selector_scores=[float(value) for value in selector_scores.cpu()],
            initial_path_indices=initial,
            final_path_indices=final,
            stable=stable,
            support=support,
            margin=margin,
            lir=lir,
        )

    def _local_irreducibility(
        self,
        example: KGQAExample,
        final: list[int],
        predicted_index: int,
        path_scores: torch.Tensor,
    ) -> float:
        if not final or predicted_index < 0:
            return 0.0
        device = path_scores.device
        masks = torch.zeros(len(example.paths), device=device)
        masks[final] = 1.0
        non_deletable = 0
        for path_index in final:
            trial = masks.clone()
            trial[path_index] = 0.0
            logits = self.model.answer_logits(example, trial, path_scores)
            trial_index, support, margin = score_state(logits)
            if trial_index != predicted_index or not self._is_stable(support, margin):
                non_deletable += 1
        return non_deletable / len(final)


def score_state(logits: torch.Tensor) -> tuple[int, float, float]:
    if not len(logits):
        return -1, float("-inf"), float("-inf")
    values, indices = torch.sort(logits, descending=True)
    support = float(values[0])
    margin = float(values[0] - values[1]) if len(values) > 1 else float("inf")
    return int(indices[0]), support, margin
