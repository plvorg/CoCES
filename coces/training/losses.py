from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from coces.config import CoCESConfig
from coces.data.schema import KGQAExample
from coces.models.coces import CoCESModel


@dataclass
class LossOutput:
    total: torch.Tensor
    rank: torch.Tensor
    answer: torch.Tensor
    pair: torch.Tensor
    sparse: torch.Tensor
    counterfactual: torch.Tensor
    distractor: torch.Tensor
    weak: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            key: float(value.detach().cpu())
            for key, value in self.__dict__.items()
        }


def evaluator_pretraining_loss(
    model: CoCESModel, example: KGQAExample
) -> LossOutput:
    device = next(model.parameters()).device
    masks = torch.ones(len(example.paths), device=device)
    path_scores = model.evaluator.encode_path_scores(example)
    answer_logits = model.answer_logits(example, masks, path_scores)
    rank, answer, pair = ranking_loss(answer_logits, example, model.config)
    zero = rank.new_zeros(())
    return LossOutput(rank, rank, answer, pair, zero, zero, zero, zero)


def selector_pretraining_loss(
    model: CoCESModel, example: KGQAExample
) -> LossOutput:
    selector_logits, masks = model.selector_masks(example)
    weak = weak_supervision_loss(selector_logits, example, model.config)
    distractor = distractor_loss(masks, example)
    total = (
        model.config.loss.lambda_weak * weak
        + model.config.loss.lambda_distractor * distractor
    )
    zero = total.new_zeros(())
    return LossOutput(total, zero, zero, zero, zero, zero, distractor, weak)


def joint_loss(model: CoCESModel, example: KGQAExample) -> LossOutput:
    output = model(example)
    rank, answer, pair = ranking_loss(
        output["answer_logits"], example, model.config
    )
    sparse = output["masks"].mean() if len(example.paths) else rank.new_zeros(())
    counterfactual = counterfactual_loss(
        model,
        example,
        output["masks"],
        output["path_scores"],
        output["answer_logits"],
    )
    distractor = distractor_loss(output["masks"], example)
    weak = weak_supervision_loss(output["selector_logits"], example, model.config)
    weights = model.config.loss
    total = (
        rank
        + weights.lambda_sparse * sparse
        + weights.lambda_cf * counterfactual
        + weights.lambda_distractor * distractor
        + weights.lambda_weak * weak
    )
    return LossOutput(
        total, rank, answer, pair, sparse, counterfactual, distractor, weak
    )


def ranking_loss(
    answer_logits: torch.Tensor,
    example: KGQAExample,
    config: CoCESConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gold_indices = [
        index
        for index, candidate in enumerate(example.candidates)
        if candidate.entity_id in set(example.gold_answers)
    ]
    if not gold_indices:
        raise ValueError(f"Example {example.id} has no gold answer in candidates")
    log_probs = F.log_softmax(answer_logits, dim=0)
    gold_tensor = torch.tensor(
        gold_indices, dtype=torch.long, device=answer_logits.device
    )
    answer_loss = -torch.logsumexp(log_probs[gold_tensor], dim=0)
    positive_logit = torch.logsumexp(answer_logits[gold_tensor], dim=0)
    negative_indices = [
        index for index in range(len(example.candidates)) if index not in gold_indices
    ]
    if negative_indices:
        negative_tensor = torch.tensor(
            negative_indices, dtype=torch.long, device=answer_logits.device
        )
        pair_loss = F.relu(
            config.loss.margin_mu - positive_logit + answer_logits[negative_tensor]
        ).mean()
    else:
        pair_loss = answer_logits.new_zeros(())
    return (
        answer_loss + config.loss.beta * pair_loss,
        answer_loss,
        pair_loss,
    )


def counterfactual_loss(
    model: CoCESModel,
    example: KGQAExample,
    masks: torch.Tensor,
    path_scores: torch.Tensor,
    base_answer_logits: torch.Tensor,
) -> torch.Tensor:
    gold_answer_indices = [
        index
        for index, candidate in enumerate(example.candidates)
        if candidate.entity_id in set(example.gold_answers)
    ]
    eligible_paths = [
        index for index, path in enumerate(example.paths) if path.end in example.gold_answers
    ]
    if not eligible_paths or not gold_answer_indices:
        return masks.sum() * 0.0
    eligible_tensor = torch.tensor(
        eligible_paths, dtype=torch.long, device=masks.device
    )
    top_count = min(model.config.loss.cf_top_m, len(eligible_paths))
    top_local = torch.topk(masks[eligible_tensor], k=top_count).indices
    selected_indices = eligible_tensor[top_local]
    base_support = torch.logsumexp(
        base_answer_logits[
            torch.tensor(gold_answer_indices, dtype=torch.long, device=masks.device)
        ],
        dim=0,
    )
    penalties: list[torch.Tensor] = []
    for path_index in selected_indices:
        keep = torch.ones_like(masks)
        keep[path_index] = 0.0
        deleted_masks = masks * keep
        deleted_logits = model.answer_logits(example, deleted_masks, path_scores)
        deleted_support = torch.logsumexp(
            deleted_logits[
                torch.tensor(
                    gold_answer_indices, dtype=torch.long, device=masks.device
                )
            ],
            dim=0,
        )
        contribution = base_support - deleted_support
        penalties.append(F.relu(model.config.loss.cf_epsilon - contribution))
    return torch.stack(penalties).mean()


def distractor_loss(masks: torch.Tensor, example: KGQAExample) -> torch.Tensor:
    indices = [
        index for index, path in enumerate(example.paths) if path.is_distractor
    ]
    if not indices:
        return masks.sum() * 0.0
    selected = masks[torch.tensor(indices, dtype=torch.long, device=masks.device)]
    return -torch.log((1.0 - selected).clamp_min(1e-7)).mean()


def weak_supervision_loss(
    selector_logits: torch.Tensor,
    example: KGQAExample,
    config: CoCESConfig,
) -> torch.Tensor:
    if not example.paths:
        return selector_logits.sum() * 0.0
    labels = torch.tensor(
        [path.weak_label for path in example.paths],
        dtype=selector_logits.dtype,
        device=selector_logits.device,
    )
    weights = torch.where(
        labels > 0.5,
        torch.full_like(labels, config.loss.weak_positive_weight),
        torch.full_like(labels, config.loss.weak_negative_weight),
    )
    return F.binary_cross_entropy_with_logits(
        selector_logits, labels, weight=weights
    )
