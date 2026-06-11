from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from coces.config import CoCESConfig
from coces.data.dataset import CoCESDataset, collate_examples
from coces.data.schema import KGQAExample
from coces.models.coces import CoCESModel
from coces.utils.device import resolve_device
from coces.utils.seed import set_seed

from .losses import (
    LossOutput,
    evaluator_pretraining_loss,
    joint_loss,
    selector_pretraining_loss,
)


LossFunction = Callable[[CoCESModel, KGQAExample], LossOutput]


@dataclass
class Stage:
    name: str
    epochs: int
    loss_function: LossFunction


class CoCESTrainer:
    def __init__(self, model: CoCESModel, config: CoCESConfig) -> None:
        self.model = model
        self.config = config
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)
        set_seed(config.training.seed)
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, train_dataset: CoCESDataset, dev_dataset: CoCESDataset) -> None:
        stages = [
            Stage(
                "evaluator_pretrain",
                self.config.training.evaluator_pretrain_epochs,
                evaluator_pretraining_loss,
            ),
            Stage(
                "selector_pretrain",
                self.config.training.selector_pretrain_epochs,
                selector_pretraining_loss,
            ),
            Stage("joint", self.config.training.joint_epochs, joint_loss),
        ]
        history: list[dict[str, float | int | str]] = []
        for stage in stages:
            if stage.epochs <= 0:
                continue
            self._configure_trainable(stage.name)
            optimizer = self._build_optimizer(stage.name)
            train_loader = self._loader(train_dataset, shuffle=True)
            total_updates = max(
                1,
                math.ceil(
                    len(train_loader)
                    / self.config.training.gradient_accumulation_steps
                )
                * stage.epochs,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_updates * self.config.training.warmup_ratio),
                num_training_steps=total_updates,
            )
            best_dev = float("inf")
            best_state: dict[str, torch.Tensor] | None = None
            for epoch in range(1, stage.epochs + 1):
                train_metrics = self._train_epoch(
                    train_loader, stage, optimizer, scheduler
                )
                dev_metrics = self.evaluate(dev_dataset, stage.loss_function)
                record: dict[str, float | int | str] = {
                    "stage": stage.name,
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"dev_{key}": value for key, value in dev_metrics.items()},
                }
                history.append(record)
                print(json.dumps(record, ensure_ascii=False))
                if dev_metrics["total"] < best_dev:
                    best_dev = dev_metrics["total"]
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    self.model.save(
                        self.output_dir / f"best-{stage.name}",
                        extra={"stage": stage.name, "epoch": epoch},
                    )
            self.model.save(
                self.output_dir / f"last-{stage.name}",
                extra={"stage": stage.name, "epoch": stage.epochs},
            )
            if best_state is not None:
                self.model.load_state_dict(best_state)
                self.model.to(self.device)
        self.model.save(self.output_dir / "final", extra={"stage": "complete"})
        with (self.output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    def evaluate(
        self, dataset: CoCESDataset, loss_function: LossFunction
    ) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = defaultdict(float)
        count = 0
        with torch.no_grad():
            for examples in self._loader(dataset, shuffle=False):
                for example in examples:
                    losses = loss_function(self.model, example).detached()
                    for key, value in losses.items():
                        totals[key] += value
                    count += 1
        return {key: value / max(count, 1) for key, value in totals.items()}

    def _train_epoch(
        self,
        loader: DataLoader[list[KGQAExample]],
        stage: Stage,
        optimizer: AdamW,
        scheduler: object,
    ) -> dict[str, float]:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = defaultdict(float)
        examples_seen = 0
        accumulation = self.config.training.gradient_accumulation_steps
        progress = tqdm(loader, desc=stage.name)
        for batch_index, examples in enumerate(progress, start=1):
            batch_loss = torch.zeros((), device=self.device)
            for example in examples:
                losses = stage.loss_function(self.model, example)
                batch_loss = batch_loss + losses.total
                for key, value in losses.detached().items():
                    totals[key] += value
                examples_seen += 1
            batch_loss = batch_loss / max(len(examples), 1) / accumulation
            batch_loss.backward()
            if batch_index % accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if examples_seen:
                progress.set_postfix(total=totals["total"] / examples_seen)
        return {
            key: value / max(examples_seen, 1) for key, value in totals.items()
        }

    def _configure_trainable(self, stage: str) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        if stage == "evaluator_pretrain":
            for parameter in self.model.selector.parameters():
                parameter.requires_grad = False
        elif stage == "selector_pretrain":
            for parameter in self.model.evaluator.parameters():
                parameter.requires_grad = False

    def _build_optimizer(self, stage: str) -> AdamW:
        training = self.config.training
        if stage == "joint":
            groups = [
                {
                    "params": [
                        parameter
                        for parameter in self.model.selector.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": training.selector_lr,
                },
                {
                    "params": [
                        parameter
                        for parameter in self.model.evaluator.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": training.joint_evaluator_lr,
                },
            ]
        else:
            lr = (
                training.evaluator_lr
                if stage == "evaluator_pretrain"
                else training.selector_lr
            )
            groups = [
                {
                    "params": [
                        parameter
                        for parameter in self.model.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": lr,
                }
            ]
        return AdamW(groups, weight_decay=training.weight_decay)

    def _loader(self, dataset: CoCESDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_examples,
        )
