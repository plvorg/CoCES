from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_file: str = "data/processed/train.jsonl"
    dev_file: str = "data/processed/dev.jsonl"
    test_file: str = "data/processed/test.jsonl"
    max_paths: int = 100
    max_answers: int = 32
    max_path_length: int = 4
    num_workers: int = 0


@dataclass
class ModelConfig:
    encoder_name: str = "bert-base-uncased"
    max_selector_length: int = 192
    max_evaluator_length: int = 224
    structural_feature_dim: int = 5
    dropout: float = 0.1
    eta: float = 1e-8
    context_aware: bool = False


@dataclass
class LossConfig:
    beta: float = 0.2
    margin_mu: float = 0.5
    lambda_sparse: float = 0.02
    lambda_cf: float = 0.5
    lambda_distractor: float = 0.5
    lambda_weak: float = 0.5
    cf_epsilon: float = 0.1
    cf_top_m: int = 5
    weak_positive_weight: float = 3.0
    weak_negative_weight: float = 1.0


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    evaluator_pretrain_epochs: int = 2
    selector_pretrain_epochs: int = 2
    joint_epochs: int = 5
    evaluator_lr: float = 2e-5
    selector_lr: float = 2e-5
    joint_evaluator_lr: float = 2e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    output_dir: str = "outputs/coces"
    log_every: int = 20


@dataclass
class InferenceConfig:
    selector_threshold: float = 0.45
    support_threshold: float = 0.0
    margin_threshold: float = 0.25
    allow_threshold_relaxation: bool = True
    minimum_paths: int = 1
    enable_pruning: bool = True


@dataclass
class GenerationConfig:
    enabled: bool = False
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    max_new_tokens: int = 128
    temperature: float = 0.0


@dataclass
class CoCESConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CoCESConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CoCESConfig":
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            loss=LossConfig(**raw.get("loss", {})),
            training=TrainingConfig(**raw.get("training", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
