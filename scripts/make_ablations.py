#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


ABLATIONS = {
    "full": {},
    "wo_cf": {"loss.lambda_cf": 0.0},
    "wo_sparse": {"loss.lambda_sparse": 0.0},
    "wo_distractor": {"loss.lambda_distractor": 0.0},
    "wo_weak": {"loss.lambda_weak": 0.0},
    "wo_pruning": {"inference.enable_pruning": False},
    "context_aware": {"model.context_aware": True},
}


def set_nested(config: dict, path: str, value: object) -> None:
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoCES ablation configs.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.base).open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in ABLATIONS.items():
        config = copy.deepcopy(base)
        for path, value in overrides.items():
            set_nested(config, path, value)
        config["training"]["output_dir"] = str(
            Path(config["training"]["output_dir"]).parent / name
        )
        with (output_dir / f"{name}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    main()

