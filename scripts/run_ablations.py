#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate all CoCES ablations.")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for config_path in sorted(Path(args.config_dir).glob("*.yaml")):
        name = config_path.stem
        subprocess.run(
            [sys.executable, "scripts/train.py", "--config", str(config_path)],
            check=True,
        )
        checkpoint = _checkpoint_from_config(config_path)
        predictions = output_dir / f"{name}.jsonl"
        subprocess.run(
            [
                sys.executable,
                "scripts/predict.py",
                "--checkpoint",
                str(checkpoint),
                "--input",
                args.test_file,
                "--output",
                str(predictions),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--predictions",
                str(predictions),
                "--output",
                str(output_dir / f"{name}-metrics.json"),
            ],
            check=True,
        )


def _checkpoint_from_config(path: Path) -> Path:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return Path(config["training"]["output_dir"]) / "final"


if __name__ == "__main__":
    main()

