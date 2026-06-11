#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict

from tqdm import tqdm

from coces.config import CoCESConfig
from coces.data.dataset import CoCESDataset
from coces.inference.generator import EvidenceGenerator
from coces.inference.pruning import ConservativePruner
from coces.models.coces import CoCESModel
from coces.utils.device import resolve_device
from coces.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoCES prediction and pruning.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", help="Processed JSONL; defaults to config test file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    model = CoCESModel.load(args.checkpoint)
    config: CoCESConfig = model.config
    model.to(resolve_device(config.training.device))
    dataset = CoCESDataset(args.input or config.data.test_file)
    pruner = ConservativePruner(model, config.inference)
    generator = (
        EvidenceGenerator(config.generation)
        if args.generate or config.generation.enabled
        else None
    )
    rows = []
    for example in tqdm(dataset.examples, desc="Predicting"):
        result = pruner.predict(example)
        row = {
            "id": example.id,
            "question": example.question,
            "gold_answers": example.gold_answers,
            **asdict(result),
            "evidence": [
                example.paths[index].verbalize()
                for index in result.final_path_indices
            ],
        }
        if generator and result.predicted_name:
            row["generated_answer"] = generator.generate(
                example, result.predicted_name, result.final_path_indices
            )
        rows.append(row)
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()

