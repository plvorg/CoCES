from __future__ import annotations

import argparse
import json
from dataclasses import asdict


def main() -> None:
    parser = argparse.ArgumentParser(prog="coces")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run three-stage training.")
    train_parser.add_argument("--config", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict and prune evidence.")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--input")
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--generate", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="Compute paper metrics.")
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--output")

    prepare_parser = subparsers.add_parser("prepare", help="Construct candidate paths.")
    prepare_parser.add_argument("--dataset", choices=["webqsp", "cwq", "generic"], required=True)
    prepare_parser.add_argument("--input", required=True)
    prepare_parser.add_argument("--triples", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--names")
    prepare_parser.add_argument("--types")
    prepare_parser.add_argument("--relations")
    prepare_parser.add_argument("--max-hops", type=int, default=3)
    prepare_parser.add_argument("--max-paths", type=int, default=100)
    prepare_parser.add_argument("--max-answers", type=int, default=32)
    prepare_parser.add_argument("--max-branching", type=int, default=80)

    args = parser.parse_args()
    if args.command == "train":
        _train(args)
    elif args.command == "predict":
        _predict(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "prepare":
        _prepare(args)


def _train(args: argparse.Namespace) -> None:
    from coces.config import CoCESConfig
    from coces.data.dataset import CoCESDataset
    from coces.models.coces import CoCESModel
    from coces.training.trainer import CoCESTrainer

    config = CoCESConfig.from_yaml(args.config)
    trainer = CoCESTrainer(CoCESModel(config), config)
    trainer.fit(
        CoCESDataset(config.data.train_file),
        CoCESDataset(config.data.dev_file),
    )


def _predict(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from coces.data.dataset import CoCESDataset
    from coces.inference.generator import EvidenceGenerator
    from coces.inference.pruning import ConservativePruner
    from coces.models.coces import CoCESModel
    from coces.utils.device import resolve_device
    from coces.utils.io import write_jsonl

    model = CoCESModel.load(args.checkpoint)
    config = model.config
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


def _evaluate(args: argparse.Namespace) -> None:
    from coces.evaluation import compute_metrics
    from coces.utils.io import read_jsonl, write_json

    metrics = compute_metrics(read_jsonl(args.predictions)).to_dict()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        write_json(metrics, args.output)


def _prepare(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from coces.data.adapters import iter_raw_examples
    from coces.data.graph import KnowledgeGraph
    from coces.data.path_search import SearchConfig, build_example
    from coces.utils.io import write_jsonl

    graph = KnowledgeGraph.from_tsv(
        args.triples,
        names_path=args.names,
        types_path=args.types,
        relations_path=args.relations,
    )
    search = SearchConfig(
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        max_answers=args.max_answers,
        max_branching=args.max_branching,
    )
    rows = []
    for raw in tqdm(
        list(iter_raw_examples(args.input, args.dataset)),
        desc="Constructing candidate paths",
    ):
        if not raw["question"] or not raw["topic_entities"] or not raw["gold_answers"]:
            continue
        rows.append(
            build_example(
                graph,
                raw["id"],
                raw["question"],
                raw["topic_entities"],
                raw["gold_answers"],
                search,
                raw["gold_relations"],
            ).to_dict()
        )
    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
