#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tqdm import tqdm

from coces.data.adapters import iter_raw_examples
from coces.data.graph import KnowledgeGraph
from coces.data.path_search import SearchConfig, build_example
from coces.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CoCES candidate path datasets.")
    parser.add_argument("--dataset", choices=["webqsp", "cwq", "generic"], required=True)
    parser.add_argument("--input", required=True, help="Raw dataset JSON file.")
    parser.add_argument("--triples", required=True, help="KG triples TSV: head, relation, tail.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--names", help="Optional entity-name TSV.")
    parser.add_argument("--types", help="Optional entity-type TSV.")
    parser.add_argument("--relations", help="Optional relation-name TSV.")
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--max-paths", type=int, default=100)
    parser.add_argument("--max-answers", type=int, default=32)
    parser.add_argument("--max-branching", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = KnowledgeGraph.from_tsv(
        args.triples,
        names_path=args.names,
        types_path=args.types,
        relations_path=args.relations,
    )
    search_config = SearchConfig(
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        max_answers=args.max_answers,
        max_branching=args.max_branching,
    )
    processed = []
    raw_examples = list(iter_raw_examples(args.input, args.dataset))
    for raw in tqdm(raw_examples, desc="Constructing candidate paths"):
        if not raw["question"] or not raw["topic_entities"] or not raw["gold_answers"]:
            continue
        processed.append(
            build_example(
                graph=graph,
                example_id=raw["id"],
                question=raw["question"],
                topic_entities=raw["topic_entities"],
                gold_answers=raw["gold_answers"],
                gold_relations=raw["gold_relations"],
                config=search_config,
            ).to_dict()
        )
    write_jsonl(processed, args.output)


if __name__ == "__main__":
    main()
