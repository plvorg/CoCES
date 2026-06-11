from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    reverse: bool = False


class KnowledgeGraph:
    """In-memory adjacency index for Freebase-style TSV triples."""

    def __init__(self) -> None:
        self.outgoing: dict[str, list[Edge]] = defaultdict(list)
        self.degree: dict[str, int] = defaultdict(int)
        self.entity_names: dict[str, str] = {}
        self.entity_types: dict[str, set[str]] = defaultdict(set)
        self.relation_names: dict[str, str] = {}

    def add_triple(self, source: str, relation: str, target: str) -> None:
        self.outgoing[source].append(Edge(source, relation, target, False))
        self.outgoing[target].append(Edge(target, relation, source, True))
        self.degree[source] += 1
        self.degree[target] += 1

    def neighbors(self, entity: str) -> list[Edge]:
        return self.outgoing.get(entity, [])

    def entity_name(self, entity: str) -> str:
        return self.entity_names.get(entity, entity)

    def relation_name(self, relation: str) -> str:
        value = self.relation_names.get(relation, relation)
        return value.replace("_", " ").replace(".", " / ")

    @classmethod
    def from_tsv(
        cls,
        triples_path: str | Path,
        names_path: str | Path | None = None,
        types_path: str | Path | None = None,
        relations_path: str | Path | None = None,
    ) -> "KnowledgeGraph":
        graph = cls()
        for row in _read_tsv(triples_path):
            if len(row) >= 3:
                graph.add_triple(row[0], row[1], row[2])
        if names_path:
            graph.entity_names.update(_read_mapping(names_path))
        if relations_path:
            graph.relation_names.update(_read_mapping(relations_path))
        if types_path:
            for row in _read_tsv(types_path):
                if len(row) >= 2:
                    graph.entity_types[row[0]].add(row[1])
        return graph


def _read_tsv(path: str | Path) -> Iterator[list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if row and not row[0].startswith("#"):
                yield row


def _read_mapping(path: str | Path) -> dict[str, str]:
    return {row[0]: row[1] for row in _read_tsv(path) if len(row) >= 2}


def write_triples(triples: Iterable[tuple[str, str, str]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(triples)

