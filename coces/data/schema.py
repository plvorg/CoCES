from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidencePath:
    nodes: list[str]
    relations: list[str]
    node_names: list[str] = field(default_factory=list)
    relation_names: list[str] = field(default_factory=list)
    weak_label: float = 0.0
    is_distractor: bool = False
    relation_similarity: float = 0.0
    hub_score: float = 0.0
    path_rank: int = 0
    direction_consistency: float = 1.0

    @property
    def start(self) -> str:
        return self.nodes[0]

    @property
    def end(self) -> str:
        return self.nodes[-1]

    @property
    def length(self) -> int:
        return len(self.relations)

    def verbalize(self) -> str:
        nodes = self.node_names or self.nodes
        relations = self.relation_names or self.relations
        parts = [nodes[0]]
        for relation, node in zip(relations, nodes[1:]):
            parts.extend([f"-- {relation} -->", node])
        return " ".join(parts)

    def structural_features(self, max_path_length: int = 4) -> list[float]:
        return [
            min(self.length / max(max_path_length, 1), 1.0),
            float(self.relation_similarity),
            float(self.hub_score),
            1.0 / (1.0 + max(self.path_rank, 0)),
            float(self.direction_consistency),
        ]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidencePath":
        return cls(**value)


@dataclass
class AnswerCandidate:
    entity_id: str
    name: str = ""
    entity_type: str = ""
    type_match: float = 0.0
    relation_coverage: float = 0.0
    direction_consistency: float = 0.0
    hub_penalty: float = 0.0
    shortest_path: int = 0
    negative_source: str = ""

    def safe_features(self, max_path_length: int = 4) -> list[float]:
        normalized_shortest = (
            self.shortest_path / max(max_path_length, 1) if self.shortest_path else 1.0
        )
        return [
            float(self.type_match),
            float(self.relation_coverage),
            float(self.direction_consistency),
            float(self.hub_penalty),
            float(normalized_shortest),
        ]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnswerCandidate":
        return cls(**value)


@dataclass
class KGQAExample:
    id: str
    question: str
    topic_entities: list[str]
    gold_answers: list[str]
    paths: list[EvidencePath]
    candidates: list[AnswerCandidate]
    relation_cues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KGQAExample":
        return cls(
            id=str(value["id"]),
            question=value["question"],
            topic_entities=list(value.get("topic_entities", [])),
            gold_answers=list(value.get("gold_answers", [])),
            paths=[EvidencePath.from_dict(path) for path in value.get("paths", [])],
            candidates=[
                AnswerCandidate.from_dict(candidate)
                for candidate in value.get("candidates", [])
            ],
            relation_cues=list(value.get("relation_cues", [])),
        )
