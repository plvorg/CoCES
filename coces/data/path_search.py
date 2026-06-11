from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .graph import KnowledgeGraph
from .schema import AnswerCandidate, EvidencePath, KGQAExample


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
QUESTION_TYPE_HINTS = {
    "who": {"people.person", "person"},
    "where": {"location.location", "location"},
    "when": {"type.datetime", "date"},
}


@dataclass
class SearchConfig:
    max_hops: int = 3
    max_paths: int = 100
    max_answers: int = 32
    max_branching: int = 80
    hub_degree_threshold: int = 1000
    allow_cycles: bool = False


def tokenize_relation(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) > 1}


def extract_relation_cues(question: str) -> list[str]:
    stop = {
        "what", "which", "who", "where", "when", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on", "by", "to", "did", "does", "do",
    }
    return [token for token in tokenize_relation(question) if token not in stop]


def relation_similarity(cues: set[str], relation_name: str) -> float:
    relation_tokens = tokenize_relation(relation_name)
    if not cues or not relation_tokens:
        return 0.0
    intersection = len(cues & relation_tokens)
    return intersection / math.sqrt(len(cues) * len(relation_tokens))


def search_paths(
    graph: KnowledgeGraph,
    topic_entities: list[str],
    question: str,
    config: SearchConfig,
) -> list[EvidencePath]:
    cues = set(extract_relation_cues(question))
    paths: list[EvidencePath] = []
    frontier: list[tuple[list[str], list[str], list[bool]]] = [
        ([entity], [], []) for entity in topic_entities
    ]

    for _depth in range(config.max_hops):
        next_frontier: list[tuple[list[str], list[str], list[bool]]] = []
        for nodes, relations, reverse_flags in frontier:
            edges = sorted(
                graph.neighbors(nodes[-1]),
                key=lambda edge: (
                    -relation_similarity(cues, graph.relation_name(edge.relation)),
                    graph.degree.get(edge.target, 0),
                ),
            )[: config.max_branching]
            for edge in edges:
                if not config.allow_cycles and edge.target in nodes:
                    continue
                new_nodes = nodes + [edge.target]
                new_relations = relations + [edge.relation]
                new_reverse = reverse_flags + [edge.reverse]
                relation_scores = [
                    relation_similarity(cues, graph.relation_name(relation))
                    for relation in new_relations
                ]
                max_degree = max(graph.degree.get(node, 0) for node in new_nodes[1:])
                hub_score = min(max_degree / max(config.hub_degree_threshold, 1), 1.0)
                path = EvidencePath(
                    nodes=new_nodes,
                    relations=new_relations,
                    node_names=[graph.entity_name(node) for node in new_nodes],
                    relation_names=[
                        ("inverse " if reverse else "") + graph.relation_name(relation)
                        for relation, reverse in zip(new_relations, new_reverse)
                    ],
                    relation_similarity=max(relation_scores, default=0.0),
                    hub_score=hub_score,
                    direction_consistency=sum(not flag for flag in new_reverse)
                    / len(new_reverse),
                )
                paths.append(path)
                next_frontier.append((new_nodes, new_relations, new_reverse))
        frontier = next_frontier

    paths.sort(key=_path_priority, reverse=True)
    deduplicated: list[EvidencePath] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for path in paths:
        key = (tuple(path.nodes), tuple(path.relations))
        if key in seen:
            continue
        seen.add(key)
        path.path_rank = len(deduplicated)
        deduplicated.append(path)
        if len(deduplicated) >= config.max_paths:
            break
    return deduplicated


def build_example(
    graph: KnowledgeGraph,
    example_id: str,
    question: str,
    topic_entities: list[str],
    gold_answers: list[str],
    config: SearchConfig,
    gold_relations: set[str] | None = None,
) -> KGQAExample:
    paths = search_paths(graph, topic_entities, question, config)
    gold_set = set(gold_answers)
    gold_relations = gold_relations or set()
    for path in paths:
        relation_match = not gold_relations or set(path.relations).issubset(
            gold_relations
        )
        path.weak_label = float(path.end in gold_set and relation_match)
        path.is_distractor = _is_distractor(path, gold_set)

    candidate_sources = _construct_candidate_answers(
        graph, paths, gold_answers, config.max_answers
    )
    candidates = [
        _build_candidate(
            graph,
            entity,
            paths,
            question,
            config.max_hops,
            negative_source=source,
        )
        for entity, source in candidate_sources
    ]
    return KGQAExample(
        id=example_id,
        question=question,
        topic_entities=topic_entities,
        gold_answers=gold_answers,
        paths=paths,
        candidates=candidates,
        relation_cues=extract_relation_cues(question),
    )


def _path_priority(path: EvidencePath) -> float:
    length_penalty = 0.05 * max(path.length - 1, 0)
    return (
        path.relation_similarity
        + 0.2 * path.direction_consistency
        - 0.4 * path.hub_score
        - length_penalty
    )


def _is_distractor(path: EvidencePath, gold_answers: set[str]) -> bool:
    wrong_endpoint = path.end not in gold_answers
    directionally_wrong = path.direction_consistency < 0.5
    hub_shortcut = path.hub_score >= 0.8
    partial_support = path.relation_similarity > 0.1 and wrong_endpoint
    return wrong_endpoint or directionally_wrong or hub_shortcut or partial_support


def _build_candidate(
    graph: KnowledgeGraph,
    entity: str,
    paths: list[EvidencePath],
    question: str,
    max_hops: int,
    negative_source: str = "",
) -> AnswerCandidate:
    candidate_paths = [path for path in paths if path.end == entity]
    shortest = min((path.length for path in candidate_paths), default=max_hops + 1)
    question_word = question.strip().split(maxsplit=1)[0].lower() if question.strip() else ""
    expected_types = QUESTION_TYPE_HINTS.get(question_word, set())
    actual_types = graph.entity_types.get(entity, set())
    type_match = float(not expected_types or bool(expected_types & actual_types))
    return AnswerCandidate(
        entity_id=entity,
        name=graph.entity_name(entity),
        entity_type=",".join(sorted(actual_types)),
        type_match=type_match,
        relation_coverage=max(
            (path.relation_similarity for path in candidate_paths), default=0.0
        ),
        direction_consistency=max(
            (path.direction_consistency for path in candidate_paths), default=0.0
        ),
        hub_penalty=max((path.hub_score for path in candidate_paths), default=1.0),
        shortest_path=shortest,
        negative_source=negative_source,
    )


def _construct_candidate_answers(
    graph: KnowledgeGraph,
    paths: list[EvidencePath],
    gold_answers: list[str],
    max_answers: int,
) -> list[tuple[str, str]]:
    gold_set = set(gold_answers)
    sources: dict[str, str] = {answer: "gold" for answer in gold_answers}

    for path in paths:
        if path.end not in gold_set:
            source = "wrong_path_endpoint"
            if path.hub_score >= 0.8:
                source = "hub_shortcut"
            elif path.relation_similarity > 0.1:
                source = "relation_similar_wrong"
            sources.setdefault(path.end, source)

    gold_types = set().union(
        *(graph.entity_types.get(answer, set()) for answer in gold_answers)
    )
    if gold_types:
        for entity in sorted(graph.entity_types):
            if len(sources) >= max_answers:
                break
            if entity not in sources and graph.entity_types[entity] & gold_types:
                sources[entity] = "same_type_wrong"

    for entity in sorted(graph.outgoing):
        if entity not in sources:
            sources[entity] = "random"
        if len(sources) >= max_answers:
            break

    ordered = [(answer, "gold") for answer in gold_answers]
    ordered.extend(
        (entity, source)
        for entity, source in sources.items()
        if entity not in gold_set
    )
    return ordered[: max(max_answers, len(gold_answers))]
