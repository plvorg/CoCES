from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from coces.utils.io import read_json


MID_PATTERN = re.compile(r"(?:ns:)?([mg]\.[A-Za-z0-9_]+)")
RELATION_PATTERN = re.compile(r"(?:ns:)?([A-Za-z0-9_]+\.[A-Za-z0-9_.]+)")


def iter_raw_examples(path: str | Path, dataset: str) -> Iterator[dict[str, Any]]:
    raw = read_json(path)
    dataset = dataset.lower()
    if dataset == "webqsp":
        yield from _iter_webqsp(raw)
    elif dataset in {"cwq", "complexwebquestions"}:
        yield from _iter_cwq(raw)
    else:
        yield from _iter_generic(raw)


def _iter_webqsp(raw: Any) -> Iterator[dict[str, Any]]:
    questions = raw.get("Questions", raw) if isinstance(raw, dict) else raw
    for index, item in enumerate(questions):
        parses = item.get("Parses", [])
        topic_entities: list[str] = []
        answers: list[str] = []
        gold_relations: set[str] = set()
        for parse in parses:
            topic = parse.get("TopicEntityMid")
            if topic:
                topic_entities.append(topic)
            for answer in parse.get("Answers", []):
                answer_id = answer.get("AnswerArgument") or answer.get("EntityName")
                if answer_id:
                    answers.append(str(answer_id))
            gold_relations.update(_relations_from_sparql(parse.get("Sparql", "")))
        yield {
            "id": str(item.get("QuestionId", index)),
            "question": item.get("ProcessedQuestion") or item.get("RawQuestion", ""),
            "topic_entities": _unique(topic_entities),
            "gold_answers": _unique(answers),
            "gold_relations": gold_relations,
        }


def _iter_cwq(raw: Any) -> Iterator[dict[str, Any]]:
    items = raw.get("Questions", raw) if isinstance(raw, dict) else raw
    for index, item in enumerate(items):
        sparql = item.get("sparql") or item.get("Sparql") or ""
        topic_entities = (
            item.get("topic_entities")
            or item.get("topicEntityID")
            or item.get("TopicEntityMid")
            or []
        )
        if isinstance(topic_entities, str):
            topic_entities = [topic_entities]
        if not topic_entities:
            topic_entities = MID_PATTERN.findall(sparql)
        answers = item.get("answers") or item.get("Answers") or []
        normalized_answers: list[str] = []
        for answer in answers:
            if isinstance(answer, dict):
                value = (
                    answer.get("answer_id")
                    or answer.get("AnswerArgument")
                    or answer.get("id")
                    or answer.get("value")
                )
            else:
                value = answer
            if value is not None:
                normalized_answers.append(str(value))
        yield {
            "id": str(item.get("ID") or item.get("id") or index),
            "question": item.get("question") or item.get("Question") or "",
            "topic_entities": _unique(list(topic_entities)),
            "gold_answers": _unique(normalized_answers),
            "gold_relations": _relations_from_sparql(sparql),
        }


def _iter_generic(raw: Any) -> Iterator[dict[str, Any]]:
    items = raw.get("examples", raw) if isinstance(raw, dict) else raw
    for index, item in enumerate(items):
        yield {
            "id": str(item.get("id", index)),
            "question": item["question"],
            "topic_entities": list(item.get("topic_entities", [])),
            "gold_answers": list(item.get("gold_answers", [])),
            "gold_relations": set(item.get("gold_relations", [])),
        }


def _relations_from_sparql(sparql: str) -> set[str]:
    return {
        relation
        for relation in RELATION_PATTERN.findall(sparql)
        if not relation.startswith(("m.", "g."))
    }


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))

