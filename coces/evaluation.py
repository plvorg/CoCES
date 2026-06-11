from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class Metrics:
    hits_at_1: float
    f1: float
    aes: float
    lir: float
    coverage: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def compute_metrics(predictions: Iterable[dict[str, Any]]) -> Metrics:
    rows = list(predictions)
    if not rows:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0)
    hits = 0.0
    f1_total = 0.0
    evidence_total = 0.0
    lir_total = 0.0
    covered = 0.0
    for row in rows:
        gold = set(row.get("gold_answers", []))
        predicted_values = row.get("predicted_answers")
        if predicted_values is None:
            predicted = row.get("predicted_answer")
            predicted_values = [predicted] if predicted else []
        predicted_set = set(predicted_values)
        if predicted_values and predicted_values[0] in gold:
            hits += 1.0
        precision = (
            len(predicted_set & gold) / len(predicted_set) if predicted_set else 0.0
        )
        recall = len(predicted_set & gold) / len(gold) if gold else 0.0
        f1_total += (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        evidence_total += len(row.get("final_path_indices", []))
        lir_total += float(row.get("lir", 0.0))
        covered += float(bool(predicted_values))
    count = len(rows)
    return Metrics(
        hits_at_1=hits / count,
        f1=f1_total / count,
        aes=evidence_total / count,
        lir=lir_total / count,
        coverage=covered / count,
    )

