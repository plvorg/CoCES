from coces.evaluation import compute_metrics


def test_metrics() -> None:
    metrics = compute_metrics(
        [
            {
                "gold_answers": ["a"],
                "predicted_answer": "a",
                "final_path_indices": [0, 1],
                "lir": 0.5,
            },
            {
                "gold_answers": ["b"],
                "predicted_answer": None,
                "final_path_indices": [],
                "lir": 0.0,
            },
        ]
    )
    assert metrics.hits_at_1 == 0.5
    assert metrics.f1 == 0.5
    assert metrics.aes == 1.0
    assert metrics.lir == 0.25
    assert metrics.coverage == 0.5

