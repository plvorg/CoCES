from coces.data.graph import KnowledgeGraph
from coces.data.path_search import SearchConfig, build_example


def test_build_example_marks_gold_path() -> None:
    graph = KnowledgeGraph()
    graph.add_triple("author", "wrote", "book")
    graph.add_triple("book", "adapted_as", "movie")
    graph.add_triple("movie", "directed_by", "director")
    graph.relation_names.update(
        {
            "wrote": "wrote",
            "adapted_as": "film adaptation",
            "directed_by": "directed by",
        }
    )
    example = build_example(
        graph,
        "q1",
        "Who directed the film adaptation written by the author?",
        ["author"],
        ["director"],
        SearchConfig(max_hops=3, max_paths=20),
        {"wrote", "adapted_as", "directed_by"},
    )
    assert "director" in [candidate.entity_id for candidate in example.candidates]
    assert any(path.end == "director" and path.weak_label == 1.0 for path in example.paths)

