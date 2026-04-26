"""Recall@K evaluation metrics for image retrieval."""


def recall_at_k(
    retrieved_indices: list[int],
    relevant_indices: list[int],
    k: int,
) -> float:
    """Compute Recall@K for a single query.

    Args:
        retrieved_indices: Ordered list of retrieved item indices (most
            similar first).  Only the first ``k`` elements are considered.
        relevant_indices: Set of ground-truth relevant item indices for
            this query.
        k: Cut-off rank.

    Returns:
        Fraction of relevant items found in the top-K results.  Returns
        ``0.0`` when ``relevant_indices`` is empty.
    """
    if not relevant_indices:
        return 0.0

    top_k = set(list(retrieved_indices)[:k])
    relevant_set = set(relevant_indices)
    return len(top_k & relevant_set) / len(relevant_set)


def mean_recall_at_k(
    results: list[dict],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute mean Recall@K over multiple queries for several K values.

    Args:
        results: List of per-query result dicts.  Each dict must contain:
            - ``"retrieved"``: ordered list of retrieved indices.
            - ``"relevant"``: list of ground-truth relevant indices.
        k_values: K values to evaluate.  Defaults to ``[1, 5, 10]``.

    Returns:
        Dict mapping ``"recall@K"`` string keys to mean recall floats.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    metrics: dict[str, float] = {}
    for k in k_values:
        scores = [
            recall_at_k(r["retrieved"], r["relevant"], k)
            for r in results
        ]
        metrics[f"recall@{k}"] = sum(scores) / len(scores) if scores else 0.0
    return metrics
