def recall_at_k(retrieved_indices: list[int], relevant_indices: list[int], k: int) -> float:
    if not relevant_indices:
        return 0.0
    top_k = set(list(retrieved_indices)[:k])
    return len(top_k & set(relevant_indices)) / len(relevant_indices)


def precision_at_k(retrieved_indices: list[int], relevant_indices: list[int], k: int) -> float:
    if not retrieved_indices or k == 0:
        return 0.0
    top_k = list(retrieved_indices)[:k]
    relevant_set = set(relevant_indices)
    return sum(1 for i in top_k if i in relevant_set) / k


def average_precision(retrieved_indices: list[int], relevant_indices: list[int], k: int) -> float:
    if not relevant_indices:
        return 0.0
    relevant_set = set(relevant_indices)
    hits = 0
    ap = 0.0
    for rank, idx in enumerate(list(retrieved_indices)[:k], start=1):
        if idx in relevant_set:
            hits += 1
            ap += hits / rank
    return ap / min(len(relevant_indices), k) if hits else 0.0


def mean_recall_at_k(results: list[dict], k_values: list[int] | None = None) -> dict[str, float]:
    if k_values is None:
        k_values = [1, 5, 10]
    metrics: dict[str, float] = {}
    for k in k_values:
        scores = [recall_at_k(r["retrieved"], r["relevant"], k) for r in results]
        metrics[f"recall@{k}"] = sum(scores) / len(scores) if scores else 0.0
    return metrics


def mean_precision_at_k(results: list[dict], k_values: list[int] | None = None) -> dict[str, float]:
    if k_values is None:
        k_values = [5, 10]
    metrics: dict[str, float] = {}
    for k in k_values:
        scores = [precision_at_k(r["retrieved"], r["relevant"], k) for r in results]
        metrics[f"precision@{k}"] = sum(scores) / len(scores) if scores else 0.0
    return metrics


def mean_average_precision(results: list[dict], k: int = 10) -> float:
    scores = [average_precision(r["retrieved"], r["relevant"], k) for r in results]
    return sum(scores) / len(scores) if scores else 0.0
