"""Patch-level evaluation: MaxSim retrieval quality and steering faithfulness.

Mirrors the headline image-level metrics for the patch pivot. Retrieval is the
late-interaction MaxSim (src/retrieval/patch_retrieval.py); ground truth is the
parent-folder label, same single-label proxy as the CLS path."""

import math
from collections import Counter
from pathlib import PurePath

import numpy as np
import torch

from src.retrieval.patch_retrieval import PatchCorpus, l2_normalize, maxsim_search, steer_patches


def image_feature_activation(
    corpus: PatchCorpus, image_id: int, sae, feature_id: int
) -> float:
    """Feature activation of one image = max over its patches (it fires if any region does)."""
    with torch.no_grad():
        acts = sae.encode(torch.from_numpy(corpus.image_patches(image_id))).numpy()
    return float(acts[:, feature_id].max())


def image_activation_vector(corpus: PatchCorpus, image_id: int, sae) -> np.ndarray:
    """Full per-image activation vector = max over the image's patches (hidden,)."""
    with torch.no_grad():
        acts = sae.encode(torch.from_numpy(corpus.image_patches(image_id))).numpy()
    return acts.max(axis=0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho = Pearson on ranks (no scipy dependency)."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def patch_retrieval_results(
    corpus: PatchCorpus,
    index,
    query_corpus: PatchCorpus,
    query_labels: list[str],
    corpus_labels: list[str],
    k: int = 10,
    probe: int = 8,
    in_sample: bool = False,
) -> list[dict]:
    """Per-query {retrieved, relevant} image-id lists for recall_at_k, via MaxSim."""
    label_to_idx: dict[str, list[int]] = {}
    for j, lab in enumerate(corpus_labels):
        label_to_idx.setdefault(lab, []).append(j)

    results = []
    for i in range(query_corpus.n_images):
        _, retrieved = maxsim_search(
            corpus, index, query_corpus.image_patches(i), k=k + (1 if in_sample else 0), probe=probe
        )
        ret = [int(r) for r in retrieved if not (in_sample and int(r) == i)][:k]
        relevant = [j for j in label_to_idx.get(query_labels[i], []) if not (in_sample and j == i)]
        results.append({"retrieved": ret, "relevant": relevant})
    return results


def live_features_from_sample(
    corpus: PatchCorpus, sae, n_features: int, sample_patches: int = 50000, seed: int = 0
) -> list[int]:
    """Random sample of features that fire on a sampled slice of corpus patches."""
    rng = np.random.default_rng(seed)
    n = len(corpus.data)
    idx = np.sort(rng.choice(n, size=min(sample_patches, n), replace=False))
    rows = np.asarray(corpus.data[idx], dtype=np.float32)
    rows /= np.where(np.linalg.norm(rows, axis=1, keepdims=True) > 1e-8,
                     np.linalg.norm(rows, axis=1, keepdims=True), 1.0)
    with torch.no_grad():
        acts = sae.encode(torch.from_numpy(rows)).numpy()
    live = [int(i) for i in np.flatnonzero(acts.max(axis=0) > 0)]
    if not live:
        return []
    return sorted(int(i) for i in rng.choice(live, size=min(n_features, len(live)), replace=False))


def patch_top_k_per_feature(
    corpus: PatchCorpus, sae, feature_ids: list[int], k: int = 20, chunk: int = 8192
) -> dict[int, list[int]]:
    """Global indices of the top-k activating patches per feature, in one streaming pass."""
    fids = np.asarray(feature_ids)
    best_v = np.full((len(fids), k), -np.inf, dtype=np.float32)
    best_i = np.full((len(fids), k), -1, dtype=np.int64)
    n = len(corpus.data)
    with torch.no_grad():
        for s in range(0, n, chunk):
            rows = l2_normalize(np.asarray(corpus.data[s : s + chunk], dtype=np.float32))
            acts = sae.encode(torch.from_numpy(rows)).numpy()[:, fids]  # (chunk, F)
            gidx = np.arange(s, s + rows.shape[0])
            for fi in range(len(fids)):
                cand_v = np.concatenate([best_v[fi], acts[:, fi]])
                cand_i = np.concatenate([best_i[fi], gidx])
                keep = np.argpartition(cand_v, -k)[-k:]
                best_v[fi], best_i[fi] = cand_v[keep], cand_i[keep]
    return {
        int(fids[fi]): best_i[fi][np.argsort(best_v[fi])[::-1]].tolist()
        for fi in range(len(fids))
    }


def _purity_from_labels(labels: list[str], n_total_classes: int | None) -> dict:
    """Class purity of a label list (mirrors evaluation.monosemanticity)."""
    counts = Counter(labels)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-8) for p in probs)
    n_ref = n_total_classes if n_total_classes and n_total_classes > 1 else len(counts)
    max_entropy = math.log(n_ref) if n_ref > 1 else 1.0
    purity = 1.0 if max_entropy <= 0 else 1.0 - entropy / max_entropy
    dominant, dom_count = counts.most_common(1)[0]
    return {
        "purity_score": float(max(0.0, min(1.0, purity))),
        "dominant_class": dominant,
        "dominant_class_fraction": float(dom_count / total),
        "n_classes_in_top_k": len(counts),
        "entropy": entropy,
    }


def patch_monosemanticity(
    corpus: PatchCorpus,
    sae,
    feature_ids: list[int],
    corpus_labels: list[str],
    k: int = 20,
    n_total_classes: int | None = None,
) -> list[dict]:
    """Class purity of each feature's top-k *patches* (labelled by their parent image)."""
    tops = patch_top_k_per_feature(corpus, sae, feature_ids, k=k)
    out = []
    for fid in feature_ids:
        labels = [corpus_labels[corpus.image_ids[p]] for p in tops[fid] if p >= 0]
        if not labels:
            continue
        out.append({"feature_id": fid, **_purity_from_labels(labels, n_total_classes)})
    return out


def patch_isotonicity(
    corpus: PatchCorpus,
    index,
    query_corpus: PatchCorpus,
    sae,
    feature_id: int,
    alphas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    k: int = 10,
    n_queries: int = 30,
    probe: int = 8,
    seed: int = 0,
) -> dict:
    """Spearman rho of (alpha, mean feature activation in the steered MaxSim results).
    1.0 = the slider monotonically increases its own feature."""
    rng = np.random.default_rng(seed)
    qids = rng.choice(query_corpus.n_images, size=min(n_queries, query_corpus.n_images), replace=False)
    directions = sae.feature_directions([feature_id]).cpu().numpy()
    means = []
    for a in alphas:
        per_q = []
        for qi in qids:
            qp = steer_patches(query_corpus.image_patches(int(qi)), directions, [a])
            _, ret = maxsim_search(corpus, index, qp, k=k, probe=probe)
            if len(ret):
                per_q.append(np.mean([image_feature_activation(corpus, int(i), sae, feature_id) for i in ret]))
        means.append(float(np.mean(per_q)) if per_q else 0.0)
    rho = _spearman(np.asarray(alphas), np.asarray(means))
    return {"feature_id": feature_id, "spearman_rho": rho,
            "alpha_means": {str(a): m for a, m in zip(alphas, means)}}


def patch_selectivity(
    corpus: PatchCorpus,
    index,
    query_corpus: PatchCorpus,
    sae,
    feature_id: int,
    alpha: float = 2.0,
    k: int = 10,
    n_queries: int = 30,
    probe: int = 8,
    seed: int = 0,
) -> float:
    """On-target fraction: the steered feature's activation increase in the MaxSim results
    divided by the total positive increase across all features. ~1.0 = selective slider."""
    rng = np.random.default_rng(seed)
    qids = rng.choice(query_corpus.n_images, size=min(n_queries, query_corpus.n_images), replace=False)
    directions = sae.feature_directions([feature_id]).cpu().numpy()
    deltas = np.zeros(sae.hidden_dim, dtype=np.float64)
    n = 0
    for qi in qids:
        qp = query_corpus.image_patches(int(qi))
        _, ret_b = maxsim_search(corpus, index, qp, k=k, probe=probe)
        _, ret_s = maxsim_search(corpus, index, steer_patches(qp, directions, [alpha]), k=k, probe=probe)
        if not len(ret_b) or not len(ret_s):
            continue
        base = np.mean([image_activation_vector(corpus, int(i), sae) for i in ret_b], axis=0)
        steer = np.mean([image_activation_vector(corpus, int(i), sae) for i in ret_s], axis=0)
        deltas += steer - base
        n += 1
    if n == 0:
        return float("nan")
    deltas /= n
    total_positive = float(np.clip(deltas, 0.0, None).sum())
    if total_positive < 1e-12:
        return float("nan")
    return max(float(deltas[feature_id]), 0.0) / total_positive


def patch_steering_faithfulness(
    corpus: PatchCorpus,
    index,
    query_corpus: PatchCorpus,
    sae,
    feature_id: int,
    alpha: float = 2.0,
    k: int = 10,
    n_queries: int = 50,
    probe: int = 8,
    seed: int = 0,
) -> float:
    """Ratio of the feature's mean activation in steered vs unsteered MaxSim results.
    >1 = steering the slider pulls retrieval toward the concept."""
    rng = np.random.default_rng(seed)
    n = min(n_queries, query_corpus.n_images)
    qids = rng.choice(query_corpus.n_images, size=n, replace=False)
    directions = sae.feature_directions([feature_id]).cpu().numpy()

    base, steered = [], []
    for qi in qids:
        qp = query_corpus.image_patches(int(qi))
        _, ret_b = maxsim_search(corpus, index, qp, k=k, probe=probe)
        _, ret_s = maxsim_search(corpus, index, steer_patches(qp, directions, [alpha]), k=k, probe=probe)
        if len(ret_b):
            base.append(np.mean([image_feature_activation(corpus, int(i), sae, feature_id) for i in ret_b]))
        if len(ret_s):
            steered.append(np.mean([image_feature_activation(corpus, int(i), sae, feature_id) for i in ret_s]))

    mb = float(np.mean(base)) if base else 0.0
    ms = float(np.mean(steered)) if steered else 0.0
    return ms / mb if mb > 1e-8 else float("nan")
