# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import get_adapter
from src.datasets.base import DatasetAdapter
from src.utils.io import dataset_stem, normalize_embeddings

MIN_SAMPLES = 5


def compute_reference_directions(
    adapter: DatasetAdapter,
    embeddings: np.ndarray,
    image_paths: list[str],
) -> tuple[np.ndarray, list[str]]:
    cat_sub: dict[tuple[str, str], list[int]] = {}
    cat_ref: dict[str, list[int]] = {}

    for i, path in enumerate(image_paths):
        cat, sub, is_ref = adapter.parse_path(path)
        if is_ref:
            cat_ref.setdefault(cat, []).append(i)
        else:
            cat_sub.setdefault((cat, sub), []).append(i)

    directions, names = [], []

    for (cat, sub), idxs in sorted(cat_sub.items()):
        if len(idxs) < MIN_SAMPLES:
            continue
        ref_idxs = cat_ref.get(cat, [])
        if len(ref_idxs) < MIN_SAMPLES:
            ref_idxs = [i for lst in cat_ref.values() for i in lst]
        if len(ref_idxs) < MIN_SAMPLES:
            continue

        direction = embeddings[idxs].mean(0) - embeddings[ref_idxs].mean(0)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue

        name = sub.replace("_", " ").strip()
        directions.append((direction / norm).astype(np.float32))
        names.append(name)
        print(f"  [{cat:20s}] {name:40s}  ({len(idxs)} / {len(ref_idxs)} ref)")

    return np.stack(directions), names


def compute_global_directions(
    adapter: DatasetAdapter,
    embeddings: np.ndarray,
    image_paths: list[str],
) -> tuple[np.ndarray, list[str]]:
    class_indices: dict[str, list[int]] = {}
    for i, path in enumerate(image_paths):
        _, sub, _ = adapter.parse_path(path)
        class_indices.setdefault(sub, []).append(i)

    global_mean = embeddings.mean(0)
    directions, names = [], []

    for sub, idxs in sorted(class_indices.items()):
        if len(idxs) < MIN_SAMPLES:
            continue
        direction = embeddings[idxs].mean(0) - global_mean
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        name = sub.replace("_", " ").strip()
        directions.append((direction / norm).astype(np.float32))
        names.append(name)
        print(f"  {name:45s}  ({len(idxs)} images)")

    return np.stack(directions), names


def compute_one_vs_rest_directions(
    adapter: DatasetAdapter,
    embeddings: np.ndarray,
    image_paths: list[str],
) -> tuple[np.ndarray, list[str]]:
    class_indices: dict[str, list[int]] = {}
    for i, path in enumerate(image_paths):
        _, sub, _ = adapter.parse_path(path)
        class_indices.setdefault(sub, []).append(i)

    all_idx_set = set(range(len(image_paths)))
    directions, names = [], []

    for sub, idxs in sorted(class_indices.items()):
        if len(idxs) < MIN_SAMPLES:
            continue
        rest = list(all_idx_set - set(idxs))
        if len(rest) < MIN_SAMPLES:
            continue
        direction = embeddings[idxs].mean(0) - embeddings[rest].mean(0)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        name = sub.replace("_", " ").strip()
        directions.append((direction / norm).astype(np.float32))
        names.append(name)
        print(f"  {name:45s}  ({len(idxs)} vs {len(rest)})")

    return np.stack(directions), names


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute class directions.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--image-paths", type=Path, required=True)
    parser.add_argument("--adapter", type=str, default="generic")
    parser.add_argument("--output", type=Path, default=Path("data/processed/"))
    parser.add_argument("--mode", type=str, default=None,
                        help="Override adapter direction_mode: reference|global|one_vs_rest")
    args = parser.parse_args()

    adapter = get_adapter(args.adapter)
    mode = args.mode or adapter.direction_mode

    embeddings = normalize_embeddings(np.load(args.embeddings).astype(np.float32))
    image_paths = json.loads(args.image_paths.read_text())

    print(f"Computing class directions (mode={mode}, adapter={args.adapter})...")

    fn = {
        "reference": compute_reference_directions,
        "global": compute_global_directions,
        "one_vs_rest": compute_one_vs_rest_directions,
    }.get(mode, compute_global_directions)

    directions, names = fn(adapter, embeddings, image_paths)

    print(f"\nFound {len(names)} directions.")

    stem = dataset_stem(args.embeddings)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / f"{stem}_class_directions.npy", directions)
    (args.output / f"{stem}_class_direction_names.json").write_text(
        json.dumps(names, indent=2, ensure_ascii=False)
    )
    print(f"Saved -> {args.output}/{stem}_class_directions.npy")
    print(f"Saved -> {args.output}/{stem}_class_direction_names.json")


if __name__ == "__main__":
    main()
