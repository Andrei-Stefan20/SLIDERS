# ruff: noqa: E402
"""Generate figures + data tables for the report, one folder per pipeline stage.

Reads the artifacts already on disk (embeddings, SAE activations, class directions),
plus the optional history.json (from train_sae) and metrics JSON (from
`evaluate.py --output`). Each stage is independent and skipped with a note if its
inputs are missing, so it runs on a partial pipeline.

    python scripts/make_report.py --dataset plantvillage_train \
        --eval-json reports/plantvillage_train_eval.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.io import normalize_embeddings

CHUNK = 4096


def _label(path: str) -> str:
    return Path(path).parent.name


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# --------------------------------------------------------------------------- [0]
def stage_embeddings(out: Path, embs: np.ndarray, paths: list[str], seed: int,
                     max_points: int) -> None:
    print("\n[0] Embeddings")
    labels = [_label(p) for p in paths]
    classes = sorted(set(labels))
    cls_to_int = {c: i for i, c in enumerate(classes)}
    counts = {c: labels.count(c) for c in classes}

    rows = sorted(counts.items(), key=lambda x: -x[1])
    _write_csv(out / "class_counts.csv", ["class", "count"], [[c, n] for c, n in rows])

    fig, ax = plt.subplots(figsize=(10, max(3, len(classes) * 0.22)))
    ax.barh([c for c, _ in rows][::-1], [n for _, n in rows][::-1], color="#4c72b0")
    ax.set_xlabel("images")
    ax.set_title(f"Class distribution ({len(classes)} classes, {len(paths)} images)")
    _save(fig, out / "class_distribution.png")

    # 2D PCA projection via SVD on a subsample, colored by class.
    rng = np.random.default_rng(seed)
    n = min(max_points, len(embs))
    idx = rng.choice(len(embs), size=n, replace=False)
    sub = embs[idx]
    centered = sub - sub.mean(0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    var_ratio = (s[:2] ** 2) / (s ** 2).sum()
    color_ints = np.array([cls_to_int[labels[i]] for i in idx])

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_ints, cmap="tab20", s=4, alpha=0.5)
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)")
    ax.set_title(f"DINOv2 embeddings, PCA 2D (n={n}, colored by class)")
    fig.colorbar(sc, ax=ax, label="class index")
    _save(fig, out / "embeddings_pca2d.png")

    stats = {
        "n_images": len(embs), "dim": int(embs.shape[1]), "n_classes": len(classes),
        "norm_mean": float(np.linalg.norm(embs, axis=1).mean()),
        "pc1_var_ratio": float(var_ratio[0]), "pc2_var_ratio": float(var_ratio[1]),
    }
    (out / "embedding_stats.json").write_text(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- [1]
def _activation_stats(acts: np.ndarray) -> dict:
    n, h = acts.shape
    l0 = np.empty(n, dtype=np.int32)
    nonzero = np.zeros(h, dtype=np.int64)
    fsum = np.zeros(h, dtype=np.float64)
    fsumsq = np.zeros(h, dtype=np.float64)
    for start in range(0, n, CHUNK):
        c = np.asarray(acts[start:start + CHUNK], dtype=np.float32)
        active = c > 0
        l0[start:start + len(c)] = active.sum(1)
        nonzero += active.sum(0)
        fsum += c.sum(0)
        fsumsq += (c.astype(np.float64) ** 2).sum(0)
    mean = fsum / n
    var = np.maximum(fsumsq / n - mean ** 2, 0.0)
    return {"l0": l0, "firing_freq": nonzero / n, "var": var, "dead": nonzero == 0}


def stage_training(out: Path, history_path: Path, acts: np.ndarray | None) -> None:
    print("\n[1] SAE training")
    if history_path.exists():
        hist = json.loads(history_path.read_text())
        ep = [r["epoch"] for r in hist]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(ep, [r["train_loss"] for r in hist], label="train")
        axes[0].plot(ep, [r["val_recon"] for r in hist], label="val recon")
        axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
        axes[1].plot(ep, [r["val_l0"] for r in hist], color="#c44e52")
        axes[1].axhspan(20, 80, color="green", alpha=0.1, label="interpretable band")
        axes[1].set_title("val L0 (active features/sample)"); axes[1].set_xlabel("epoch"); axes[1].legend()
        axes[2].plot(ep, [r["dead_frac"] for r in hist], color="#8172b3")
        axes[2].set_title("Dead feature fraction"); axes[2].set_xlabel("epoch")
        _save(fig, out / "training_curves.png")
        _write_csv(out / "training_history.csv",
                   ["epoch", "train_loss", "val_recon", "val_score", "val_l0", "dead_frac"],
                   [[r["epoch"], r["train_loss"], r["val_recon"], r["val_score"],
                     r["val_l0"], r["dead_frac"]] for r in hist])
    else:
        print(f"  SKIP curves: no history at {history_path} "
              f"(retrain with the instrumented train_sae.py to get it)")

    if acts is None:
        print("  SKIP sparsity: no activations on disk")
        return
    st = _activation_stats(acts)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(st["l0"], bins=50, color="#4c72b0")
    axes[0].set_title(f"Active features per image (mean L0={st['l0'].mean():.1f})")
    axes[0].set_xlabel("active features")
    freq = st["firing_freq"][st["firing_freq"] > 0]
    axes[1].hist(np.log10(freq), bins=50, color="#55a868")
    axes[1].set_title(f"Feature firing frequency ({int(st['dead'].sum())} dead)")
    axes[1].set_xlabel("log10(fraction of images a feature fires on)")
    _save(fig, out / "sparsity.png")

    (out / "sparsity_stats.json").write_text(json.dumps({
        "n_features": int(acts.shape[1]), "dead_features": int(st["dead"].sum()),
        "mean_l0": float(st["l0"].mean()), "median_l0": float(np.median(st["l0"])),
    }, indent=2))


# --------------------------------------------------------------------------- [2]
def _montage(paths: list[str], acts: list[float], thumb: int = 110) -> Image.Image:
    cols = len(paths)
    grid = Image.new("RGB", (cols * thumb, thumb), "white")
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB").resize((thumb, thumb))
        except Exception:
            img = Image.new("RGB", (thumb, thumb), "gray")
        grid.paste(img, (i * thumb, 0))
    return grid


def stage_features(out: Path, acts: np.ndarray, paths: list[str], names: dict | None,
                   n_features: int, montage_k: int) -> None:
    print("\n[2] Feature naming / top images")
    var = _activation_stats(acts)["var"]
    top_feats = [int(i) for i in np.argsort(var)[::-1] if var[i] > 0][:n_features]
    cols = np.asarray(acts[:, top_feats], dtype=np.float32)  # N x n_features

    table = []
    for j, fid in enumerate(top_feats):
        col = cols[:, j]
        order = np.argsort(col)[::-1][:montage_k]
        tpaths = [paths[i] for i in order]
        tacts = [float(col[i]) for i in order]
        name = (names or {}).get(str(fid), "")
        _montage(tpaths, tacts).save(out / f"feature_{fid:05d}_top.png")
        table.append([fid, name, _label(tpaths[0]), f"{tacts[0]:.3f}", f"{float(var[fid]):.4f}"])
    print(f"  wrote {len(top_feats)} montages")
    _write_csv(out / "top_features.csv",
               ["feature_id", "name", "top1_class", "top1_activation", "variance"], table)
    if names is None:
        print("  (no feature_names.json -> montages unlabeled; run name_features.py)")


# --------------------------------------------------------------------------- [3]
def stage_class_directions(out: Path, dirs_path: Path, names_path: Path) -> None:
    print("\n[3] Class directions")
    if not dirs_path.exists():
        print(f"  SKIP: no {dirs_path}")
        return
    dirs = np.load(dirs_path).astype(np.float32)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    sim = dirs @ dirs.T
    names = json.loads(names_path.read_text()) if names_path.exists() else None
    labels = names if (names and len(names) == len(dirs)) else [str(i) for i in range(len(dirs))]

    fig, ax = plt.subplots(figsize=(max(6, len(dirs) * 0.35), max(5, len(dirs) * 0.32)))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(dirs))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(dirs))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Class direction cosine similarity")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out / "class_direction_similarity.png")
    _write_csv(out / "class_direction_similarity.csv",
               ["", *labels], [[labels[i], *[f"{v:.3f}" for v in sim[i]]] for i in range(len(dirs))])


# --------------------------------------------------------------------------- [5]
def _dist_fig(values: list[float], title: str, xlabel: str, ref: float | None,
              ref_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(30, max(5, len(values))), color="#4c72b0", alpha=0.85)
    if ref is not None:
        ax.axvline(ref, color="red", linestyle="--", label=ref_label)
        ax.legend()
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("features")
    return fig


def stage_evaluation(out: Path, eval_json: Path) -> None:
    print("\n[5] Evaluation")
    if eval_json is None or not eval_json.exists():
        print("  SKIP: no metrics JSON (pass --eval-json from `evaluate.py ... --output`)")
        return
    r = json.loads(eval_json.read_text())

    if "retrieval" in r:
        rec, prec = r["retrieval"]["recall"], r["retrieval"]["precision"]
        ks = r.get("meta", {}).get("recall_ks", [1, 5, 10])
        x = np.arange(len(ks)); w = 0.38
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - w / 2, [rec[f"recall@{k}"] for k in ks], w, label="recall@k")
        ax.bar(x + w / 2, [prec[f"precision@{k}"] for k in ks], w, label="precision@k")
        ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_title("Retrieval (held-out)" if r.get("meta", {}).get("held_out") else "Retrieval (in-sample)")
        ax.legend()
        _save(fig, out / "recall_precision.png")

    if "monosemanticity" in r:
        m = r["monosemanticity"]
        vals = [f["purity_score"] for f in m["per_feature"]]
        fig = _dist_fig(vals, "Monosemanticity purity vs null", "purity",
                        m.get("null_mean"), f"null mean={m.get('null_mean', 0):.3f}")
        _save(fig, out / "monosemanticity.png")

    if "faithfulness" in r:
        vals = list(r["faithfulness"]["per_feature"].values())
        _save(_dist_fig(vals, "Steering faithfulness", "steered/unsteered activation ratio",
                        1.0, "no effect (1.0)"), out / "faithfulness.png")

    if "isotonicity" in r:
        vals = [f["spearman_rho"] for f in r["isotonicity"]["per_feature"]]
        _save(_dist_fig(vals, "Steering isotonicity", "Spearman rho",
                        0.7, "controllable (0.7)"), out / "isotonicity.png")

    if "targeted_recall" in r:
        vals = [f["delta_recall"] for f in r["targeted_recall"]["per_feature"]]
        _save(_dist_fig(vals, "Targeted class delta-recall", "steered - unsteered recall",
                        0.0, "no effect (0)"), out / "targeted_recall.png")

    if "ablation" in r:
        ab = r["ablation"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ["SAE", "PCA"]
        ax.bar(bars, [ab["sae_median_faithfulness"], ab["pca_median_faithfulness"]],
               color=["#55a868", "#c44e52"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Median steering lift (SAE advantage={ab['steering_advantage']:+.4f})")
        ax.set_ylabel("cosine lift (steered - unsteered)")
        _save(fig, out / "sae_vs_pca.png")

    # flat summary table of every distribution metric
    rows = []
    for key in ("faithfulness", "isotonicity", "targeted_recall", "monosemanticity",
                "clip_alignment", "direction_steering"):
        s = r.get(key, {}).get("summary")
        if s:
            rows.append([key, f"{s['mean']:.4f}", f"{s['median']:.4f}",
                         f"{s['ci_low']:.4f}", f"{s['ci_high']:.4f}", s["n"]])
    if rows:
        _write_csv(out / "metric_summaries.csv",
                   ["metric", "mean", "median", "ci_low", "ci_high", "n"], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate report figures per pipeline stage.")
    ap.add_argument("--dataset", default="plantvillage_train", help="artifact prefix")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--eval-json", type=Path, default=None, help="metrics JSON from evaluate.py --output")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--n-feature-montages", type=int, default=12)
    ap.add_argument("--montage-k", type=int, default=8)
    ap.add_argument("--max-projection-points", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d, m, ds = args.data_dir, args.models_dir, args.dataset
    out_root = args.output_dir or Path("reports") / ds
    stages = {n: out_root / n for n in
              ("00_embeddings", "01_training", "02_features", "03_class_directions", "05_evaluation")}
    for p in stages.values():
        p.mkdir(parents=True, exist_ok=True)

    paths = json.loads((d / f"{ds}_image_paths.json").read_text())
    embs = normalize_embeddings(np.load(d / f"{ds}_embeddings.npy").astype(np.float32))

    acts_path = d / f"{ds}_activations.npy"
    acts = np.load(acts_path, mmap_mode="r") if acts_path.exists() else None

    names_path = m / f"{ds}_feature_names.json"
    names = json.loads(names_path.read_text()) if names_path.exists() else None

    stage_embeddings(stages["00_embeddings"], embs, paths, args.seed, args.max_projection_points)
    stage_training(stages["01_training"], m / f"{ds}_sae_history.json", acts)
    if acts is not None:
        stage_features(stages["02_features"], acts, paths, names,
                       args.n_feature_montages, args.montage_k)
    else:
        print("\n[2] SKIP features: no activations on disk (run build_index.py --sae-model)")
    stage_class_directions(stages["03_class_directions"],
                           d / f"{ds}_class_directions.npy",
                           d / f"{ds}_class_direction_names.json")
    stage_evaluation(stages["05_evaluation"], args.eval_json)

    print(f"\nReport written under {out_root}")


if __name__ == "__main__":
    main()
