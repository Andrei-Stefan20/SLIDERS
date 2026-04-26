"""CLI script for training a Sparse Autoencoder on pre-extracted embeddings.

Usage:
    python scripts/train_sae.py --embeddings data/processed/plantvillage_embeddings.npy \\
        --output models/ --config configs/plantvillage.yaml
"""

# ruff: noqa: E402

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SAEConfig
from src.models.train_sae import train_sae


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder on DINOv2 embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to the .npy embeddings file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file (overrides CLI defaults).",
    )
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--lambda-sparsity", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--tied-weights", action="store_true")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data held out for validation (default: 0.1).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience in epochs (default: 10).",
    )
    parser.add_argument(
        "--dead-threshold-steps",
        type=int,
        default=1000,
        help="Steps of inactivity before a dead feature is reinitialised (default: 1000).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Use exact TopK activation instead of ReLU+L1. "
             "0 = disabled (use ReLU+L1). Good starting value: hidden_dim // 200.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="mse",
        choices=["mse", "cosine"],
        help="Reconstruction loss. 'cosine' aligns training with the "
             "inner-product retrieval geometry on the unit hypersphere.",
    )
    args = parser.parse_args()

    cfg: dict = {}
    if args.config is not None:
        yaml = cast(Any, importlib.import_module("yaml"))
        cfg = (yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get(
            "sae", {}
        )

    cli_cfg = {
        "hidden_dim": args.hidden_dim,
        "lambda_sparsity": args.lambda_sparsity,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "topk": args.topk,
        "loss_type": args.loss_type,
        "val_split": args.val_split,
        "patience": args.patience,
        "dead_threshold_steps": args.dead_threshold_steps,
    }
    sae_cfg = SAEConfig(**(cli_cfg | cfg))

    hidden_dim = sae_cfg.hidden_dim
    lambda_sparsity = sae_cfg.lambda_sparsity
    lr = sae_cfg.lr
    epochs = sae_cfg.epochs
    batch_size = sae_cfg.batch_size
    topk = sae_cfg.topk
    loss_type = sae_cfg.loss_type
    val_split = sae_cfg.val_split
    patience = sae_cfg.patience
    dead_threshold_steps = sae_cfg.dead_threshold_steps

    from src.utils.logging import setup_logging
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Training SAE")
    logger.info(f"  embeddings           : {args.embeddings}")
    logger.info(f"  hidden_dim           : {hidden_dim}")
    logger.info(f"  lambda_sparsity      : {lambda_sparsity if topk == 0 else 'N/A (topk mode)'}")
    logger.info(f"  topk                 : {topk if topk > 0 else 'disabled (ReLU+L1)'}")
    logger.info(f"  loss_type            : {loss_type}")
    logger.info(f"  lr                   : {lr}")
    logger.info(f"  epochs               : {epochs}")
    logger.info(f"  batch_size           : {batch_size}")
    logger.info(f"  val_split            : {val_split}")
    logger.info(f"  patience             : {patience}")
    logger.info(f"  dead_threshold_steps : {dead_threshold_steps}")

    train_sae(
        embeddings_path=args.embeddings,
        output_dir=args.output,
        hidden_dim=hidden_dim,
        lambda_sparsity=lambda_sparsity,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        log_every=args.log_every,
        tied_weights=args.tied_weights,
        topk=topk,
        loss_type=loss_type,
        val_split=val_split,
        patience=patience,
        dead_threshold_steps=dead_threshold_steps,
    )


if __name__ == "__main__":
    main()
