"""SAE training loop."""

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.loader import EmbeddingDataset
from src.models.losses import cosine_reconstruction_loss, reconstruction_loss, sparsity_loss
from src.models.sae import SparseAutoencoder
from src.utils.device import get_device
from src.utils.logging import get_logger


def train_sae(
    embeddings_path: Path,
    output_dir: Path,
    hidden_dim: int = 8192,
    lambda_sparsity: float = 1e-3,
    lr: float = 3e-4,
    epochs: int = 50,
    batch_size: int = 512,
    log_every: int = 100,
    tied_weights: bool = False,
    topk: int = 0,
    loss_type: str = "mse",
    val_split: float = 0.1,
    patience: int = 10,
    dead_threshold_steps: int = 1000,
) -> None:
    logger = get_logger(__name__)
    device = get_device()
    logger.info(
        f"Training SAE | device={device} | loss={loss_type} | "
        f"activation={'topk-' + str(topk) if topk > 0 else 'relu+L1'}"
    )

    dataset = EmbeddingDataset(embeddings_path)
    n_val = max(1, int(len(dataset) * val_split))
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=rng).tolist()

    train_loader = DataLoader(
        Subset(dataset, indices[n_val:]), batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        Subset(dataset, indices[:n_val]), batch_size=batch_size, shuffle=False, drop_last=False
    )
    logger.info(f"{len(dataset)} samples | {len(dataset) - n_val} train | {n_val} val")

    input_dim = dataset[0].shape[0]
    sae = SparseAutoencoder(
        input_dim=input_dim, hidden_dim=hidden_dim, tied_weights=tied_weights, topk=topk
    ).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "sae_best.pt"

    # tracks how many steps each hidden unit has been inactive
    steps_since_active = torch.zeros(hidden_dim, dtype=torch.long, device=device)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    step = 0

    recon_fn = cosine_reconstruction_loss if loss_type == "cosine" else reconstruction_loss

    for epoch in range(1, epochs + 1):
        sae.train()
        epoch_losses: list[float] = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch.to(device)
            x_hat, h = sae(x)

            recon = recon_fn(x, x_hat)
            if topk > 0:
                # in topk mode there is no sparsity penalty, K features are always active
                loss = recon
                sparse_display = (h > 0).float().sum(dim=1).mean()
            else:
                sparse = sparsity_loss(h)
                loss = recon + lambda_sparsity * sparse
                sparse_display = sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep decoder columns unit-norm so steering directions are comparable
            if not tied_weights:
                with torch.no_grad():
                    F.normalize(sae.decoder.weight.data, dim=0, out=sae.decoder.weight.data)

            with torch.no_grad():
                fired = (h > 0).any(dim=0)
                steps_since_active[fired] = 0
                steps_since_active[~fired] += 1

                # revive dead neurons by reinitializing from current residuals
                dead_mask = steps_since_active > dead_threshold_steps
                n_dead = int(dead_mask.sum().item())
                if n_dead > 0:
                    residual = (x - x_hat).detach()
                    rand_idx = torch.randint(0, residual.shape[0], (n_dead,), device=device)
                    new_dirs = F.normalize(residual[rand_idx], dim=1)
                    sae.encoder.weight.data[dead_mask] = new_dirs
                    sae.encoder.bias.data[dead_mask] = 0.0
                    if not tied_weights:
                        sae.decoder.weight.data[:, dead_mask] = new_dirs.T
                    steps_since_active[dead_mask] = 0

            epoch_losses.append(loss.item())
            step += 1

            if step % log_every == 0:
                dead_ratio = float((steps_since_active > dead_threshold_steps).float().mean().item())
                logger.info(
                    f"step {step:6d} | loss={loss.item():.4f} recon={recon.item():.4f} "
                    f"{'L0' if topk > 0 else 'sparse'}={sparse_display.item():.4f} dead={dead_ratio:.1%}"
                )

        # validation uses reconstruction only, no sparsity term
        sae.eval()
        val_total_loss = 0.0
        val_l0_total = 0.0
        val_total_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                x_hat, h = sae(x)
                n = x.shape[0]
                val_total_loss += recon_fn(x, x_hat).item() * n
                val_l0_total += float((h > 0).float().sum().item())
                val_total_n += n

        val_loss = val_total_loss / val_total_n if val_total_n > 0 else float("inf")
        val_l0 = val_l0_total / val_total_n if val_total_n > 0 else 0.0
        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        dead_final = float((steps_since_active > dead_threshold_steps).float().mean().item())

        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{epochs} | train={train_loss:.4f} val={val_loss:.4f} "
            f"val_L0={val_l0:.1f} dead={dead_final:.1%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(sae.state_dict(), best_path)
            logger.info(f"  Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)."
                )
                break

    torch.save(sae.state_dict(), output_dir / "sae_last.pt")
    logger.info(f"Done. Best val loss: {best_val_loss:.4f}, saved to {best_path}")
