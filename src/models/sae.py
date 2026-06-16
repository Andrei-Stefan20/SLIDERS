import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _meta_path(checkpoint: Path | str) -> Path:
    p = Path(checkpoint)
    return p.parent / f"{p.stem}.meta.json"


class SparseAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 8192,
        tied_weights: bool = False,
        topk: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tied_weights = tied_weights
        self.topk = topk

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        if not tied_weights:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        else:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        if not self.tied_weights:
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.t())
                self.decoder.weight.div_(
                    self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
                )
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x)
        if self.topk > 0:
            h = torch.zeros_like(pre)
            topk_vals, topk_idx = pre.topk(self.topk, dim=1)
            h.scatter_(1, topk_idx, F.relu(topk_vals))
            return h
        return F.relu(pre)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        if self.tied_weights:
            return h @ self.encoder.weight + self.decoder_bias
        return self.decoder(h)

    def feature_directions(self, feature_ids=None) -> torch.Tensor:
        """Concept directions in input space: the decoder columns (dictionary atoms
        the slider adds to the query). For tied weights the atom is the encoder row.
        Rows are unit-norm so alpha scales the same across features. See docs/adr/0002."""
        if self.tied_weights:
            dirs = self.encoder.weight.detach()
        else:
            dirs = self.decoder.weight.detach().t()  # (hidden_dim, input_dim)
        if feature_ids is not None:
            dirs = dirs[feature_ids]
        return F.normalize(dirs, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.decode(h), h

    def save_meta(self, checkpoint: Path | str) -> None:
        _meta_path(checkpoint).write_text(
            json.dumps({"topk": self.topk, "tied_weights": self.tied_weights})
        )

    @classmethod
    def load(cls, checkpoint: Path | str) -> "SparseAutoencoder":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        tied = "decoder.weight" not in state
        hidden_dim, input_dim = state["encoder.weight"].shape
        meta = _meta_path(checkpoint)
        topk = json.loads(meta.read_text()).get("topk", 0) if meta.exists() else 0
        sae = cls(input_dim=input_dim, hidden_dim=hidden_dim, tied_weights=tied, topk=topk)
        sae.load_state_dict(state)
        sae.eval()
        return sae
