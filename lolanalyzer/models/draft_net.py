"""Red neuronal de draft: valora una composicion y estima P(gana el equipo 1)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DraftNet(nn.Module):
    """Set-transformer sobre picks y bans.

    Cada token es (campeon, rol, lado, tipo). Los picks desconocidos usan el
    indice UNK, de modo que el modelo tambien funciona con drafts a medias.
    """

    def __init__(
        self,
        n_champions: int,
        n_roles: int = 5,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_champions = n_champions
        self.unk_index = n_champions + 1

        self.champ_emb = nn.Embedding(n_champions + 2, d_model, padding_idx=0)
        self.role_emb = nn.Embedding(n_roles + 1, d_model, padding_idx=0)
        self.side_emb = nn.Embedding(2, d_model)
        self.kind_emb = nn.Embedding(3, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        # Cabeza sobre representaciones de ambos equipos y sus interacciones.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module.padding_idx is None:
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(
        self,
        champ: torch.Tensor,
        role: torch.Tensor,
        side: torch.Tensor,
        kind: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.champ_emb(champ)
            + self.role_emb(role)
            + self.side_emb(side)
            + self.kind_emb(kind)
        )
        x = self.norm(x)
        h = self.encoder(x, src_key_padding_mask=~mask)

        pick = (kind == 1) & mask
        t1 = pick & (side == 0)
        t2 = pick & (side == 1)

        def pool(select: torch.Tensor) -> torch.Tensor:
            weight = select.to(h.dtype).unsqueeze(-1)
            total = weight.sum(dim=1).clamp(min=1.0)
            return (h * weight).sum(dim=1) / total

        a = pool(t1)
        b = pool(t2)
        features = torch.cat([a, b, a - b, a * b], dim=-1)
        return self.head(features).squeeze(-1)


def config_from_checkpoint(ckpt: dict) -> dict:
    return {
        "n_champions": int(ckpt["n_champions"]),
        "d_model": int(ckpt.get("d_model", 96)),
        "n_heads": int(ckpt.get("n_heads", 4)),
        "n_layers": int(ckpt.get("n_layers", 3)),
        "dropout": float(ckpt.get("dropout", 0.15)),
    }


def load_draft_net(path, device: str = "cpu") -> tuple[DraftNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DraftNet(**config_from_checkpoint(ckpt))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
