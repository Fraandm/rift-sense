"""Red neuronal de builds: predice el siguiente objeto a comprar."""
from __future__ import annotations

import torch
import torch.nn as nn


class BuildNet(nn.Module):
    """Codifica contexto (campeon, rol, aliados, rivales, slot) + historial de compras."""

    def __init__(
        self,
        n_champions: int,
        n_items: int,
        n_roles: int = 5,
        d_model: int = 96,
        hidden: int = 192,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_items = n_items
        self.champ_emb = nn.Embedding(n_champions + 2, d_model, padding_idx=0)
        self.role_emb = nn.Embedding(n_roles + 1, d_model)
        self.side_emb = nn.Embedding(2, d_model)
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)

        self.gru = nn.GRU(d_model, hidden, batch_first=True)
        ctx_dim = d_model * 5 + 1
        self.ctx = nn.Sequential(
            nn.Linear(ctx_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_items + 1),
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
        ally: torch.Tensor,
        enemy: torch.Tensor,
        items: torch.Tensor,
        slot: torch.Tensor,
    ) -> torch.Tensor:
        """items: (B, 7) ids en orden de compra; slot: (B,) numero de objetos ya comprados."""
        champ_vec = self.champ_emb(champ)
        role_vec = self.role_emb(role)
        side_vec = self.side_emb(side)
        ally_vec = self.champ_emb(ally).mean(dim=1)
        enemy_vec = self.champ_emb(enemy).mean(dim=1)
        slot_norm = (slot.clamp(0, 6).float() / 6.0).unsqueeze(-1)
        ctx = self.ctx(torch.cat([champ_vec, role_vec, side_vec, ally_vec, enemy_vec, slot_norm], dim=-1))

        seq = self.item_emb(items)
        out, _ = self.gru(seq)
        gather_idx = (slot.clamp(0, 6) - 1).clamp(min=0)
        has_history = (slot > 0).float().unsqueeze(-1)
        last = out[torch.arange(out.shape[0], device=out.device), gather_idx] * has_history
        return self.head(torch.cat([ctx, last], dim=-1))


def load_build_net(path, device: str = "cpu") -> tuple[BuildNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BuildNet(
        n_champions=int(ckpt["n_champions"]),
        n_items=int(ckpt["n_items"]),
        d_model=int(ckpt.get("d_model", 96)),
        hidden=int(ckpt.get("hidden", 192)),
        dropout=float(ckpt.get("dropout", 0.15)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
