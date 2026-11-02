import torch
import torch.nn as nn
from typing import List


PROPERTY_KEYS = [
    "formation_energy_per_atom",
    "band_gap",
]


class ScalarEmbedding(nn.Module):


    def __init__(self, hidden_dim: int, clip_val: float = 4.0):
        super().__init__()
        self.clip_val = clip_val

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )


        self.null_vec = nn.Parameter(torch.zeros(hidden_dim))

        self._init_mlp()

    def _init_mlp(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        value:   torch.Tensor,
        missing: torch.Tensor,
    ) -> torch.Tensor:
        B = value.shape[0]

        value_safe = value.clamp(-self.clip_val, self.clip_val)
        real_emb   = self.mlp(value_safe.view(B, 1).float())
        null_emb   = self.null_vec.unsqueeze(0).expand(B, -1)
        m = missing.view(B, 1).float()
        return real_emb * (1.0 - m) + null_emb * m


class PropertyEmbeddings(nn.Module):


    def __init__(self, hidden_dim: int = 128, clip_val: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.total_dim  = len(PROPERTY_KEYS) * hidden_dim

        self.embeddings = nn.ModuleDict({
            key: ScalarEmbedding(hidden_dim, clip_val=clip_val)
            for key in PROPERTY_KEYS
        })

    def forward(self, batch, p_uncond: float = 0.0) -> torch.Tensor:


        B      = (batch.num_graphs if hasattr(batch, 'num_graphs')
                  else int(batch.num_atoms.shape[0]))
        device = batch.num_atoms.device
        parts  = []

        for key in PROPERTY_KEYS:
            mod       = self.embeddings[key]
            val_attr  = getattr(batch, key, None)
            miss_attr = getattr(batch, f'{key}_missing', None)

            if val_attr is None:

                val     = torch.zeros(B, 1, device=device)
                missing = torch.ones(B, 1, device=device)
            else:
                val     = val_attr.view(B, 1).float().to(device)
                missing = (miss_attr.view(B, 1).float().to(device)
                           if miss_attr is not None
                           else torch.zeros(B, 1, device=device))

            parts.append(mod(val, missing))

        cond = torch.cat(parts, dim=-1)


        if p_uncond > 0.0 and self.training:
            drop_mask = (torch.rand(B, device=device) < p_uncond)
            if drop_mask.any():
                null_cond = self.get_uncond_embedding(B, device)
                cond = torch.where(
                    drop_mask.view(B, 1).expand_as(cond),
                    null_cond,
                    cond,
                )

        return cond

    def get_uncond_embedding(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> torch.Tensor:


        parts = []
        for key in PROPERTY_KEYS:
            null_v = self.embeddings[key].null_vec
            parts.append(null_v.unsqueeze(0).expand(batch_size, -1))
        return torch.cat(parts, dim=-1)

    def get_partial_embedding(
        self,
        batch,
        active_keys: List[str],
    ) -> torch.Tensor:


        B      = batch.num_graphs
        device = batch.num_atoms.device
        parts  = []

        for key in PROPERTY_KEYS:
            mod = self.embeddings[key]
            if key in active_keys:
                val_attr  = getattr(batch, key, None)
                miss_attr = getattr(batch, f'{key}_missing', None)
                if val_attr is not None:
                    val     = val_attr.view(B, 1).float().to(device)
                    missing = (miss_attr.view(B, 1).float().to(device)
                               if miss_attr is not None
                               else torch.zeros(B, 1, device=device))
                    parts.append(mod(val, missing))
                else:
                    parts.append(
                        mod(torch.zeros(B, 1, device=device),
                            torch.ones(B, 1, device=device)))
            else:

                null_emb = mod.null_vec.unsqueeze(0).expand(B, -1)
                parts.append(null_emb)

        return torch.cat(parts, dim=-1)