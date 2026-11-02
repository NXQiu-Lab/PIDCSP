import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse

from pidcsp.pl_modules.pidcsp_c import SinusoidsEmbedding
from pidcsp.pl_modules.pidcsp_g_diffusion import SinusoidalTimeEmbeddings
from pidcsp.common.conditional_data_utils import (
    PROPERTY_KEYS, DEFAULT_LOG_TRANSFORM_PROPS,
)

MAX_ATOMIC_NUM = 100


class VanillaMPLayer(nn.Module):


    def __init__(self, hidden_dim, act_fn=nn.SiLU(), dis_emb=None, ln=True, ip=True):
        super().__init__()
        self.dis_dim = 3 if dis_emb is None else dis_emb.dim
        self.dis_emb = dis_emb
        self.ip = ip
        self.ln = ln

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_features)

        src, dst = edge_index[0], edge_index[1]
        hi, hj = node_features[src], node_features[dst]

        if frac_diff is None:
            frac_diff = (frac_coords[dst] - frac_coords[src]) % 1.0
        frac_diff_emb = self.dis_emb(frac_diff) if self.dis_emb is not None else frac_diff

        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_flat = lattice_ips.view(-1, 9)[edge2graph]

        edge_input = torch.cat([hi, hj, lattice_flat, frac_diff_emb], dim=1)
        edge_features = self.edge_mlp(edge_input)


        agg = scatter(edge_features, src, dim=0, reduce='mean', dim_size=node_features.shape[0])
        out = self.node_mlp(torch.cat([node_features, agg], dim=1))
        return node_input + out


class CrystalClassifierBase(nn.Module):


    def __init__(
        self,
        hidden_dim:      int  = 512,
        time_dim:        int  = 256,
        num_freqs:       int  = 128,
        edge_style:      str  = 'fc',
        ln:              bool = True,
        ip:              bool = True,
        clf_hidden_dim:  int  = 256,
        n_mp_layers:     int  = 3,
        prop_keys:       list = None,
        use_max_features: bool = False,
        include_b_as_x: bool = False,
        max_role_feature_dim: int = 12,
        log_transform_props: list = None,
    ):
        super().__init__()

        self.hidden_dim     = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.edge_style     = edge_style
        self.prop_keys      = prop_keys or PROPERTY_KEYS
        self.use_max_features = bool(use_max_features)
        self.include_b_as_x = bool(include_b_as_x)
        self.log_transform_props = set(
            DEFAULT_LOG_TRANSFORM_PROPS if log_transform_props is None
            else log_transform_props
        )


        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.node_embedding = nn.Linear(MAX_ATOMIC_NUM, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + time_dim, hidden_dim)
        self.max_role_embedding = (
            nn.Linear(3, hidden_dim) if self.use_max_features else None
        )


        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)


        self.mp_layers = nn.ModuleList([
            VanillaMPLayer(hidden_dim, nn.SiLU(), self.dis_emb, ln=ln, ip=ip)
            for _ in range(n_mp_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim) if ln else nn.Identity()


        self.lattice_encoder = nn.Sequential(
            nn.Linear(16, clf_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, clf_hidden_dim),
            nn.SiLU(),
        )

        self.max_feature_encoder = (
            nn.Sequential(
                nn.Linear(max_role_feature_dim, clf_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(clf_hidden_dim),
            ) if self.use_max_features else None
        )


        fusion_input_dim = hidden_dim + clf_hidden_dim + (
            clf_hidden_dim if self.use_max_features else 0
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, clf_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, clf_hidden_dim),
            nn.SiLU(),
        )


        self.predictors = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(clf_hidden_dim, clf_hidden_dim // 2),
                nn.SiLU(),
                nn.LayerNorm(clf_hidden_dim // 2),
                nn.Dropout(0.1),
                nn.Linear(clf_hidden_dim // 2, clf_hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(clf_hidden_dim // 4, 1),
            )
            for key in self.prop_keys
        })

        self._init_weights()

    def _init_weights(self):
        for key in self.prop_keys:
            last = self.predictors[key][-1]
            nn.init.xavier_uniform_(last.weight, gain=0.1)
            nn.init.zeros_(last.bias)

        for net in [self.lattice_encoder, self.fusion]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.atom_latent_emb.weight, gain=0.5)
        nn.init.zeros_(self.atom_latent_emb.bias)

    def encode_lattice(self, lattice: torch.Tensor) -> torch.Tensor:

        B = lattice.shape[0]
        eps = 1e-8

        norms = lattice.norm(dim=-1)
        a, b, c = lattice[:, 0], lattice[:, 1], lattice[:, 2]
        cos_bc = (b * c).sum(-1) / (b.norm(dim=-1) * c.norm(dim=-1) + eps)
        cos_ac = (a * c).sum(-1) / (a.norm(dim=-1) * c.norm(dim=-1) + eps)
        cos_ab = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)
        cos_angles = torch.stack([cos_bc, cos_ac, cos_ab], dim=-1)
        volume = torch.det(lattice).abs().unsqueeze(-1)
        flat = lattice.reshape(B, 9)

        feat = torch.cat([norms, cos_angles, volume, flat], dim=-1)
        return self.lattice_encoder(feat)

    def gen_edges(self, num_atoms, frac_coords, node2graph):


        if self.edge_style == 'fc':
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            frac_diff = (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
            return fc_edges, frac_diff
        else:
            raise NotImplementedError(f"edge_style={self.edge_style} not supported")

    def forward(
        self,
        atom_type_probs: torch.Tensor,
        frac_coords:     torch.Tensor,
        lattice:         torch.Tensor,
        num_atoms:       torch.Tensor,
        node2graph:      torch.Tensor,
        t:               torch.Tensor,
    ) -> dict:
        B = lattice.shape[0]


        time_emb = self.time_embedding(t)
        time_per_atom = time_emb.repeat_interleave(num_atoms, dim=0)

        h = self.node_embedding(atom_type_probs)
        h = self.atom_latent_emb(torch.cat([h, time_per_atom], dim=-1))

        max_graph_feat = None
        if self.use_max_features:
            from pidcsp.common.max_data_utils import soft_max_role_features

            node_roles, max_graph_feat = soft_max_role_features(
                atom_type_probs,
                node2graph,
                max_atomic_num=MAX_ATOMIC_NUM,
                include_b_as_x=self.include_b_as_x,
            )
            h = h + self.max_role_embedding(node_roles)

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, node2graph)
        edge2graph = node2graph[edges[0]]

        for layer in self.mp_layers:
            h = layer(h, frac_coords, lattice, edges, edge2graph, frac_diff=frac_diff)

        h = self.final_norm(h)

        graph_feat = scatter(h, node2graph, dim=0, reduce='mean', dim_size=B)
        lat_feat = self.encode_lattice(lattice)
        fusion_parts = [graph_feat, lat_feat]
        if self.use_max_features:
            fusion_parts.append(self.max_feature_encoder(max_graph_feat))
        combined = self.fusion(torch.cat(fusion_parts, dim=-1))

        return {key: self.predictors[key](combined) for key in self.prop_keys}
