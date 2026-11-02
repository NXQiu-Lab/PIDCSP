import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse

from pidcsp.pl_modules.pidcsp_g import (
    PhysEmbedding,
    SinusoidsEmbedding,
    PhysicsGuidedEdgeLayer,
    MAX_ATOMIC_NUM,
)
from pidcsp.pl_modules.pidcsp_g_diffusion import SinusoidalTimeEmbeddings
from pidcsp.common.conditional_data_utils import PROPERTY_KEYS


LOG_TRANSFORM_PROPS = {"band_gap"}


class CrystalClassifier(nn.Module):


    def __init__(
        self,

        hidden_dim:       int   = 512,
        time_dim:         int   = 256,
        num_freqs:        int   = 128,
        edge_style:       str   = 'fc',

        clf_hidden_dim:   int   = 256,
        n_mp_layers:      int   = 3,

        phys_data_path:   str   = "",
        use_phys_emb:     bool  = True,
        phys_props:       bool  = True,
        phys_pg:          bool  = True,

        num_heads:        int   = 4,
        dropout:          float = 0.0,
        alpha:            float = 0.2,
        phys_edge_weight: float = 0.5,

        use_time_phys_mod: bool = True,
        time_mod_min:     float = 0.7,

        prop_keys:        list  = None,

        log_transform_props: list = None,
        use_max_features: bool = False,
        include_b_as_x: bool = False,
        max_role_feature_dim: int = 12,
    ):
        super().__init__()

        self.hidden_dim    = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.time_dim      = time_dim
        self.edge_style    = edge_style
        self.prop_keys     = prop_keys or PROPERTY_KEYS


        if log_transform_props is None:
            self.log_transform_props = {
                k for k in self.prop_keys if k in LOG_TRANSFORM_PROPS
            }
        else:
            self.log_transform_props = set(log_transform_props)
        self.use_max_features = bool(use_max_features)
        self.include_b_as_x = bool(include_b_as_x)


        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


        self.node_embedding = nn.Linear(MAX_ATOMIC_NUM, hidden_dim)
        self.max_role_embedding = (
            nn.Linear(3, hidden_dim) if self.use_max_features else None
        )


        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)


        self.phys_dim    = 0
        self.use_phys_emb = use_phys_emb
        self.phys_emb    = None

        if use_phys_emb and phys_data_path:
            try:
                self.phys_emb = PhysEmbedding(
                    props          = phys_props,
                    props_grad     = False,
                    pg             = phys_pg,
                    phys_data_path = phys_data_path,
                )
                if phys_props and self.phys_emb.n_properties > 0:
                    self.phys_dim = self.phys_emb.n_properties
                if phys_pg:
                    self.group_embedding = nn.Embedding(
                        self.phys_emb.group_size + 1, 8)
                    self.period_embedding = nn.Embedding(
                        self.phys_emb.period_size + 1, 8)
                    self.phys_dim += 16
                print(f"[Classifier] phys_dim={self.phys_dim}")
            except Exception as e:
                print(f"[Classifier] ⚠️  物理嵌入加载失败: {e}")
                self.phys_emb = None
                self.phys_dim = 0


        self.mp_layers = nn.ModuleList()
        for _ in range(n_mp_layers):
            if self.phys_emb is not None and self.phys_dim > 0:
                layer = PhysicsGuidedEdgeLayer(
                    hidden_dim        = hidden_dim,
                    phys_dim          = self.phys_dim,
                    act_fn            = nn.SiLU(),
                    dis_emb           = self.dis_emb,
                    ln                = True,
                    ip                = True,
                    num_heads         = num_heads,
                    dropout           = dropout,
                    alpha             = alpha,
                    phys_edge_weight  = phys_edge_weight,
                    latent_dim        = time_dim,
                    use_time_phys_mod = use_time_phys_mod,
                    time_mod_min      = time_mod_min,
                )
            else:
                layer = self._make_fallback_layer(hidden_dim)
            self.mp_layers.append(layer)

        self.final_norm = nn.LayerNorm(hidden_dim)


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

    def _make_fallback_layer(self, hidden_dim: int) -> nn.Module:

        class SimpleMPLayer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.msg  = nn.Sequential(
                    nn.Linear(dim*2, dim), nn.SiLU(), nn.Linear(dim, dim))
                self.upd  = nn.Sequential(
                    nn.Linear(dim*2, dim), nn.SiLU(), nn.Linear(dim, dim))
                self.norm = nn.LayerNorm(dim)

            def forward(self, node_features, edge_index, **kwargs):
                src, dst = edge_index
                msg = self.msg(torch.cat(
                    [node_features[src], node_features[dst]], dim=-1))
                agg = scatter(msg, dst, dim=0, reduce='mean',
                              dim_size=node_features.shape[0])
                out = self.upd(torch.cat([node_features, agg], dim=-1))
                return self.norm(out + node_features)
        return SimpleMPLayer(hidden_dim)

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

        for m in self.time_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_phys_features(
        self, atom_type_probs: torch.Tensor
    ) -> torch.Tensor:

        if self.phys_emb is None or self.phys_dim == 0:
            return None

        device = atom_type_probs.device
        atom_types_discrete = atom_type_probs.argmax(dim=-1) + 1
        atom_types_discrete = atom_types_discrete.long()

        phys_features = []

        if self.phys_emb.props and self.phys_emb.n_properties > 0:
            properties = self.phys_emb.properties.to(device)
            max_z      = properties.shape[0] - 1
            safe_types = atom_types_discrete.clamp(0, max_z)
            phys_features.append(properties[safe_types])

        if hasattr(self, 'period_embedding') and \
                self.phys_emb.period is not None:
            period     = self.phys_emb.period.to(device)
            safe_types = atom_types_discrete.clamp(0, period.shape[0]-1)
            phys_features.append(
                self.period_embedding(period[safe_types]))

        if hasattr(self, 'group_embedding') and \
                self.phys_emb.group is not None:
            group      = self.phys_emb.group.to(device)
            safe_types = atom_types_discrete.clamp(0, group.shape[0]-1)
            phys_features.append(
                self.group_embedding(group[safe_types]))

        return (torch.cat(phys_features, dim=-1)
                if phys_features else None)

    def encode_lattice(self, lattice: torch.Tensor) -> torch.Tensor:


        B   = lattice.shape[0]
        eps = 1e-8

        norms = lattice.norm(dim=-1)
        a, b, c = lattice[:, 0], lattice[:, 1], lattice[:, 2]
        cos_bc  = (b * c).sum(-1) / (
            b.norm(dim=-1) * c.norm(dim=-1) + eps)
        cos_ac  = (a * c).sum(-1) / (
            a.norm(dim=-1) * c.norm(dim=-1) + eps)
        cos_ab  = (a * b).sum(-1) / (
            a.norm(dim=-1) * b.norm(dim=-1) + eps)
        cos_angles = torch.stack([cos_bc, cos_ac, cos_ab], dim=-1)
        volume     = torch.det(lattice).abs().unsqueeze(-1)
        flat       = lattice.reshape(B, 9)

        feat = torch.cat([norms, cos_angles, volume, flat], dim=-1)
        return self.lattice_encoder(feat)

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):


        if self.edge_style == 'fc':
            lis      = [torch.ones(n, n, device=num_atoms.device)
                        for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)

            frac_diff   = (frac_coords[fc_edges[1]]
                           - frac_coords[fc_edges[0]]) % 1.
            return fc_edges, frac_diff
        else:
            raise NotImplementedError(
                f"edge_style={self.edge_style} not supported")

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


        time_emb       = self.time_embedding(t)
        time_feat      = self.time_proj(time_emb)
        time_per_atom  = time_feat[node2graph]


        h = self.node_embedding(atom_type_probs)
        h = h + time_per_atom

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


        phys_features = self.get_phys_features(atom_type_probs)


        edges, frac_diff = self.gen_edges(
            num_atoms, frac_coords, lattice, node2graph
        )
        edge2graph = node2graph[edges[0]]


        t_per_atom_raw = time_emb[node2graph]

        for mp_layer in self.mp_layers:
            if isinstance(mp_layer, PhysicsGuidedEdgeLayer):
                h = mp_layer(
                    node_features = h,
                    phys_features = phys_features,
                    frac_coords   = frac_coords,
                    lattices      = lattice,
                    edge_index    = edges,
                    edge2graph    = edge2graph,
                    frac_diff     = frac_diff,
                    t_per_atom    = t_per_atom_raw,
                )
            else:
                h = mp_layer(h, edges)

        h = self.final_norm(h)


        graph_feat = scatter(
            h, node2graph, dim=0,
            reduce='mean', dim_size=B,
        )


        lat_feat = self.encode_lattice(lattice)


        fusion_parts = [graph_feat, lat_feat]
        if self.use_max_features:
            fusion_parts.append(self.max_feature_encoder(max_graph_feat))
        combined = self.fusion(torch.cat(fusion_parts, dim=-1))


        return {
            key: self.predictors[key](combined)
            for key in self.prop_keys
        }
