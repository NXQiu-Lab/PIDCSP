import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_softmax
from torch_geometric.utils import dense_to_sparse

from pidcsp.common.data_utils import radius_graph_pbc, repeat_blocks

MAX_ATOMIC_NUM = 100


class PhysEmbedding(nn.Module):


    def __init__(
        self,
        props=True,
        props_grad=False,
        pg=False,
        phys_data_path="embeddings/phys_embedding_data.pt"
    ) -> None:
        super().__init__()

        self.props = props
        self.props_grad = props_grad
        self.pg = pg

        self.group_size = 0
        self.period_size = 0
        self.n_properties = 0


        phys_data_path = PhysEmbedding._resolve_path(phys_data_path)

        if not os.path.exists(phys_data_path):
            raise FileNotFoundError(
                f"Physical embedding data not found at {phys_data_path}. "
                f"Run: python scripts/prepare_pidcsp_physics.py"
            )

        data = torch.load(phys_data_path, map_location="cpu")

        if pg:
            self.group_size = data["group_size"]
            self.period_size = data["period_size"]
            self.register_buffer("group", data["group"])
            self.register_buffer("period", data["period"])
        else:
            self.register_buffer("group", None)
            self.register_buffer("period", None)

        if props:
            self.n_properties = data["n_properties"]
            self.properties_list = data["properties_list"]
            properties = data["properties"]

            if props_grad:
                self.register_parameter("properties", nn.Parameter(properties))
            else:
                self.register_buffer("properties", properties)
        else:
            self.n_properties = 0
            self.register_buffer("properties", None)

    @staticmethod
    def _resolve_path(path: str) -> str:


        import os
        from pathlib import Path


        if os.path.exists(path):
            return path


        path_obj = Path(path)
        try:
            parts = path_obj.parts
            if 'embeddings' in parts:
                idx      = list(parts).index('embeddings')
                rel_path = Path(*parts[idx:])
            else:
                rel_path = Path('embeddings') / 'phys_embedding_data.pt'
        except Exception:
            rel_path = Path('embeddings') / 'phys_embedding_data.pt'


        try:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            candidate    = project_root / rel_path
            if candidate.exists():
                print(f"[PhysEmbedding] 路径自动修正:")
                print(f"  旧: {path}")
                print(f"  新: {candidate}")
                return str(candidate)
        except Exception:
            pass


        env_root = os.environ.get('PROJECT_ROOT', '')
        if env_root:
            candidate = Path(env_root) / rel_path
            if candidate.exists():
                print(f"[PhysEmbedding] 使用 PROJECT_ROOT: {candidate}")
                return str(candidate)


        cwd_candidate = Path.cwd() / rel_path
        if cwd_candidate.exists():
            print(f"[PhysEmbedding] 使用 cwd: {cwd_candidate}")
            return str(cwd_candidate)


        print(f"[PhysEmbedding] ⚠️  无法自动解析路径: {path}")
        return path


class SinusoidsEmbedding(nn.Module):


    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class OriginalCSPLayer(nn.Module):


    def __init__(self, hidden_dim=512, act_fn=nn.SiLU(), dis_emb=None, ln=True, ip=True):
        super().__init__()
        self.dis_dim = 3 if dis_emb is None else dis_emb.dim
        self.dis_emb = dis_emb
        self.ip = ip
        self.ln = ln

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        if ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, frac_coords, lattices, edge_index,
                edge2graph, frac_diff=None, **kwargs):

        src, dst = edge_index[0], edge_index[1]
        node_input = node_features

        if self.ln:
            node_features = self.layer_norm(node_features)

        h_i, h_j = node_features[src], node_features[dst]

        if frac_diff is None:
            frac_diff = (frac_coords[dst] - frac_coords[src]) % 1.
        if self.dis_emb:
            frac_diff = self.dis_emb(frac_diff)

        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_flat = lattice_ips.view(-1, 9)[edge2graph]

        edge_input = torch.cat([h_i, h_j, lattice_flat, frac_diff], dim=1)
        edge_features = self.edge_mlp(edge_input)
        agg = scatter(edge_features, dst, dim=0, reduce='mean',
                      dim_size=node_features.shape[0])
        out = self.node_mlp(torch.cat([node_features, agg], dim=1))
        return node_input + out


class PhysicsGuidedEdgeLayer(nn.Module):


    def __init__(
        self,
        hidden_dim: int = 512,
        phys_dim: int = 12,
        act_fn: nn.Module = nn.SiLU(),
        dis_emb: nn.Module = None,
        ln: bool = True,
        ip: bool = True,

        num_heads: int = 8,
        dropout: float = 0.0,
        alpha: float = 0.2,

        phys_edge_weight: float = 0.5,

        latent_dim: int = 256,
        use_time_phys_mod: bool = True,
        time_mod_min: float = 0.7,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.phys_dim = phys_dim
        self.ln = ln
        self.ip = ip
        self.num_heads = num_heads


        self.dis_emb = dis_emb
        self.dis_dim = dis_emb.dim if dis_emb is not None else 3


        edge_input_dim = hidden_dim * 2 + 9 + self.dis_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )


        phys_edge_input_dim = phys_dim * 3
        self.phys_edge_mlp = nn.Sequential(
            nn.Linear(phys_edge_input_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, hidden_dim),
        )


        self.phys_edge_scale = nn.Parameter(torch.tensor(phys_edge_weight))


        self.use_time_phys_mod = use_time_phys_mod and latent_dim > 0
        self.time_mod_min = time_mod_min
        if self.use_time_phys_mod:
            self.time_phys_proj = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )


        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, num_heads)
        )
        self.leaky_relu = nn.LeakyReLU(alpha)

        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )


        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        phys_features: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        edge_index: torch.Tensor,
        edge2graph: torch.Tensor,
        frac_diff: torch.Tensor = None,
        t_per_atom: torch.Tensor = None,
    ) -> torch.Tensor:

        N = node_features.size(0)
        src, dst = edge_index[0], edge_index[1]

        node_input = node_features

        if self.ln:
            node_features = self.layer_norm(node_features)


        edge_base = self._compute_base_edge_features(
            node_features, frac_coords, lattices,
            edge_index, edge2graph, frac_diff
        )


        edge_phys = self._compute_phys_edge_features(phys_features, src, dst)


        phys_scale = torch.sigmoid(self.phys_edge_scale)

        if self.use_time_phys_mod and t_per_atom is not None:

            t_edge = t_per_atom[src]
            time_mod_raw = torch.sigmoid(self.time_phys_proj(t_edge))
            time_mod = self.time_mod_min + (1.0 - self.time_mod_min) * time_mod_raw
            phys_scale = phys_scale * time_mod

        edge_features = edge_base + phys_scale * edge_phys


        attn_weights = self._compute_attention(edge_features, dst)


        weighted_features = edge_features * attn_weights
        agg = scatter(weighted_features, dst, dim=0, reduce='sum', dim_size=N)


        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)


        return node_input + out

    def _compute_base_edge_features(
        self, node_features, frac_coords, lattices,
        edge_index, edge2graph, frac_diff=None
    ):

        src, dst = edge_index[0], edge_index[1]
        h_i = node_features[src]
        h_j = node_features[dst]

        if frac_diff is None:
            frac_diff = (frac_coords[dst] - frac_coords[src]) % 1.

        if self.dis_emb is not None:
            frac_diff_emb = self.dis_emb(frac_diff)
        else:
            frac_diff_emb = frac_diff

        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flat = lattice_ips.view(-1, 9)
        lattice_ips_edges = lattice_ips_flat[edge2graph]

        edge_input = torch.cat([h_i, h_j, lattice_ips_edges, frac_diff_emb], dim=1)
        return self.edge_mlp(edge_input)

    def _compute_phys_edge_features(self, phys_features, src, dst):

        phys_i = phys_features[src]
        phys_j = phys_features[dst]
        phys_diff = torch.abs(phys_i - phys_j)
        phys_edge_input = torch.cat([phys_i, phys_j, phys_diff], dim=-1)
        return self.phys_edge_mlp(phys_edge_input)

    def _compute_attention(self, edge_features, dst):

        attn_scores = self.attention_mlp(edge_features)
        attn_scores = self.leaky_relu(attn_scores)
        attn_weights = scatter_softmax(attn_scores, dst, dim=0)
        attn_weights = attn_weights.mean(dim=-1, keepdim=True)
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)
        return attn_weights


class PIDCSPNet(nn.Module):


    def __init__(
        self,

        hidden_dim: int = 512,
        latent_dim: int = 0,
        num_layers: int = 6,
        max_atoms: int = 100,
        act_fn: str = 'silu',
        dis_emb: str = 'sin',
        num_freqs: int = 128,
        edge_style: str = 'fc',
        cutoff: float = 6.0,
        max_neighbors: int = 20,
        ln: bool = True,
        ip: bool = True,
        smooth: bool = False,
        pred_type: bool = False,
        pred_scalar: bool = False,

        use_phys_emb: bool = True,
        phys_props: bool = True,
        phys_props_grad: bool = False,
        phys_pg: bool = True,
        phys_data_path: str = "embeddings/phys_embedding_data.pt",

        use_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.0,
        alpha: float = 0.2,
        phys_edge_weight: float = 0.5,

        use_time_phys_mod: bool = True,
        time_mod_min: float = 0.7,
        use_phys_type_guidance: bool = True,
        use_phys_lattice_bias: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.edge_style = edge_style
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.ip = ip
        self.smooth = smooth
        self.pred_type = pred_type
        self.pred_scalar = pred_scalar
        self.use_phys_emb = use_phys_emb
        self.use_attention = use_attention
        self.phys_pg = phys_pg


        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.SiLU()


        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)


        self.phys_dim = 0

        if use_phys_emb:
            self.phys_emb = PhysEmbedding(
                props=phys_props,
                props_grad=phys_props_grad,
                pg=phys_pg,
                phys_data_path=phys_data_path
            )

            if phys_props and self.phys_emb.n_properties > 0:
                self.phys_dim = self.phys_emb.n_properties


            if phys_pg:
                self.group_embedding = nn.Embedding(
                    self.phys_emb.group_size + 1, 8
                )
                self.period_embedding = nn.Embedding(
                    self.phys_emb.period_size + 1, 8
                )
                self.phys_dim += 16


        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)


        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        else:
            self.dis_emb = None


        for i in range(num_layers):
            if use_attention and use_phys_emb and self.phys_dim > 0:

                layer = PhysicsGuidedEdgeLayer(
                    hidden_dim=hidden_dim,
                    phys_dim=self.phys_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    ip=ip,
                    num_heads=num_heads,
                    dropout=dropout,
                    alpha=alpha,
                    phys_edge_weight=phys_edge_weight,

                    latent_dim=latent_dim,
                    use_time_phys_mod=use_time_phys_mod,
                    time_mod_min=time_mod_min,
                )
            else:
                layer = OriginalCSPLayer(
                    hidden_dim=hidden_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    ip=ip
                )
            self.add_module(f"csp_layer_{i}", layer)


        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)


        if pred_type:
            self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

            if use_phys_type_guidance and use_phys_emb and self.phys_dim > 0:
                self.type_phys_proj = nn.Linear(self.phys_dim, hidden_dim)
                self.type_phys_gate = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            else:
                self.type_phys_proj = None
                self.type_phys_gate = None


        if use_phys_lattice_bias and use_phys_emb and self.phys_dim > 0:
            self.lattice_phys_bias = nn.Sequential(
                nn.Linear(self.phys_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 9),
            )
            self.lattice_phys_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.lattice_phys_bias = None

        if pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)


    def get_phys_features(self, atom_types: torch.Tensor) -> torch.Tensor:


        if not self.use_phys_emb or self.phys_dim == 0:
            return None

        device = atom_types.device
        phys_features = []


        if atom_types.dim() == 2 and atom_types.size(-1) == MAX_ATOMIC_NUM:

            atom_types_discrete = atom_types.argmax(dim=-1) + 1
        else:

            atom_types_discrete = atom_types

        atom_types_discrete = atom_types_discrete.long()


        if self.phys_emb.props and self.phys_emb.n_properties > 0:
            properties = self.phys_emb.properties.to(device)
            max_z = properties.shape[0] - 1
            safe_atom_types = atom_types_discrete.clamp(0, max_z)
            props = properties[safe_atom_types]
            phys_features.append(props)


        if self.phys_pg and self.phys_emb.period is not None:
            period = self.phys_emb.period.to(device)
            max_idx = period.shape[0] - 1
            safe_atom_types = atom_types_discrete.clamp(0, max_idx)
            period_idx = period[safe_atom_types]
            period_emb = self.period_embedding(period_idx)
            phys_features.append(period_emb)


        if self.phys_pg and self.phys_emb.group is not None:
            group = self.phys_emb.group.to(device)
            max_idx = group.shape[0] - 1
            safe_atom_types = atom_types_discrete.clamp(0, max_idx)
            group_idx = group[safe_atom_types]
            group_emb = self.group_embedding(group_idx)
            phys_features.append(group_emb)

        if phys_features:
            return torch.cat(phys_features, dim=-1)
        return None


    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        return tensor_cat[reorder_idx]

    def reorder_symmetric_edges(self, edge_index, cell_offsets, neighbors, edge_vector):
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)
        edge_index_cat = torch.cat([
            edge_index_new,
            torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
        ], dim=1)

        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device), neighbors
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(batch_edge, minlength=neighbors.size(0))

        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2, repeats=2,
            continuous_indexing=True, repeat_inc=edge_index_new.size(1)
        )

        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return edge_index_new, cell_offsets_new, neighbors_new, edge_vector_new

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == 'fc':
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            frac_diff = (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.
            return fc_edges, frac_diff
        elif self.edge_style == 'knn':
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords, None, None, num_atoms, self.cutoff, self.max_neighbors,
                device=num_atoms.device, lattices=lattices
            )
            j_index, i_index = edge_index
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()
            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
                edge_index, to_jimages, num_bonds, distance_vectors
            )
            return edge_index_new, -edge_vector_new


    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):


        edges, frac_diff = self.gen_edges(
            num_atoms, frac_coords, lattices, node2graph
        )
        edge2graph = node2graph[edges[0]]


        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)


        phys_features = self.get_phys_features(atom_types)


        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)


        for i in range(self.num_layers):
            layer = self._modules[f"csp_layer_{i}"]

            if isinstance(layer, PhysicsGuidedEdgeLayer):

                node_features = layer(
                    node_features=node_features,
                    phys_features=phys_features,
                    frac_coords=frac_coords,
                    lattices=lattices,
                    edge_index=edges,
                    edge2graph=edge2graph,
                    frac_diff=frac_diff,
                    t_per_atom=t_per_atom,
                )
            else:

                node_features = layer(
                    node_features, frac_coords, lattices,
                    edges, edge2graph, frac_diff=frac_diff
                )


        if self.ln:
            node_features = self.final_layer_norm(node_features)


        coord_out = self.coord_out(node_features)


        graph_features = scatter(
            node_features, node2graph, dim=0, reduce='mean'
        )

        if self.pred_scalar:
            return self.scalar_out(graph_features)


        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(-1, 3, 3)

        if self.lattice_phys_bias is not None and phys_features is not None:

            graph_phys = scatter(
                phys_features, node2graph, dim=0, reduce='mean'
            )
            lattice_bias = self.lattice_phys_bias(graph_phys).view(-1, 3, 3)
            gate = torch.sigmoid(self.lattice_phys_gate)
            lattice_out = lattice_out + gate * lattice_bias

        if self.ip:
            lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)


        if self.pred_type:
            if self.type_phys_proj is not None and phys_features is not None:

                phys_for_type = self.type_phys_proj(phys_features)

                gate = self.type_phys_gate(node_features)

                type_input = node_features + gate * phys_for_type
                type_out = self.type_out(type_input)
            else:
                type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out


    def get_phys_edge_scales(self) -> list:

        scales = []
        for i in range(self.num_layers):
            layer = self._modules[f"csp_layer_{i}"]
            if isinstance(layer, PhysicsGuidedEdgeLayer):
                scale = torch.sigmoid(layer.phys_edge_scale).item()
                scales.append(scale)
        return scales

    def get_lattice_phys_gate(self) -> float:

        if self.lattice_phys_bias is not None:
            return torch.sigmoid(self.lattice_phys_gate).item()
        return 0.0

    def get_type_phys_gate_stats(self) -> dict:

        if hasattr(self, 'type_phys_gate') and self.type_phys_gate is not None:
            bias = self.type_phys_gate[0].bias
            return {
                'mean_gate': torch.sigmoid(bias).mean().item(),
            }
        return {}