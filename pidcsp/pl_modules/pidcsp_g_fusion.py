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


class PhysicsFusionLayer(nn.Module):


    def __init__(
        self,
        hidden_dim: int = 512,
        phys_hidden_dim: int = 512,
        phys_raw_dim: int = 12,
        act_fn: nn.Module = nn.SiLU(),
        dis_emb: nn.Module = None,
        ln: bool = True,
        ip: bool = True,
        num_heads: int = 8,
        dropout: float = 0.0,
        alpha: float = 0.2,
        phys_edge_weight: float = 0.3,
        phys_node_weight: float = 0.2,
        layer_idx: int = 0,
        num_layers: int = 6,

        enable_node_fusion: bool = True,
        fusion_dropout: float = 0.15,
        consistency_aware: bool = True,
        use_layer_decay: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.phys_hidden_dim = phys_hidden_dim
        self.phys_raw_dim = phys_raw_dim
        self.ln = ln
        self.ip = ip
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.enable_node_fusion = enable_node_fusion
        self.consistency_aware = consistency_aware

        self.dis_emb = dis_emb
        self.dis_dim = dis_emb.dim if dis_emb is not None else 3


        edge_input_dim = hidden_dim * 2 + 9 + self.dis_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )


        phys_edge_input_dim = phys_raw_dim * 3
        self.phys_edge_mlp = nn.Sequential(
            nn.Linear(phys_edge_input_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.phys_edge_scale = nn.Parameter(torch.tensor(phys_edge_weight))


        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, num_heads)
        )
        self.leaky_relu = nn.LeakyReLU(alpha)

        if dropout > 0:
            self.attn_dropout = nn.Dropout(dropout)
        else:
            self.attn_dropout = None


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )


        if enable_node_fusion:

            if consistency_aware:

                self.fusion_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                    act_fn,
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid()
                )
            else:
                self.fusion_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    act_fn,
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid()
                )


            self.phys_transform = nn.Sequential(
                nn.Linear(phys_hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, hidden_dim)
            )


            if use_layer_decay:

                progress = layer_idx / max(num_layers - 1, 1)
                decay_factor = 0.3 + 0.7 * (1 + math.cos(math.pi * progress)) / 2
                init_weight = phys_node_weight * decay_factor
            else:

                init_weight = phys_node_weight

            self.layer_phys_scale = nn.Parameter(torch.tensor(init_weight))


            self.fusion_dropout = nn.Dropout(fusion_dropout)

        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_learn: torch.Tensor,
        h_phys: torch.Tensor,
        phys_raw: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        edge_index: torch.Tensor,
        edge2graph: torch.Tensor,
        frac_diff: torch.Tensor = None
    ) -> torch.Tensor:

        N = h_learn.size(0)
        src, dst = edge_index[0], edge_index[1]

        h_input = h_learn

        if self.ln:
            h_learn = self.layer_norm(h_learn)


        edge_base = self._compute_base_edge(
            h_learn, frac_coords, lattices,
            edge_index, edge2graph, frac_diff
        )

        edge_phys = self._compute_phys_edge(phys_raw, src, dst)

        phys_edge_scale = torch.sigmoid(self.phys_edge_scale)
        edge_features = edge_base + phys_edge_scale * edge_phys

        attn_weights = self._compute_attention(edge_features, dst)
        weighted_features = edge_features * attn_weights
        agg = scatter(weighted_features, dst, dim=0, reduce='sum', dim_size=N)


        h_msg = self.node_mlp(torch.cat([h_learn, agg], dim=1))
        h_updated = h_input + h_msg


        if self.enable_node_fusion and h_phys is not None:
            h_out = self._apply_physics_fusion(h_updated, h_phys)
        else:
            h_out = h_updated

        return h_out

    def _apply_physics_fusion(
        self,
        h_updated: torch.Tensor,
        h_phys: torch.Tensor
    ) -> torch.Tensor:


        if self.consistency_aware:

            h_learn_norm = F.normalize(h_updated, p=2, dim=-1)
            h_phys_norm = F.normalize(h_phys, p=2, dim=-1)
            consistency = (h_learn_norm * h_phys_norm).sum(dim=-1, keepdim=True)


            gate_input = torch.cat([h_updated, h_phys, consistency], dim=-1)
        else:
            gate_input = torch.cat([h_updated, h_phys], dim=-1)


        gate = self.fusion_gate(gate_input)


        phys_transformed = self.phys_transform(h_phys)


        phys_transformed = self.fusion_dropout(phys_transformed)


        layer_scale = torch.sigmoid(self.layer_phys_scale)


        h_out = h_updated + gate * layer_scale * phys_transformed

        return h_out

    def _compute_base_edge(self, h_learn, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):

        src, dst = edge_index[0], edge_index[1]

        h_i = h_learn[src]
        h_j = h_learn[dst]

        if frac_diff is None:
            x_i = frac_coords[src]
            x_j = frac_coords[dst]
            frac_diff = (x_j - x_i) % 1.

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

    def _compute_phys_edge(self, phys_raw, src, dst):

        phys_i = phys_raw[src]
        phys_j = phys_raw[dst]
        phys_diff = torch.abs(phys_i - phys_j)

        phys_edge_input = torch.cat([phys_i, phys_j, phys_diff], dim=-1)
        return self.phys_edge_mlp(phys_edge_input)

    def _compute_attention(self, edge_features, dst):

        attn_scores = self.attention_mlp(edge_features)
        attn_scores = self.leaky_relu(attn_scores)

        attn_weights = scatter_softmax(attn_scores, dst, dim=0)
        attn_weights = attn_weights.mean(dim=-1, keepdim=True)

        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights)

        return attn_weights


class OriginalCSPLayer(nn.Module):


    def __init__(self, hidden_dim=512, act_fn=nn.SiLU(), dis_emb=None, ln=True, ip=True):
        super().__init__()
        self.dis_dim = 3 if dis_emb is None else dis_emb.dim
        self.dis_emb = dis_emb
        self.ip = ip
        self.ln = ln

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn, nn.Linear(hidden_dim, hidden_dim), act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn, nn.Linear(hidden_dim, hidden_dim), act_fn
        )
        if ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
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

        agg = scatter(edge_features, dst, dim=0, reduce='mean', dim_size=node_features.shape[0])
        out = self.node_mlp(torch.cat([node_features, agg], dim=1))

        return node_input + out


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

        phys_edge_weight: float = 0.3,
        phys_node_weight: float = 0.15,

        fusion_dropout: float = 0.15,
        consistency_aware: bool = True,
        node_fusion_layers: str = 'first_half',
        use_layer_decay: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.max_atoms = max_atoms
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
        self.node_fusion_layers = node_fusion_layers


        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.SiLU()


        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)


        self.phys_raw_dim = 0

        if use_phys_emb:
            self.phys_emb = PhysEmbedding(
                props=phys_props,
                props_grad=phys_props_grad,
                pg=phys_pg,
                phys_data_path=phys_data_path
            )

            if phys_props and self.phys_emb.n_properties > 0:
                self.phys_raw_dim = self.phys_emb.n_properties

            phys_emb_dim = self.phys_raw_dim
            if phys_pg:
                self.group_embedding = nn.Embedding(
                    self.phys_emb.group_size + 1, hidden_dim // 4
                )
                self.period_embedding = nn.Embedding(
                    self.phys_emb.period_size + 1, hidden_dim // 4
                )
                phys_emb_dim += hidden_dim // 2

            self.phys_emb_dim = phys_emb_dim

            if phys_emb_dim > 0:
                self.phys_node_proj = nn.Sequential(
                    nn.Linear(phys_emb_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )


        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)


        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        else:
            self.dis_emb = None


        for i in range(num_layers):

            if node_fusion_layers == 'all':
                enable_fusion = True
            elif node_fusion_layers == 'first_half':
                enable_fusion = (i < num_layers // 2)
            elif node_fusion_layers == 'last_half':
                enable_fusion = (i >= num_layers // 2)
            else:
                enable_fusion = False

            if use_attention and use_phys_emb and self.phys_raw_dim > 0:
                layer = PhysicsFusionLayer(
                    hidden_dim=hidden_dim,
                    phys_hidden_dim=hidden_dim,
                    phys_raw_dim=self.phys_raw_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    ip=ip,
                    num_heads=num_heads,
                    dropout=dropout,
                    alpha=alpha,
                    phys_edge_weight=phys_edge_weight,
                    phys_node_weight=phys_node_weight,
                    layer_idx=i,
                    num_layers=num_layers,
                    enable_node_fusion=enable_fusion,
                    fusion_dropout=fusion_dropout,
                    consistency_aware=consistency_aware,
                    use_layer_decay=use_layer_decay,
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
        if pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

    def get_phys_raw(self, atom_types: torch.Tensor) -> torch.Tensor:

        if not self.use_phys_emb or self.phys_raw_dim == 0:
            return None

        device = atom_types.device
        properties = self.phys_emb.properties.to(device)
        max_z = properties.shape[0] - 1

        if self.smooth:
            n_atoms_in_probs = atom_types.shape[1]

            if n_atoms_in_probs > max_z + 1:
                probs = atom_types[:, :max_z + 1]
            else:
                probs = atom_types
                properties = properties[:n_atoms_in_probs]

            probs = F.softmax(probs, dim=-1)
            phys_raw = torch.matmul(probs, properties)
            return phys_raw
        else:
            safe_atom_types = atom_types.clamp(0, max_z)
            return properties[safe_atom_types]

    def get_phys_node_embedding(self, atom_types: torch.Tensor) -> torch.Tensor:

        if not self.use_phys_emb:
            return None

        device = atom_types.device
        phys_features = []


        if self.phys_emb.props and self.phys_emb.n_properties > 0:
            properties = self.phys_emb.properties.to(device)
            max_z = properties.shape[0] - 1

            if self.smooth:
                n_atoms_in_probs = atom_types.shape[1]
                if n_atoms_in_probs > max_z + 1:
                    probs = atom_types[:, :max_z + 1]
                else:
                    probs = atom_types
                    properties = properties[:n_atoms_in_probs]

                probs = F.softmax(probs, dim=-1)
                props = torch.matmul(probs, properties)
            else:
                safe_atom_types = atom_types.clamp(0, max_z)
                props = properties[safe_atom_types]

            phys_features.append(props)


        if self.phys_pg and self.phys_emb.period is not None:
            period = self.phys_emb.period.to(device)
            max_idx = period.shape[0] - 1

            if self.smooth:
                n_atoms_in_probs = atom_types.shape[1]
                probs = F.softmax(atom_types[:, :min(n_atoms_in_probs, max_idx + 1)], dim=-1)
                period_indices = period[:probs.shape[1]]
                period_emb_per_atom = self.period_embedding(period_indices)
                period_emb = torch.matmul(probs, period_emb_per_atom)
            else:
                safe_atom_types = atom_types.clamp(0, max_idx)
                period_idx = period[safe_atom_types]
                period_emb = self.period_embedding(period_idx)

            phys_features.append(period_emb)


        if self.phys_pg and self.phys_emb.group is not None:
            group = self.phys_emb.group.to(device)
            max_idx = group.shape[0] - 1

            if self.smooth:
                n_atoms_in_probs = atom_types.shape[1]
                probs = F.softmax(atom_types[:, :min(n_atoms_in_probs, max_idx + 1)], dim=-1)
                group_indices = group[:probs.shape[1]]
                group_emb_per_atom = self.group_embedding(group_indices)
                group_emb = torch.matmul(probs, group_emb_per_atom)
            else:
                safe_atom_types = atom_types.clamp(0, max_idx)
                group_idx = group[safe_atom_types]
                group_emb = self.group_embedding(group_idx)

            phys_features.append(group_emb)

        if phys_features:
            phys_features = torch.cat(phys_features, dim=-1)
            return self.phys_node_proj(phys_features)

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
        cell_offsets_new = self.select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
        edge_vector_new = self.select_symmetric_edges(edge_vector, mask, edge_reorder_idx, True)

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


        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]


        if self.smooth:
            h_learn = self.node_embedding(atom_types)
        else:
            h_learn = self.node_embedding(atom_types - 1)


        phys_raw = None
        h_phys = None

        if self.use_phys_emb:
            phys_raw = self.get_phys_raw(atom_types)
            h_phys = self.get_phys_node_embedding(atom_types)


        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        h_learn = torch.cat([h_learn, t_per_atom], dim=1)
        h_learn = self.atom_latent_emb(h_learn)


        for i in range(self.num_layers):
            layer = self._modules[f"csp_layer_{i}"]

            if isinstance(layer, PhysicsFusionLayer):
                h_learn = layer(
                    h_learn=h_learn,
                    h_phys=h_phys,
                    phys_raw=phys_raw,
                    frac_coords=frac_coords,
                    lattices=lattices,
                    edge_index=edges,
                    edge2graph=edge2graph,
                    frac_diff=frac_diff
                )
            else:
                h_learn = layer(
                    h_learn, frac_coords, lattices,
                    edges, edge2graph, frac_diff=frac_diff
                )


        if self.ln:
            h_learn = self.final_layer_norm(h_learn)

        coord_out = self.coord_out(h_learn)
        graph_features = scatter(h_learn, node2graph, dim=0, reduce='mean')

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(-1, 3, 3)

        if self.ip:
            lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)

        if self.pred_type:
            type_out = self.type_out(h_learn)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out