import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional
from torch.utils.data import Dataset
from torch_geometric.data import Data
from omegaconf import ValueNode

from pidcsp.common.data_utils import (
    preprocess,
    add_scaled_lattice_prop,
    get_scaler_from_data_list,
)
from pidcsp.common.conditional_data_utils import (
    parse_elements_to_chemsys,
    compute_property_statistics,
    normalize_property,
    PROPERTY_KEYS,
)


class ConditionalCrystDataset(Dataset):


    def __init__(
        self,
        name:                 ValueNode,
        path:                 ValueNode,
        prop:                 ValueNode,
        niggli:               ValueNode,
        primitive:            ValueNode,
        graph_method:         ValueNode,
        preprocess_workers:   ValueNode,
        lattice_scale_method: ValueNode,
        save_path:            ValueNode,
        tolerance:            ValueNode,
        use_space_group:      ValueNode,
        use_pos_index:        ValueNode,

        use_conditions:       bool = True,
        prop_stats_path:      str  = None,
        prop_keys             = None,
        log_transform_props   = None,
        **kwargs,
    ):
        super().__init__()
        self.path                 = path
        self.name                 = name
        self.prop                 = prop
        self.niggli               = niggli
        self.primitive            = primitive
        self.graph_method         = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group      = use_space_group
        self.use_pos_index        = use_pos_index
        self.tolerance            = tolerance
        self.use_conditions       = use_conditions
        self.prop_keys            = list(prop_keys or PROPERTY_KEYS)
        self.log_transform_props  = (
            None if log_transform_props is None else set(log_transform_props)
        )


        self.df = pd.read_csv(path)


        self._preprocess(save_path, preprocess_workers, prop)
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)


        if use_conditions:
            self._inject_conditions()


        self.prop_stats: Dict = {}
        if prop_stats_path and os.path.exists(prop_stats_path):
            self.prop_stats = torch.load(prop_stats_path, map_location='cpu')
            print(f"[Dataset:{name}] 从文件加载prop_stats: {prop_stats_path}")
        elif use_conditions:

            self.prop_stats = compute_property_statistics(
                self.cached_data,
                self.prop_keys,
                log_transform_props=self.log_transform_props,
            )


        self.lattice_scaler = None
        self.scaler         = None


    def _preprocess(self, save_path, preprocess_workers, prop):

        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli          = self.niggli,
                primitive       = self.primitive,
                graph_method    = self.graph_method,
                prop_list       = [prop],
                use_space_group = self.use_space_group,
                tol             = self.tolerance,
            )
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def _inject_conditions(self):


        available_keys = []
        for key in self.prop_keys:
            if key in self.df.columns:
                available_keys.append(key)
            else:
                print(f"[Dataset:{self.name}] ⚠️  CSV中未找到列: {key}")

        n_injected = 0
        for i, row in self.df.iterrows():
            if i >= len(self.cached_data):
                break
            d = self.cached_data[i]

            for key in available_keys:
                val = row.get(key, None)
                if pd.isna(val):
                    d[key] = None
                else:
                    d[key] = float(val)


            elements_raw = row.get('elements', '')
            d['chemical_system'] = parse_elements_to_chemsys(
                str(elements_raw) if not pd.isna(elements_raw) else ''
            )
            n_injected += 1

        print(f"[Dataset:{self.name}] 属性注入完成: "
              f"{n_injected}条, 属性={available_keys}")

    def set_property_stats(self, stats: Dict):


        self.prop_stats = stats


    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index: int) -> Data:
        data_dict = self.cached_data[index]


        prop_val = self.scaler.transform(data_dict[self.prop])

        (frac_coords, atom_types, lengths, angles,
         edge_indices, to_jimages, num_atoms) = data_dict['graph_arrays']

        data = Data(
            frac_coords = torch.Tensor(frac_coords),
            atom_types  = torch.LongTensor(atom_types),
            lengths     = torch.Tensor(lengths).view(1, -1),
            angles      = torch.Tensor(angles).view(1, -1),
            edge_index  = torch.LongTensor(edge_indices.T).contiguous(),
            to_jimages  = torch.LongTensor(to_jimages),
            num_atoms   = num_atoms,
            num_bonds   = edge_indices.shape[0],
            num_nodes   = num_atoms,
            y           = prop_val.view(1, -1),
        )

        if self.use_space_group:
            data.spacegroup   = torch.LongTensor([data_dict['spacegroup']])
            data.ops          = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic, indexes = {}, []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)


        if self.use_conditions:
            if not self.prop_stats:


                for key in self.prop_keys:
                    setattr(data, key,
                            torch.tensor([0.0], dtype=torch.float32))
                    setattr(data, f"{key}_missing",
                            torch.tensor([1.0], dtype=torch.float32))
            else:
                for key in self.prop_keys:
                    raw_val = data_dict.get(key, None)
                    stats   = self.prop_stats.get(
                        key, {"mean": 0.0, "std": 1.0})

                    norm_v, missing = normalize_property(
                        raw_val,
                        mean       = stats["mean"],
                        std        = stats["std"],
                        clip_range = 5.0,
                        log_transform=bool(
                            stats.get("log_transformed", False)
                            or (self.log_transform_props is not None
                                and key in self.log_transform_props)
                        ),
                    )

                    setattr(data, key,
                            torch.tensor([norm_v], dtype=torch.float32))
                    setattr(data, f"{key}_missing",
                            torch.tensor([float(missing)], dtype=torch.float32))

            data.chemical_system = data_dict.get('chemical_system', 'Unknown')

        return data

    def __repr__(self) -> str:
        return (f"ConditionalCrystDataset("
                f"name={self.name}, "
                f"size={len(self)}, "
                f"props={self.prop_keys})")
