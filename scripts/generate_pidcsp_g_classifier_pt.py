import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chemparse
import numpy as np
import pandas as pd
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from eval_utils import get_crystals_list, lattices_to_params_shape, load_model


chemical_symbols = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
]
_SYMBOL_TO_IDX = {s: i for i, s in enumerate(chemical_symbols)}

TARGET_PROPS = ["formation_energy_per_atom", "band_gap"]


class DiskCache:
    def __init__(self, base_dir: str, prop_name: str):
        self.cache_dir = Path(base_dir) / ".sampling_cache" / prop_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cached_path(self, row_idx: int) -> Path:
        return self.cache_dir / f"record_{row_idx}.npz"

    def is_done(self, row_idx: int) -> bool:
        return self.get_cached_path(row_idx).exists()

    def save(self, row_idx: int, crystal: dict, formula: str, target_val: float):

        save_dict = {
            "formula": str(formula),
            "target_val": float(target_val),
            "row_idx": int(row_idx),
            "lengths": np.array(crystal["lengths"], dtype=np.float32),
            "angles": np.array(crystal["angles"], dtype=np.float32),
            "frac_coords": np.array(crystal["frac_coords"], dtype=np.float32),
            "atom_types": np.array(crystal["atom_types"], dtype=np.int64),
        }
        np.savez_compressed(self.get_cached_path(row_idx), **save_dict)

    def load_all_cached(self, expected_indices: List[int]) -> Tuple[List[dict], List[str], List[float], List[int]]:

        crystals, formulas, target_vals, row_indices = [], [], [], []
        print(f"  [Cache] 正在从本地缓存组装结果...")

        for idx in expected_indices:
            path = self.get_cached_path(idx)
            if path.exists():
                try:
                    data = np.load(path, allow_pickle=True)
                    c = {
                        "lengths": data["lengths"],
                        "angles": data["angles"],
                        "frac_coords": data["frac_coords"],
                        "atom_types": data["atom_types"],
                    }
                    crystals.append(c)
                    formulas.append(str(data["formula"]))
                    target_vals.append(float(data["target_val"]))
                    row_indices.append(int(data["row_idx"]))
                except Exception as e:
                    print(f"  [WARN] 无法读取缓存文件 {path}: {e}")
        return crystals, formulas, target_vals, row_indices


def is_lattice_physical(lengths: np.ndarray, angles: np.ndarray) -> bool:
    if np.any(np.isnan(lengths)) or np.any(np.isnan(angles)): return False
    if np.any(np.isinf(lengths)) or np.any(np.isinf(angles)): return False
    if np.any(lengths <= 0.5) or np.any(lengths >= 100.0): return False
    if np.any(angles <= 5.0) or np.any(angles >= 175.0): return False
    try:
        lat = Lattice.from_parameters(*lengths, *angles)
        return lat.volume > 0.1
    except Exception:
        return False


def validate_crystal(crystal: dict) -> bool:
    lengths     = np.array(crystal["lengths"], dtype=np.float32)
    angles      = np.array(crystal["angles"], dtype=np.float32)
    frac_coords = np.array(crystal["frac_coords"], dtype=np.float32)
    atom_types  = np.array(crystal["atom_types"], dtype=np.int64)

    if not is_lattice_physical(lengths, angles): return False
    if np.any(np.isnan(frac_coords)) or np.any(np.isinf(frac_coords)): return False

    try:
        at_clamped = np.clip(atom_types, 1, len(chemical_symbols) - 1)
        species    = [chemical_symbols[i] for i in at_clamped]
        struct     = Structure(
            lattice=Lattice.from_parameters(*lengths, *angles),
            species=species, coords=frac_coords, coords_are_cartesian=False
        )
        dm = struct.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        if dm.min() < 0.5: return False
    except Exception:
        return False
    return True


def formula_to_atom_list(formula: str) -> List[int]:
    composition = chemparse.parse_formula(formula)
    atom_list = []
    for elem, count in sorted(composition.items()):
        if elem not in _SYMBOL_TO_IDX:
            raise ValueError(f"未知元素 '{elem}'")
        atom_list.extend([_SYMBOL_TO_IDX[elem]] * int(round(count)))
    return atom_list


def fix_atom_types(at: torch.Tensor, atom_list: List[int], num_atoms: torch.Tensor) -> torch.Tensor:
    if at.dtype in (torch.float16, torch.float32, torch.float64):
        at = at.argmax(dim=-1) if at.dim() == 2 else at.round().long()
    corrected = torch.zeros_like(at.long())
    expected = torch.LongTensor(atom_list)
    start = 0
    for n in num_atoms.tolist():
        n = int(n)
        corrected[start:start+n] = expected if n == len(atom_list) else at[start:start+n]
        start += n
    return corrected


class SingleFormulaDataset(Dataset):
    def __init__(self, atom_list: List[int], n: int = 1):
        self.atom_list, self.n = atom_list, n

    def __len__(self): return self.n
    def __getitem__(self, _):
        return Data(atom_types=torch.LongTensor(self.atom_list), num_atoms=len(self.atom_list), num_nodes=len(self.atom_list))


def sample_one_batch(sampler, atom_list: List[int], target_val: float, prop_name: str, batch_size: int, device: str) -> List[Optional[dict]]:
    dataset = SingleFormulaDataset(atom_list, n=batch_size)
    batch = Batch.from_data_list([dataset[i] for i in range(batch_size)]).to(device)

    final, _ = sampler.sample(batch=batch, target_props={prop_name: target_val}, device=device)

    fc  = final["frac_coords"].detach().cpu()
    na  = final["num_atoms"].detach().cpu()
    at  = final["atom_types"].detach().cpu()
    lat = final["lattices"].detach().cpu()

    at_fixed = fix_atom_types(at, atom_list, na)
    lengths, angles = lattices_to_params_shape(lat)
    crystal_list = get_crystals_list(fc, at_fixed, lengths, angles, na)

    return [c if (c is not None and validate_crystal(c)) else None for c in crystal_list]


def sample_one_record(sampler, atom_list: List[int], target_val: float, prop_name: str, device: str, retry_batch: int = 4, max_rounds: int = 20) -> Optional[dict]:
    for _ in range(max_rounds):
        try:
            for c in sample_one_batch(sampler, atom_list, target_val, prop_name, retry_batch, device):
                if c is not None: return c
        except Exception:
            continue


    try:
        res = sample_one_batch(sampler, atom_list, target_val, prop_name, 1, device)
        if res and res[0]: return res[0]
    except Exception: pass
    return None


def pack_and_save_pt(crystals: List[dict], formulas: List[str], target_vals: List[float], row_indices: List[int], pt_path: str):
    valid = [(c, f, v, r) for c, f, v, r in zip(crystals, formulas, target_vals, row_indices) if c is not None]
    if not valid:
        print("  [ERROR] 无有效结构，未保存 PT 文件。")
        return

    crystals_v, formulas_v, tvals_v, ridx_v = zip(*valid)


    num_atoms_list = [len(c["atom_types"]) for c in crystals_v]

    pt_data = {
        "num_atoms": torch.tensor(num_atoms_list, dtype=torch.long),
        "lengths": torch.from_numpy(np.stack([np.array(c["lengths"], dtype=np.float32) for c in crystals_v])),
        "angles": torch.from_numpy(np.stack([np.array(c["angles"], dtype=np.float32) for c in crystals_v])),
        "frac_coords": torch.from_numpy(np.concatenate([np.array(c["frac_coords"], dtype=np.float32) for c in crystals_v], axis=0)),
        "atom_types": torch.from_numpy(np.concatenate([np.array(c["atom_types"], dtype=np.int64) for c in crystals_v], axis=0)).long(),
        "target_vals": torch.tensor(list(tvals_v), dtype=torch.float32),
        "formulas": list(formulas_v),
        "row_indices": torch.tensor(list(ridx_v), dtype=torch.long),
    }

    os.makedirs(os.path.dirname(os.path.abspath(pt_path)), exist_ok=True)
    torch.save(pt_data, pt_path)

    print(f"\n  [PT 已成功保存] {pt_path}")
    print(f"  包含结构数: {len(crystals_v)} | 总原子数: {sum(num_atoms_list)}")


def load_classifier(ckpt_path: str, device: str):
    from pidcsp.pl_modules.classifier.pidcsp_g_classifier import CrystalClassifier
    phys_data_path = str(Path(__file__).resolve().parent.parent / "embeddings" / "phys_embedding_data.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    clf = CrystalClassifier(
        hidden_dim=ckpt.get("hidden_dim", 512), time_dim=ckpt.get("time_dim", 256),
        num_freqs=ckpt.get("num_freqs", 128), edge_style=ckpt.get("edge_style", "fc"),
        clf_hidden_dim=ckpt.get("clf_hidden_dim", 256), n_mp_layers=ckpt.get("n_mp_layers", 3),
        phys_data_path=phys_data_path, num_heads=ckpt.get("num_heads", 4), prop_keys=ckpt.get("prop_keys", TARGET_PROPS),
    )
    clf.load_state_dict(ckpt["classifier_state_dict"])
    return clf.to(device).eval(), ckpt.get("prop_stats", {})


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"\n{'='*60}\n  测试集分类器引导扩散采样 -> PT文件 (终极安全版)\n{'='*60}")

    model, _, _ = load_model(Path(args.model_path), load_data=False)
    model.to(device).eval()

    from pidcsp.pl_modules.classifier.pidcsp_g_guidance import ClassifierGuidanceSampler
    classifier, clf_prop_stats = load_classifier(args.classifier_ckpt, device)
    sampler = ClassifierGuidanceSampler(
        backbone=model, classifier=classifier, prop_stats=clf_prop_stats,
        guidance_scale_lattice=args.guidance_scale_lattice, guidance_scale_coord=args.guidance_scale_coord,
        grad_clip_lattice=args.grad_clip_lattice, grad_clip_coord=args.grad_clip_coord,
        t_guidance_end_ratio=args.t_guidance_end_ratio, step_lr=args.step_lr, verbose=False,
    )

    df = pd.read_csv(args.test_csv)
    formula_col = args.formula_col if args.formula_col in df.columns else "formula"

    for prop_name in args.props:
        if prop_name not in df.columns: continue
        print(f"\n{'='*40}\n  开始处理属性: {prop_name}\n{'='*40}")

        disk_cache = DiskCache(args.save_path, prop_name)

        sub = df[[formula_col, prop_name]].copy().dropna()
        sub.columns = ["formula", "target_val"]
        sub["row_idx"] = sub.index

        all_indices = sub["row_idx"].tolist()
        pending_records = [r for r in sub.to_dict("records") if not disk_cache.is_done(r["row_idx"])]

        print(f"  总记录数: {len(all_indices)} | 已缓存: {len(all_indices) - len(pending_records)} | 本次待生成: {len(pending_records)}")


        if pending_records:
            for rec in tqdm(pending_records, desc=f"  {prop_name}"):
                try:
                    atom_list = formula_to_atom_list(str(rec["formula"]))
                    crystal = sample_one_record(sampler, atom_list, float(rec["target_val"]), prop_name, device, args.retry_batch, args.max_rounds)
                    if crystal is not None:
                        disk_cache.save(rec["row_idx"], crystal, rec["formula"], rec["target_val"])
                except Exception as e:
                    print(f"\n  [ERROR] 记录 {rec['row_idx']} ({rec['formula']}) 生成失败: {e}")


        crystals, formulas, target_vals, row_indices = disk_cache.load_all_cached(all_indices)
        pack_and_save_pt(crystals, formulas, target_vals, row_indices, os.path.join(args.save_path, f"{prop_name}_gen.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--classifier_ckpt", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--props", nargs="+", default=["formation_energy_per_atom", "band_gap"])
    p.add_argument("--formula_col", default="pretty_formula")
    p.add_argument("--guidance_scale_lattice", type=float, default=3.0)
    p.add_argument("--guidance_scale_coord", type=float, default=1.5)
    p.add_argument("--grad_clip_lattice", type=float, default=0.3)
    p.add_argument("--grad_clip_coord", type=float, default=0.05)
    p.add_argument("--t_guidance_end_ratio", type=float, default=0.8)
    p.add_argument("--step_lr", type=float, default=1e-5)
    p.add_argument("--retry_batch", type=int, default=4)
    p.add_argument("--max_rounds", type=int, default=20)
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
