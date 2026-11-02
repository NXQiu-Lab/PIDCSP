import argparse
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import chemparse
import numpy as np
import pandas as pd
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list


chemical_symbols = [
    "X",
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu",
]
_SYMBOL_TO_IDX = {s: i for i, s in enumerate(chemical_symbols)}

TARGET_PROPS = [
    "formation_energy_per_atom",
    "band_gap",
]


def load_cgcnn_supported_elements(atom_init_path: str) -> set:


    if not atom_init_path or not os.path.exists(atom_init_path):

        print("  [WARN] atom_init.json 未指定或不存在，跳过 CGCNN 元素过滤")
        return None

    with open(atom_init_path, "r") as f:
        import json
        data = json.load(f)

    supported = {int(k) for k in data.keys()}
    print(f"  [CGCNN] atom_init.json 加载完成，支持 {len(supported)} 种元素序数")
    return supported


def is_cgcnn_safe(formula: str, supported_atomic_nums: set) -> bool:


    if supported_atomic_nums is None:
        return True

    try:
        composition = chemparse.parse_formula(formula)
    except Exception:
        return False

    for elem in composition:
        if elem not in _SYMBOL_TO_IDX:
            return False
        atomic_num = _SYMBOL_TO_IDX[elem]


        if atomic_num not in supported_atomic_nums:
            return False

    return True


def is_formula_parseable(formula: str) -> bool:

    try:
        composition = chemparse.parse_formula(formula)
        if not composition:
            return False
        for elem in composition:
            if elem not in _SYMBOL_TO_IDX:
                return False
        return True
    except Exception:
        return False


def sample_formula_target_pairs(
    csv_path:              str,
    n_targets:             int   = 100,
    q_low:                 float = 0.05,
    q_high:                float = 0.95,
    formula_col:           str   = "pretty_formula",
    seed:                  int   = 42,
    cgcnn_supported_elems: set   = None,
    target_props:          list   = None,
) -> dict:


    rng = np.random.default_rng(seed)
    df  = pd.read_csv(csv_path)
    print(f"\n[CSV] 读取测试集: {csv_path}，共 {len(df)} 行")


    if formula_col not in df.columns:
        for alt in ["formula", "composition", "material_id", "pretty_formula"]:
            if alt in df.columns:
                formula_col = alt
                print(f"  [INFO] 使用列 '{formula_col}' 作为化学式")
                break
        else:
            raise ValueError(
                f"未找到化学式列，已有列: {list(df.columns)}"
            )

    result = {}

    for prop in (target_props or TARGET_PROPS):
        if prop not in df.columns:
            raise ValueError(f"CSV中未找到属性列: {prop}")

        sub = df[[formula_col, prop]].dropna().copy()
        sub.columns = ["formula", "val"]
        n_before = len(sub)


        mask_safe = sub["formula"].apply(
            lambda f: (
                is_formula_parseable(f)
                and is_cgcnn_safe(f, cgcnn_supported_elems)
            )
        )
        sub = sub[mask_safe].reset_index(drop=True)
        n_after  = len(sub)
        n_drop   = n_before - n_after

        print(f"\n  [{prop}]")
        if n_drop > 0:
            print(f"    CGCNN过滤: 剔除 {n_drop} 条 "
                  f"({n_drop/n_before*100:.1f}%)，剩余 {n_after} 条")


        lo = float(sub["val"].quantile(q_low))
        hi = float(sub["val"].quantile(q_high))
        sub = sub[(sub["val"] >= lo) & (sub["val"] <= hi)].reset_index(drop=True)

        print(f"    Q{int(q_low*100)}%={lo:.4f},  Q{int(q_high*100)}%={hi:.4f}")
        print(f"    分位数范围内有效样本数: {len(sub)}")

        if len(sub) < n_targets:
            print(f"    [WARN] 样本数({len(sub)}) < n_targets({n_targets})，"
                  f"将重复采样")

        bins     = np.linspace(lo, hi, n_targets + 1)
        pairs    = []
        used_idx = set()

        for tidx in range(n_targets):
            bin_lo = bins[tidx]
            bin_hi = bins[tidx + 1]

            mask       = (sub["val"] >= bin_lo) & (sub["val"] <= bin_hi)
            candidates = sub[mask & ~sub.index.isin(used_idx)]

            if len(candidates) == 0:
                remaining = sub[~sub.index.isin(used_idx)]
                if len(remaining) == 0:
                    remaining = sub
                bin_center  = (bin_lo + bin_hi) / 2
                closest_idx = (remaining["val"] - bin_center).abs().idxmin()
                row = remaining.loc[closest_idx]
            else:
                row = candidates.sample(
                    1, random_state=int(rng.integers(1e6))
                ).iloc[0]

            formula    = str(row["formula"])
            target_val = float(row["val"])
            used_idx.add(row.name if hasattr(row, "name") else -1)

            pairs.append({
                "tidx":       tidx,
                "formula":    formula,
                "target_val": target_val,
            })


        for p_item in pairs[:5]:
            sign = "+" if p_item["target_val"] >= 0 else ""
            print(f"    t{p_item['tidx']:03d}  "
                  f"{p_item['formula']:<22s}  "
                  f"val={sign}{p_item['target_val']:.4f}")
        if len(pairs) > 10:
            print(f"    ... (共 {len(pairs)} 个目标) ...")
        for p_item in pairs[-5:]:
            sign = "+" if p_item["target_val"] >= 0 else ""
            print(f"    t{p_item['tidx']:03d}  "
                  f"{p_item['formula']:<22s}  "
                  f"val={sign}{p_item['target_val']:.4f}")

        result[prop] = pairs

    return result


class ConditionalSampleDataset(Dataset):
    def __init__(self, formula: str, num_evals: int):
        super().__init__()
        self.formula   = formula
        self.num_evals = num_evals

        composition    = chemparse.parse_formula(formula)
        self.chem_list = []
        for elem, count in sorted(composition.items()):
            if elem not in _SYMBOL_TO_IDX:
                raise ValueError(f"元素 '{elem}' 不在 chemical_symbols 中")
            self.chem_list.extend([_SYMBOL_TO_IDX[elem]] * int(count))

    def __len__(self):
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )


def fix_atom_types(at_output, fixed_types, num_atoms):
    at = at_output
    if at.dtype in (torch.float16, torch.float32, torch.float64):
        at = at.argmax(dim=-1) if at.dim() == 2 else at.round().long()
    at = at.long()

    expected  = torch.LongTensor(fixed_types)
    n_per     = len(fixed_types)
    corrected = torch.zeros_like(at)
    start     = 0
    for n in num_atoms.tolist():
        n = int(n)
        corrected[start:start + n] = (
            expected if n == n_per else at[start:start + n]
        )
        start += n
    return corrected


def is_lattice_physical(lengths, angles) -> bool:
    if np.any(np.isnan(lengths)) or np.any(np.isnan(angles)):
        return False
    if np.any(np.isinf(lengths)) or np.any(np.isinf(angles)):
        return False
    if np.any(lengths <= 0.5) or np.any(lengths >= 100.0):
        return False
    if np.any(angles <= 5.0) or np.any(angles >= 175.0):
        return False
    try:
        lat = Lattice.from_parameters(
            lengths[0], lengths[1], lengths[2],
            angles[0],  angles[1],  angles[2],
        )
        return lat.volume > 0.1
    except Exception:
        return False


def crystal_to_pymatgen(crystal_array: dict) -> Structure:
    lengths     = crystal_array["lengths"]
    angles      = crystal_array["angles"]
    atom_types  = crystal_array["atom_types"]
    frac_coords = crystal_array["frac_coords"]

    if not is_lattice_physical(lengths, angles):
        raise ValueError(
            f"非物理晶格: a/b/c={np.round(lengths, 2).tolist()}"
        )

    at_int  = np.clip(np.array(atom_types, dtype=np.int64),
                      1, len(chemical_symbols) - 1)
    species = [chemical_symbols[i] for i in at_int]

    structure = Structure(
        lattice=Lattice.from_parameters(
            lengths[0], lengths[1], lengths[2],
            angles[0],  angles[1],  angles[2],
        ),
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
    )

    dm = structure.distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    if dm.min() < 0.5:
        raise ValueError(f"原子间距过近: {dm.min():.3f} Å")

    return structure


def outputs_to_structures(
    frac_coords: torch.Tensor,
    atom_types:  torch.Tensor,
    lattices:    torch.Tensor,
    num_atoms:   torch.Tensor,
    chem_list:   list,
) -> list:

    at_fixed        = fix_atom_types(atom_types, chem_list, num_atoms)
    lengths, angles = lattices_to_params_shape(lattices)
    crystal_list    = get_crystals_list(
        frac_coords, at_fixed, lengths, angles, num_atoms
    )
    structures = []
    for c in crystal_list:
        try:
            structures.append(crystal_to_pymatgen(c))
        except ValueError:
            structures.append(None)
    return structures


def load_classifier(ckpt_path: str, device):


    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone_type = ckpt.get("backbone_type", "physics")

    if backbone_type == "vanilla_pidcsp_c":
        from pidcsp.pl_modules.classifier.baseline_classifier import CrystalClassifierBase
        clf = CrystalClassifierBase(
            hidden_dim     = ckpt.get("hidden_dim",     512),
            time_dim       = ckpt.get("time_dim",       256),
            num_freqs      = ckpt.get("num_freqs",      128),
            edge_style     = ckpt.get("edge_style",     "fc"),
            ln             = ckpt.get("ln",              True),
            ip             = ckpt.get("ip",              True),
            clf_hidden_dim = ckpt.get("clf_hidden_dim", 256),
            n_mp_layers    = ckpt.get("n_mp_layers",    3),
            prop_keys      = ckpt.get("prop_keys", TARGET_PROPS),
            log_transform_props = ckpt.get("log_transform_props", None),
            use_max_features = ckpt.get("use_max_features", False),
            include_b_as_x = ckpt.get("include_b_as_x", False),
            max_role_feature_dim = ckpt.get("max_role_feature_dim", 12),
        )
        print(f"\n[Classifier] 识别为无物理增强变体 (CrystalClassifierBase)")
    else:
        from pidcsp.pl_modules.classifier.pidcsp_g_classifier import CrystalClassifier
        project_root   = Path(__file__).resolve().parent.parent
        phys_data_path = str(project_root / "embeddings" / "phys_embedding_data.pt")
        clf = CrystalClassifier(
            hidden_dim     = ckpt.get("hidden_dim",     512),
            time_dim       = ckpt.get("time_dim",       256),
            num_freqs      = ckpt.get("num_freqs",      128),
            edge_style     = ckpt.get("edge_style",     "fc"),
            clf_hidden_dim = ckpt.get("clf_hidden_dim", 256),
            n_mp_layers    = ckpt.get("n_mp_layers",    3),
            phys_data_path = ckpt.get("phys_data_path", phys_data_path) or phys_data_path,
            num_heads      = ckpt.get("num_heads",      4),
            prop_keys      = ckpt.get("prop_keys", TARGET_PROPS),
            log_transform_props = ckpt.get("log_transform_props", None),
            use_max_features = ckpt.get("use_max_features", False),
            include_b_as_x = ckpt.get("include_b_as_x", False),
            max_role_feature_dim = ckpt.get("max_role_feature_dim", 12),
        )
        print(f"\n[Classifier] 识别为物理增强变体 (CrystalClassifier)")

    clf.load_state_dict(ckpt["classifier_state_dict"])
    clf.to(device).eval()

    prop_stats = ckpt.get("prop_stats", {})
    print(f"[Classifier] 加载成功: {ckpt_path}")
    if prop_stats:
        for k, s in prop_stats.items():
            log_flag = " [log空间]" if s.get("log_transformed", False) else ""
            print(f"  {k}{log_flag}: mean={s['mean']:.4f}, std={s['std']:.4f}")
    else:
        print("  ⚠️  prop_stats 为空，目标属性将不归一化")

    clf_prop_keys = clf.prop_keys if hasattr(clf, "prop_keys") else []
    for prop in TARGET_PROPS:
        if prop not in clf_prop_keys:
            print(f"  ⚠️  分类器不支持属性 '{prop}'")

    return clf, prop_stats


def generate_classifier_guided(
    sampler,
    prop_name:  str,
    formula:    str,
    prop_val:   float,
    n_samples:  int,
    batch_size: int,
    device,
) -> list:


    try:
        dataset = ConditionalSampleDataset(formula, n_samples)
    except ValueError as e:
        print(f"    [ERROR] 化学式解析失败 {formula}: {e}")
        return [None] * n_samples

    clf_prop_keys = (
        sampler.classifier.prop_keys
        if hasattr(sampler.classifier, "prop_keys")
        else []
    )
    if prop_name in clf_prop_keys:
        target_props = {prop_name: prop_val}
    else:
        print(f"    [WARN] 分类器不支持 '{prop_name}'，退化为无引导纯扩散")
        target_props = {}

    all_fc  = []
    all_at  = []
    all_lat = []
    all_na  = []

    for start in range(0, n_samples, batch_size):
        end       = min(start + batch_size, n_samples)
        data_list = [dataset[i] for i in range(start, end)]
        batch     = Batch.from_data_list(data_list).to(device)


        final, _ = sampler.sample(
            batch        = batch,
            target_props = target_props,
            device       = device,
        )

        all_fc.append(final["frac_coords"].detach().cpu())
        all_at.append(final["atom_types"].detach().cpu())
        all_lat.append(final["lattices"].detach().cpu())
        all_na.append(final["num_atoms"].detach().cpu())

    frac_coords = torch.cat(all_fc,  dim=0)
    atom_types  = torch.cat(all_at,  dim=0)
    lattices    = torch.cat(all_lat, dim=0)
    num_atoms   = torch.cat(all_na,  dim=0)

    return outputs_to_structures(
        frac_coords, atom_types, lattices, num_atoms, dataset.chem_list
    )


def main(args):


    model, _, cfg = load_model(Path(args.model_path), load_data=False)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model.to(device)
    model.eval()
    print(f"\n[Model] 加载完成，设备: {device}")


    cgcnn_supported_elems = load_cgcnn_supported_elements(args.atom_init)


    from pidcsp.pl_modules.classifier.pidcsp_g_guidance import (
        ClassifierGuidanceSampler,
    )

    classifier, clf_prop_stats = load_classifier(args.classifier_ckpt, device)
    target_props = [
        value.strip() for value in args.properties.split(',') if value.strip()
    ] if args.properties else list(getattr(classifier, 'prop_keys', TARGET_PROPS))

    sampler = ClassifierGuidanceSampler(
        backbone               = model,
        classifier             = classifier,
        prop_stats             = clf_prop_stats,
        guidance_scale_lattice = args.guidance_scale_lattice,
        guidance_scale_coord   = args.guidance_scale_coord,
        grad_clip_lattice      = args.grad_clip_lattice,
        grad_clip_coord        = args.grad_clip_coord,
        t_guidance_end_ratio   = args.t_guidance_end_ratio,
        step_lr                = args.step_lr,
        log_transform_props    = getattr(classifier, 'log_transform_props', None),
        verbose                = False,
    )
    print(f"[Sampler] ClassifierGuidanceSampler 构建完成")
    print(f"  guidance_scale_lattice = {args.guidance_scale_lattice}")
    print(f"  guidance_scale_coord   = {args.guidance_scale_coord}")
    print(f"  n_targets              = {args.n_targets}")
    print(f"  n_samples              = {args.n_samples}")


    pairs_dict = sample_formula_target_pairs(
        csv_path              = args.test_csv,
        n_targets             = args.n_targets,
        q_low                 = args.q_low,
        q_high                = args.q_high,
        formula_col           = args.formula_col,
        seed                  = args.seed,
        cgcnn_supported_elems = cgcnn_supported_elems,
        target_props          = target_props,
    )


    os.makedirs(args.save_path, exist_ok=True)

    with open(os.path.join(args.save_path, "sampled_pairs.json"),
              "w", encoding="utf-8") as f:
        json.dump(pairs_dict, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.save_path, "run_config.json"),
              "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"\n[Info] 采样对信息已保存: {args.save_path}/sampled_pairs.json")


    summary = {}

    for prop_name in target_props:
        prop_dir = os.path.join(args.save_path, prop_name)
        os.makedirs(prop_dir, exist_ok=True)

        pairs = pairs_dict[prop_name]
        print(f"\n{'='*60}")
        print(f"[Prop] {prop_name}  ({len(pairs)} 个目标对，"
              f"每个生成 {args.n_samples} 个候选)")
        print(f"{'='*60}")

        prop_meta   = []
        total_saved = 0

        for pair in tqdm(pairs, desc=prop_name[:28], ncols=90):
            tidx    = pair["tidx"]
            formula = pair["formula"]
            tval    = pair["target_val"]
            sign    = "+" if tval >= 0 else ""

            try:
                struct_list = generate_classifier_guided(
                    sampler    = sampler,
                    prop_name  = prop_name,
                    formula    = formula,
                    prop_val   = tval,
                    n_samples  = args.n_samples,
                    batch_size = args.batch_size,
                    device     = device,
                )
            except Exception as e:
                print(f"\n  t{tidx:03d} ✗ 生成失败: {e}")
                traceback.print_exc()
                struct_list = [None] * args.n_samples


            n_success = 0
            for sidx, s in enumerate(struct_list):
                if s is None:
                    continue
                fname = (
                    f"{formula}"
                    f"_t{tidx:03d}"
                    f"_s{sidx + 1:02d}"
                    f"_val{sign}{tval:.4f}"
                    f".cif"
                )
                fpath = os.path.join(prop_dir, fname)
                try:
                    CifWriter(s).write_file(fpath)
                    n_success += 1
                    total_saved += 1
                except Exception as e:
                    pass

            prop_meta.append({
                "tidx":       tidx,
                "formula":    formula,
                "target_val": tval,
                "n_samples":  args.n_samples,
                "n_success":  n_success,
            })


        meta_out = {
            "prop_name":   prop_name,
            "n_targets":   len(pairs),
            "n_samples":   args.n_samples,
            "total_saved": total_saved,
            "targets":     prop_meta,
        }
        with open(os.path.join(prop_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        n_ok = sum(1 for m in prop_meta if m["n_success"] > 0)
        print(f"\n[{prop_name}] 完成")
        print(f"  目标数: {len(pairs)}，有效目标(≥1个结构): {n_ok}")
        print(f"  总保存结构数: {total_saved} / {len(pairs)*args.n_samples}")
        summary[prop_name] = meta_out

    with open(os.path.join(args.save_path, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] 根目录: {args.save_path}")
    print(f"下一步运行 step1_cif2pt.py → step2_cgcnn_predict.py → step3_compute_metrics.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )


    p.add_argument("--model_path",      required=True,
                   help="扩散模型 checkpoint 路径")
    p.add_argument("--classifier_ckpt", required=True,
                   help="分类器 checkpoint 路径（.pt）")
    p.add_argument("--test_csv",        required=True,
                   help="测试集 CSV 路径")
    p.add_argument("--properties",     default="",
                   help="逗号分隔的目标属性；为空时读取分类器checkpoint中的prop_keys")
    p.add_argument("--save_path",       required=True,
                   help="结果保存根目录")


    p.add_argument("--atom_init", default="",
                   help="CGCNN atom_init.json 路径，用于预过滤不支持的化学式"
                        "（留空则不过滤）")


    p.add_argument("--formula_col", default="pretty_formula",
                   help="CSV 中化学式列名")
    p.add_argument("--n_targets",   type=int,   default=100,
                   help="每个属性采样的目标(formula,val)对数量")
    p.add_argument("--n_samples",   type=int,   default=10,
                   help="每个目标对生成的候选结构数")
    p.add_argument("--batch_size",  type=int,   default=10,
                   help="生成时的批大小（控制显存）")
    p.add_argument("--q_low",       type=float, default=0.05,
                   help="属性分位数下界")
    p.add_argument("--q_high",      type=float, default=0.95,
                   help="属性分位数上界")
    p.add_argument("--seed",        type=int,   default=42)


    p.add_argument("--guidance_scale_lattice", type=float, default=3.0)
    p.add_argument("--guidance_scale_coord",   type=float, default=1.5)
    p.add_argument("--grad_clip_lattice",      type=float, default=0.3)
    p.add_argument("--grad_clip_coord",        type=float, default=0.05)
    p.add_argument("--t_guidance_end_ratio",   type=float, default=0.8)
    p.add_argument("--step_lr",                type=float, default=1e-5)


    p.add_argument("--cpu", action="store_true",
                   help="强制使用 CPU")

    main(p.parse_args())
