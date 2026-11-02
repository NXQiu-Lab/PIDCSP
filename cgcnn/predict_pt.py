import argparse
import json
import os
import sys
import traceback
import warnings
import functools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")


from cgcnn.data import AtomCustomJSONInitializer, GaussianDistance, collate_pool
from cgcnn.model import CrystalGraphConvNet


class Normalizer:
    def __init__(self):
        self.mean = torch.tensor(0.0)
        self.std  = torch.tensor(1.0)

    def denorm(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.std + self.mean

    def load_state_dict(self, d: dict):
        self.mean = d["mean"]
        self.std  = d["std"]


class PTStructureDataset(Dataset):


    def __init__(
        self,
        pt_path:     str,
        atom_init:   str,
        radius:      float = 8.0,
        max_num_nbr: int   = 12,
        dmin:        float = 0.0,
        step:        float = 0.2,
    ):
        super().__init__()
        self.radius      = radius
        self.max_num_nbr = max_num_nbr


        if not os.path.exists(atom_init):
            raise FileNotFoundError(f"atom_init.json 不存在: {atom_init}")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"PT文件不存在: {pt_path}")


        self.ari = AtomCustomJSONInitializer(atom_init)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)


        sample_key = next(iter(self.ari.atom_types))
        print(f"  [atom_init] key类型: {type(sample_key).__name__}，"
              f"样例: {sorted(list(self.ari.atom_types))[:5]}")


        print(f"  [PTDataset] 加载 {pt_path} ...")
        raw = torch.load(pt_path, map_location="cpu")

        self._validate_pt_fields(raw)


        num_atoms_tensor = raw["num_atoms"]
        lengths_tensor   = raw["lengths"]
        angles_tensor    = raw["angles"]
        frac_coords_np   = raw["frac_coords"].numpy()
        atom_types_list  = raw["atom_types"].tolist()

        num_atoms_list   = num_atoms_tensor.tolist()
        lengths_list     = lengths_tensor.numpy().tolist()
        angles_list      = angles_tensor.numpy().tolist()

        self.target_vals  = raw["target_vals"].tolist()
        self.formulas     = raw["formulas"]
        self.row_indices  = raw["row_indices"].tolist()


        print(f"  [PTDataset] 重建 Structure...")
        struct_list = []
        atom_offset = 0

        for i in range(len(num_atoms_list)):
            atom_num  = int(num_atoms_list[i])
            length    = lengths_list[i]
            angle     = angles_list[i]
            atom_type = atom_types_list[atom_offset: atom_offset + atom_num]
            frac_coord = frac_coords_np[atom_offset: atom_offset + atom_num]

            try:
                lattice = Lattice.from_parameters(
                    a=length[0], b=length[1], c=length[2],
                    alpha=angle[0], beta=angle[1], gamma=angle[2],
                )
                structure = Structure(
                    lattice, atom_type, frac_coord.tolist(),
                    to_unit_cell=True
                )
                struct_list.append(structure)
            except Exception as e:

                print(f"  [WARN] Structure重建失败 idx={i}: {e}")
                struct_list.append(None)

            atom_offset += atom_num

        self.struct_list = struct_list
        self.train_id    = list(range(len(struct_list)))

        n_ok   = sum(1 for s in struct_list if s is not None)
        n_fail = len(struct_list) - n_ok
        print(f"  共 {len(struct_list)} 个结构"
              f"（重建成功={n_ok}，失败={n_fail}）")

    @staticmethod
    def _validate_pt_fields(raw: dict):

        required = [
            "num_atoms", "lengths", "angles",
            "frac_coords", "atom_types",
            "target_vals", "formulas", "row_indices",
        ]
        missing = [f for f in required if f not in raw]
        if missing:
            raise KeyError(
                f"PT文件缺少字段: {missing}\n"
                f"  已有字段: {list(raw.keys())}"
            )

    def __len__(self) -> int:
        return len(self.train_id)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx: int):
        crystal = self.struct_list[idx]

        if crystal is None:
            raise RuntimeError(
                f"结构 idx={idx} (formula={self.formulas[idx]}) 重建失败，"
                f"无法构建图特征"
            )


        try:
            atom_fea = np.vstack([
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ])
        except AssertionError as e:

            bad_nums = []
            for i in range(len(crystal)):
                z = crystal[i].specie.number
                if z not in self.ari.atom_types:
                    bad_nums.append(z)
            raise AssertionError(
                f"结构 idx={idx} formula={self.formulas[idx]} "
                f"含有 atom_init.json 中不存在的原子序数: {bad_nums}\n"
                f"  atom_init 支持的序数范围: "
                f"{min(self.ari.atom_types)}~{max(self.ari.atom_types)}"
            )


        all_nbrs = crystal.get_all_neighbors(
            self.radius, include_index=True
        )
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:

                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) +
                    [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr)) +
                    [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr[:self.max_num_nbr]))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr[:self.max_num_nbr]))
                )

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea     = np.array(nbr_fea)
        nbr_fea     = self.gdf.expand(nbr_fea)

        atom_fea    = torch.Tensor(atom_fea)
        nbr_fea     = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target      = torch.Tensor([float(self.target_vals[idx])])


        meta = json.dumps({
            "formula":   self.formulas[idx],
            "target_val": self.target_vals[idx],
            "row_idx":   int(self.row_indices[idx]),
        })

        return (atom_fea, nbr_fea, nbr_fea_idx), target, meta


def load_cgcnn(
    model_path: str,
    dataset:    PTStructureDataset,
    cuda:       bool,
) -> Tuple[CrystalGraphConvNet, Normalizer, argparse.Namespace]:

    ckpt       = torch.load(model_path, map_location=lambda s, l: s)
    model_args = argparse.Namespace(**ckpt["args"])

    normalizer = Normalizer()
    normalizer.load_state_dict(ckpt["normalizer"])


    sample_input, _, _ = dataset[0]
    orig_atom_fea_len  = sample_input[0].shape[-1]
    nbr_fea_len        = sample_input[1].shape[-1]

    print(f"  [CGCNN] orig_atom_fea_len={orig_atom_fea_len}, "
          f"nbr_fea_len={nbr_fea_len}")

    model = CrystalGraphConvNet(
        orig_atom_fea_len = orig_atom_fea_len,
        nbr_fea_len       = nbr_fea_len,
        atom_fea_len      = model_args.atom_fea_len,
        n_conv            = model_args.n_conv,
        h_fea_len         = model_args.h_fea_len,
        n_h               = model_args.n_h,
        classification    = (model_args.task == "classification"),
    )
    model.load_state_dict(ckpt["state_dict"])
    if cuda:
        model = model.cuda()
    model.eval()

    return model, normalizer, model_args


def run_prediction(
    dataset:    PTStructureDataset,
    model:      CrystalGraphConvNet,
    normalizer: Normalizer,
    model_args: argparse.Namespace,
    cuda:       bool,
    batch_size: int,
    workers:    int,
) -> List[Dict]:

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = workers,
        collate_fn  = collate_pool,
        pin_memory  = cuda,
    )

    all_results: List[Dict] = []
    error_count = 0

    with torch.no_grad():
        for batch_input, targets, meta_strs in tqdm(loader, desc="  CGCNN预测"):
            try:
                if cuda:
                    input_var = (
                        Variable(batch_input[0].cuda(non_blocking=True)),
                        Variable(batch_input[1].cuda(non_blocking=True)),
                        batch_input[2].cuda(non_blocking=True),
                        [x.cuda(non_blocking=True) for x in batch_input[3]],
                    )
                else:
                    input_var = (
                        Variable(batch_input[0]),
                        Variable(batch_input[1]),
                        batch_input[2],
                        batch_input[3],
                    )

                output = model(*input_var)

                if model_args.task == "regression":
                    preds = normalizer.denorm(
                        output.data.cpu()
                    ).view(-1).tolist()
                else:
                    preds = torch.exp(
                        output.data.cpu()
                    )[:, 1].tolist()

                for meta_str, tval, pred in zip(
                    meta_strs, targets.view(-1).tolist(), preds
                ):
                    meta = json.loads(meta_str)
                    all_results.append({
                        "formula":   meta["formula"],
                        "target_val": meta["target_val"],
                        "pred_val":  float(pred),
                        "row_idx":   meta["row_idx"],
                    })

            except Exception as e:
                error_count += 1
                print(f"\n  [WARN] 预测batch出错（已跳过）: {e}")
                if error_count <= 3:
                    traceback.print_exc()

    if error_count > 0:
        print(f"  [WARN] 共 {error_count} 个batch预测失败（已跳过）")

    return all_results


def save_csv(results: List[Dict], csv_path: str) -> pd.DataFrame:
    rows = [
        {
            "formula":   r["formula"],
            "target_val": round(r["target_val"], 6),
            "pred_val":  round(r["pred_val"],   6),
            "row_idx":   r["row_idx"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows, columns=[
        "formula", "target_val", "pred_val", "row_idx"
    ])
    df.to_csv(csv_path, index=False)
    print(f"  [CSV] 已保存: {csv_path}  ({len(df)} 条)")
    return df


def compute_metrics(
    df:            pd.DataFrame,
    sr_thresholds: List[float],
    prop_name:     str,
) -> dict:
    if len(df) == 0:
        print("  [WARN] 无有效数据")
        return {}

    targets = df["target_val"].values.astype(float)
    preds   = df["pred_val"].values.astype(float)
    err     = preds - targets
    abs_err = np.abs(err)

    mae  = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mse  = float(np.mean(err ** 2))
    bias = float(np.mean(err))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    sr = {}
    for tau in sr_thresholds:
        sr[f"SR@{tau}"] = float(np.mean(abs_err < tau))

    metrics = {
        "prop_name":    prop_name,
        "n_samples":    int(len(df)),
        "MAE":          mae,
        "MSE":          mse,
        "RMSE":         rmse,
        "R2":           r2,
        "bias":         bias,
        "target_range": [float(targets.min()), float(targets.max())],
        "pred_range":   [float(preds.min()),   float(preds.max())],
        "SR":           sr,
    }

    print(f"\n  ┌─ [{prop_name}] 评估指标 {'─'*30}")
    print(f"  │  样本数:      {metrics['n_samples']}")
    print(f"  │  MAE:         {mae:.5f}")
    print(f"  │  RMSE:        {rmse:.5f}")
    print(f"  │  MSE:         {mse:.5f}")
    print(f"  │  R²:          {r2:.5f}")
    print(f"  │  Bias:        {bias:+.5f}  "
          f"({'高估' if bias > 0 else '低估'})")
    print(f"  │  目标值范围:  [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"  │  预测值范围:  [{preds.min():.4f},  {preds.max():.4f}]")
    for k, v in sr.items():
        bar = "█" * int(v * 20)
        print(f"  │  {k}:   {v:.5f}  ({v*100:5.1f}%)  |{bar:<20}|")
    print(f"  └{'─'*55}")

    return metrics


def print_summary_table(
    all_metrics:   Dict[str, dict],
    sr_thresholds: List[float],
):
    sr_keys = [f"SR@{t}" for t in sr_thresholds]
    print(f"\n{'='*70}")
    print("  多属性汇总对比")
    print(f"{'='*70}")
    sr_hdr = "  ".join([f"{k:>8}" for k in sr_keys])
    hdr = (
        f"  {'属性':<35}  {'n':>6}  {'MAE':>8}  "
        f"{'RMSE':>8}  {'R²':>7}  {'Bias':>8}  {sr_hdr}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for prop_name, m in all_metrics.items():
        if not m:
            print(f"  {prop_name:<35}  {'N/A':>6}")
            continue
        sr_str = "  ".join([
            f"{m['SR'].get(k, 0.):>8.5f}" for k in sr_keys
        ])
        print(
            f"  {prop_name:<35}  "
            f"{m['n_samples']:>6}  "
            f"{m['MAE']:>8.5f}  "
            f"{m['RMSE']:>8.5f}  "
            f"{m['R2']:>7.4f}  "
            f"{m['bias']:>+8.5f}  "
            f"{sr_str}"
        )
    print(f"{'='*70}")


def main(args: argparse.Namespace):
    cuda = torch.cuda.is_available() and not args.cpu
    os.makedirs(args.save_path, exist_ok=True)


    pt_file_map: Dict[str, str] = {}
    for item in args.pt_files:
        prop, path = item.split(":", 1)
        pt_file_map[prop.strip()] = path.strip()

    cgcnn_map: Dict[str, str] = {}
    for item in args.cgcnn_models:
        prop, path = item.split(":", 1)
        cgcnn_map[prop.strip()] = path.strip()

    print(f"\n{'='*60}")
    print("  CGCNN 属性预测  &  评估指标计算")
    print(f"{'='*60}")
    print(f"  设备:       {'cuda' if cuda else 'cpu'}")
    print(f"  属性:       {list(pt_file_map.keys())}")
    print(f"  atom_init:  {args.atom_init}")
    print(f"  SR阈值:     {args.sr_thresholds}")

    all_metrics: Dict[str, dict] = {}

    for prop_name, pt_path in pt_file_map.items():
        print(f"\n{'='*60}")
        print(f"  属性: {prop_name}")
        print(f"  PT:   {pt_path}")
        print(f"{'='*60}")

        if not os.path.exists(pt_path):
            print(f"  [ERROR] PT文件不存在: {pt_path}")
            all_metrics[prop_name] = {}
            continue

        if prop_name not in cgcnn_map:
            print(f"  [ERROR] 未指定 '{prop_name}' 的CGCNN模型")
            all_metrics[prop_name] = {}
            continue

        cgcnn_path = cgcnn_map[prop_name]
        if not os.path.exists(cgcnn_path):
            print(f"  [ERROR] CGCNN模型不存在: {cgcnn_path}")
            all_metrics[prop_name] = {}
            continue


        try:
            dataset = PTStructureDataset(
                pt_path     = pt_path,
                atom_init   = args.atom_init,
                radius      = args.radius,
                max_num_nbr = args.max_num_nbr,
            )
        except Exception as e:
            print(f"  [ERROR] 数据集加载失败: {e}")
            traceback.print_exc()
            all_metrics[prop_name] = {}
            continue

        if len(dataset) == 0:
            print(f"  [SKIP] 数据集为空")
            all_metrics[prop_name] = {}
            continue


        try:
            print(f"  [加载CGCNN] {cgcnn_path}")
            model, normalizer, model_args = load_cgcnn(
                cgcnn_path, dataset, cuda
            )
            print(f"  ✓ CGCNN加载完成")
        except Exception as e:
            print(f"  [ERROR] CGCNN加载失败: {e}")
            traceback.print_exc()
            all_metrics[prop_name] = {}
            continue


        try:
            results = run_prediction(
                dataset    = dataset,
                model      = model,
                normalizer = normalizer,
                model_args = model_args,
                cuda       = cuda,
                batch_size = args.batch_size,
                workers    = args.workers,
            )
        except Exception as e:
            print(f"  [ERROR] 预测失败: {e}")
            traceback.print_exc()
            all_metrics[prop_name] = {}
            continue

        print(f"  预测完成: {len(results)} 条")


        csv_path = os.path.join(
            args.save_path, f"{prop_name}_predictions.csv"
        )
        df = save_csv(results, csv_path)


        metrics = compute_metrics(df, args.sr_thresholds, prop_name)

        json_path = os.path.join(
            args.save_path, f"{prop_name}_metrics.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  [JSON] 指标已保存: {json_path}")

        all_metrics[prop_name] = metrics


    valid_m = {k: v for k, v in all_metrics.items() if v}
    if len(valid_m) > 1:
        print_summary_table(valid_m, args.sr_thresholds)

    summary_path = os.path.join(args.save_path, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] 全局汇总: {summary_path}")
    print(f"[Done] 输出目录: {args.save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--pt_files", nargs="+", required=True,
        metavar="PROP:PT_PATH",
    )
    p.add_argument(
        "--cgcnn_models", nargs="+", required=True,
        metavar="PROP:MODEL_PATH",
    )
    p.add_argument("--atom_init",  required=True,
                   help="atom_init.json路径")
    p.add_argument("--save_path",  required=True,
                   help="CSV/JSON输出目录")
    p.add_argument(
        "--sr_thresholds", nargs="+", type=float,
        default=[0.1, 0.2, 0.5, 1.0],
    )
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--workers",      type=int,   default=0)
    p.add_argument("--radius",       type=float, default=8.0)
    p.add_argument("--max_num_nbr",  type=int,   default=12)
    p.add_argument("--cpu",          action="store_true")
    main(p.parse_args())