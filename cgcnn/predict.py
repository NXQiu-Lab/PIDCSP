import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")
from cgcnn.data import collate_pool, AtomCustomJSONInitializer, GaussianDistance
from cgcnn.model import CrystalGraphConvNet
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice


class Normalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std  = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std  = state_dict["std"]


class StructureDatasetFromPt(Dataset):


    def __init__(
        self,
        pt_dir:     str,
        target_val: float,
        atom_init:  str,
        max_num_nbr: int   = 12,
        radius:      float = 8.0,
        dmin:        float = 0.0,
        step:        float = 0.2,
    ):
        self.target_val  = target_val
        self.max_num_nbr = max_num_nbr
        self.radius      = radius


        pt_path = os.path.join(pt_dir, "eval_gen.pt")
        assert os.path.exists(pt_path), f"eval_gen.pt 不存在: {pt_path}"

        saved = torch.load(pt_path, map_location="cpu")


        if isinstance(saved, dict) and "data" in saved:

            data         = saved["data"]
            self.cif_ids = saved.get("cif_ids", None)
            num_atoms    = data["num_atoms"]
            lengths      = data["lengths"]
            angles       = data["angles"]
            frac_coords  = data["frac_coords"]
            atom_types   = data["atom_types"]
            self._build_structures_flat(
                num_atoms, lengths, angles, frac_coords, atom_types)

        else:


            self.cif_ids = None
            num_atoms    = saved["num_atoms"]
            lengths      = saved["lengths"]
            angles       = saved["angles"]
            frac_coords  = saved["frac_coords"]
            atom_types   = saved["atom_types"]
            self._build_structures_batched(
                num_atoms, lengths, angles, frac_coords, atom_types)

        if self.cif_ids is None:
            self.cif_ids = [f"crystal_{i}" for i in range(len(self.structs))]


        assert os.path.exists(atom_init), \
            f"atom_init.json 不存在: {atom_init}"
        self.ari = AtomCustomJSONInitializer(atom_init)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)


    def _build_structures_flat(self, num_atoms, lengths, angles,
                               frac_coords, atom_types):
        num_atoms_list   = num_atoms.tolist()
        lengths_np       = lengths.numpy()
        angles_np        = angles.numpy()
        frac_coords_np   = frac_coords.numpy()
        atom_types_list  = atom_types.tolist()

        self.structs = []
        offset = 0
        for i, n in enumerate(num_atoms_list):
            n = int(n)
            lat = Lattice.from_parameters(
                lengths_np[i, 0], lengths_np[i, 1], lengths_np[i, 2],
                angles_np[i, 0],  angles_np[i, 1],  angles_np[i, 2],
            )
            species = atom_types_list[offset: offset + n]
            coords  = frac_coords_np[offset: offset + n]
            self.structs.append(
                Structure(lat, species, coords,
                          to_unit_cell=True, coords_are_cartesian=False)
            )
            offset += n


    def _build_structures_batched(self, num_atoms, lengths, angles,
                                  frac_coords, atom_types):
        lengths_list    = lengths.numpy().tolist()
        angles_list     = angles.numpy().tolist()
        num_atoms_list  = num_atoms.tolist()
        frac_coords_list = frac_coords.numpy().tolist()
        atom_types_list  = atom_types.tolist()

        self.structs = []
        for i in range(len(num_atoms_list)):
            now_atom = 0
            for a in range(len(num_atoms_list[i])):
                length   = lengths_list[i][a]
                angle    = angles_list[i][a]
                atom_num = num_atoms_list[i][a]
                atom_type = atom_types_list[i][now_atom: now_atom + atom_num]
                frac_coord = frac_coords_list[i][now_atom: now_atom + atom_num]
                lat = Lattice.from_parameters(
                    a=length[0], b=length[1], c=length[2],
                    alpha=angle[0], beta=angle[1], gamma=angle[2],
                )
                self.structs.append(
                    Structure(lat, atom_type, frac_coord, to_unit_cell=True)
                )
                now_atom += atom_num

    def __len__(self):
        return len(self.structs)

    def __getitem__(self, idx):
        cif_id  = self.cif_ids[idx]
        crystal = self.structs[idx]

        atom_fea = np.vstack([
            self.ari.get_atom_fea(crystal[i].specie.number)
            for i in range(len(crystal))
        ])
        atom_fea = torch.Tensor(atom_fea)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f"{cif_id} 邻居数不足，用重复填充（当前: {len(nbr)}）")
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr))
                    + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr[:self.max_num_nbr])))

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea     = np.array(nbr_fea)
        nbr_fea     = self.gdf.expand(nbr_fea)

        return (
            torch.Tensor(atom_fea),
            torch.Tensor(nbr_fea),
            torch.LongTensor(nbr_fea_idx),
        ), torch.Tensor([float(self.target_val)]), cif_id


def load_checkpoint(model_path: str, cuda: bool):
    checkpoint = torch.load(
        model_path,
        map_location=lambda storage, loc: storage,
    )
    model_args = argparse.Namespace(**checkpoint["args"])
    normalizer = Normalizer(torch.zeros(3))
    normalizer.load_state_dict(checkpoint["normalizer"])
    return checkpoint, model_args, normalizer


def build_model(checkpoint, model_args, sample_dataset, cuda: bool):
    (atom_fea, nbr_fea, _), _, _ = sample_dataset[0]
    orig_atom_fea_len = atom_fea.shape[-1]
    nbr_fea_len       = nbr_fea.shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len   = model_args.atom_fea_len,
        n_conv         = model_args.n_conv,
        h_fea_len      = model_args.h_fea_len,
        n_h            = model_args.n_h,
        classification = (model_args.task == "classification"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    if cuda:
        model = model.cuda()
    model.eval()
    return model


def predict_one_pt(
    pt_dir:     str,
    target_val: float,
    atom_init:  str,
    model,
    model_args,
    normalizer,
    cuda:       bool,
    batch_size: int = 256,
    workers:    int = 0,
) -> list:

    try:

        dataset = StructureDatasetFromPt(
            pt_dir     = pt_dir,
            target_val = target_val,
            atom_init  = atom_init,
        )
    except Exception as e:
        print(f"    [ERROR] 数据集加载失败: {e}")
        import traceback; traceback.print_exc()
        return []

    if len(dataset) == 0:
        print(f"    [WARN] 数据集为空: {pt_dir}")
        return []

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = workers,
        collate_fn  = collate_pool,
    )

    results = []
    with torch.no_grad():
        for input_data, target, cif_ids in loader:
            if cuda:
                input_var = (
                    Variable(input_data[0].cuda(non_blocking=True)),
                    Variable(input_data[1].cuda(non_blocking=True)),
                    input_data[2].cuda(non_blocking=True),
                    [idx.cuda(non_blocking=True) for idx in input_data[3]],
                )
            else:
                input_var = (
                    Variable(input_data[0]),
                    Variable(input_data[1]),
                    input_data[2],
                    input_data[3],
                )

            output = model(*input_var)

            if model_args.task == "regression":
                preds = normalizer.denorm(output.data.cpu()).view(-1).tolist()
            else:
                preds = torch.exp(output.data.cpu())[:, 1].tolist()

            for cif_id, pred in zip(cif_ids, preds):
                results.append((cif_id, target_val, pred))

    return results


def main(args):
    cuda = not args.disable_cuda and torch.cuda.is_available()

    print(f"\n[Step2] CGCNN预测")
    print(f"  属性目录: {args.prop_dir}")
    print(f"  模型路径: {args.model_path}")
    print(f"  atom_init: {args.atom_init}")
    print(f"  设备:      {'cuda' if cuda else 'cpu'}")


    if not os.path.exists(args.atom_init):
        print(f"  [ERROR] atom_init.json 不存在: {args.atom_init}")
        return


    meta_path = os.path.join(args.prop_dir, "pt_meta.json")
    if not os.path.exists(meta_path):
        print(f"  [ERROR] 未找到 pt_meta.json，请先运行 step1_cif2pt.py")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    checkpoint, model_args, normalizer = load_checkpoint(args.model_path, cuda)

    model       = None
    all_results = []

    for entry in meta["targets"]:
        tidx      = entry["tidx"]
        tval      = entry["target_val"]
        formula   = entry.get("formula", "unknown")
        subdir    = entry["subdir"]
        n_success = entry["n_success"]
        sign      = "+" if tval >= 0 else ""
        pt_dir    = os.path.join(args.prop_dir, subdir)

        if n_success == 0 or not os.path.exists(
            os.path.join(pt_dir, "eval_gen.pt")
        ):
            print(f"  [跳过] t{tidx:03d}  {formula}  "
                  f"val={sign}{tval:.4f}  (无有效结构)")
            continue

        print(f"  [预测] t{tidx:03d}  {formula:<18s}"
              f"  val={sign}{tval:.4f}  ", end="", flush=True)


        if model is None:
            try:
                tmp = StructureDatasetFromPt(
                    pt_dir     = pt_dir,
                    target_val = tval,
                    atom_init  = args.atom_init,
                )
                model = build_model(checkpoint, model_args, tmp, cuda)
                print(f"\n  [Model] CGCNN 初始化完成")
            except Exception as e:
                print(f"\n  [ERROR] 模型初始化失败: {e}")
                import traceback; traceback.print_exc()
                return

        t0 = time.time()
        results = predict_one_pt(
            pt_dir     = pt_dir,
            target_val = tval,
            atom_init  = args.atom_init,
            model      = model,
            model_args = model_args,
            normalizer = normalizer,
            cuda       = cuda,
            batch_size = args.batch_size,
            workers    = args.workers,
        )
        elapsed = time.time() - t0
        print(f"  {len(results)} 条  ({elapsed:.1f}s)")

        if results:
            pred_csv = os.path.join(pt_dir, "predictions.csv")
            with open(pred_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["cif_id", "target_val", "pred"])
                writer.writerows(results)

        all_results.extend(results)

    if all_results:
        all_csv = os.path.join(args.prop_dir, "all_predictions.csv")
        with open(all_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cif_id", "target_val", "pred"])
            writer.writerows(all_results)
        print(f"\n[Step2 完成]  共 {len(all_results)} 条预测")
        print(f"  汇总 CSV → {all_csv}")
    else:
        print(f"\n[Step2] 无有效预测结果")


if __name__ == "__main__":
    import csv
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--prop_dir",    required=True,
                   help="单个属性目录，如 .../formation_energy_per_atom")
    p.add_argument("--model_path",  required=True,
                   help="CGCNN model_best.pth.tar 路径")
    p.add_argument("--atom_init",   required=True,
                   help="atom_init.json 完整路径（必须指定）")
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--workers",     type=int,   default=0)
    p.add_argument("--disable_cuda", action="store_true")
    main(p.parse_args())