import os
import torch
from pathlib import Path
from typing import Optional, Sequence, Dict

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from pidcsp.common.data_utils import get_scaler_from_data_list
from pidcsp.common.conditional_data_utils import PROPERTY_KEYS


def worker_init_fn(worker_id: int):

    import numpy as np
    import random
    np.random.seed(worker_id)
    random.seed(worker_id)


class ConditionalCrystDataModule(pl.LightningDataModule):


    def __init__(
        self,
        datasets:             DictConfig,
        num_workers:          DictConfig,
        batch_size:           DictConfig,
        scaler_path:          str = None,
        prop_stats_save_path: str = None,
    ):
        super().__init__()
        self.datasets             = datasets
        self.num_workers          = num_workers
        self.batch_size           = batch_size
        self.prop_stats_save_path = prop_stats_save_path

        self.train_dataset:  Optional[Dataset]           = None
        self.val_datasets:   Optional[Sequence[Dataset]] = None
        self.test_datasets:  Optional[Sequence[Dataset]] = None
        self.property_stats: Dict                        = {}


        self._cached_train_ds: Optional[Dataset] = None


        self._get_scaler(scaler_path)


    def _get_scaler(self, scaler_path: Optional[str]):


        if scaler_path is None:

            train_ds = hydra.utils.instantiate(self.datasets.train)
            self._cached_train_ds = train_ds

            self.lattice_scaler = get_scaler_from_data_list(
                train_ds.cached_data, key='scaled_lattice')
            self.scaler = get_scaler_from_data_list(
                train_ds.cached_data, key=train_ds.prop)


            self.property_stats = getattr(train_ds, 'prop_stats', {})

            print(f"[DataModule] Scaler构建完成")
            print(f"[DataModule] prop_stats属性: "
                  f"{list(self.property_stats.keys())}")
        else:

            scaler_dir = Path(scaler_path)
            try:
                self.lattice_scaler = torch.load(
                    scaler_dir / 'lattice_scaler.pt', map_location='cpu')
                self.scaler = torch.load(
                    scaler_dir / 'prop_scaler.pt', map_location='cpu')

                stats_file = scaler_dir / 'prop_stats.pt'
                if stats_file.exists():
                    self.property_stats = torch.load(
                        stats_file, map_location='cpu')
                    print(f"[DataModule] 从文件加载prop_stats: {stats_file}")
                else:
                    print(f"[DataModule] ⚠️  prop_stats文件不存在，"
                          f"将在setup()中从训练集计算")

                print(f"[DataModule] 从文件加载scaler: {scaler_path}")
            except Exception as e:
                print(f"[DataModule] ⚠️  加载scaler失败({e})，"
                      f"重新从训练集计算")
                train_ds = hydra.utils.instantiate(self.datasets.train)
                self._cached_train_ds = train_ds
                self.lattice_scaler = get_scaler_from_data_list(
                    train_ds.cached_data, key='scaled_lattice')
                self.scaler = get_scaler_from_data_list(
                    train_ds.cached_data, key=train_ds.prop)
                self.property_stats = getattr(train_ds, 'prop_stats', {})


    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":

            if self._cached_train_ds is not None:
                self.train_dataset    = self._cached_train_ds
                self._cached_train_ds = None
                print("[DataModule] 复用缓存的训练集")
            else:
                self.train_dataset = hydra.utils.instantiate(
                    self.datasets.train)
                print("[DataModule] 重新实例化训练集")


            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler         = self.scaler


            train_stats = getattr(self.train_dataset, 'prop_stats', {})
            if train_stats:
                self.property_stats = train_stats


            self.val_datasets = [
                hydra.utils.instantiate(cfg)
                for cfg in self.datasets.val
            ]
            for ds in self.val_datasets:
                ds.lattice_scaler = self.lattice_scaler
                ds.scaler         = self.scaler
                if hasattr(ds, 'set_property_stats') and self.property_stats:
                    ds.set_property_stats(self.property_stats)


            self._save_prop_stats()

            print(f"[DataModule] setup完成")
            print(f"  训练集: {len(self.train_dataset)}")
            print(f"  验证集: {[len(d) for d in self.val_datasets]}")

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(cfg)
                for cfg in self.datasets.test
            ]
            for ds in self.test_datasets:
                ds.lattice_scaler = self.lattice_scaler
                ds.scaler         = self.scaler
                if hasattr(ds, 'set_property_stats') and self.property_stats:
                    ds.set_property_stats(self.property_stats)

            print(f"  测试集: {[len(d) for d in self.test_datasets]}")

    def _save_prop_stats(self):

        if not self.prop_stats_save_path or not self.property_stats:
            return
        save_dir = Path(self.prop_stats_save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        stats_path = save_dir / 'prop_stats.pt'
        torch.save(self.property_stats, stats_path)

        lattice_path = save_dir / 'lattice_scaler.pt'
        torch.save(self.lattice_scaler, lattice_path)

        prop_scaler_path = save_dir / 'prop_scaler.pt'
        torch.save(self.scaler, prop_scaler_path)

        print(f"[DataModule] prop_stats已保存: {stats_path}")
        for key, s in self.property_stats.items():
            print(f"  {key}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                  f"count={s.get('count', 'N/A')}")


    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle         = shuffle,
            batch_size      = self.batch_size.train,
            num_workers     = self.num_workers.train,
            worker_init_fn  = worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                ds,
                shuffle        = False,
                batch_size     = self.batch_size.val,
                num_workers    = self.num_workers.val,
                worker_init_fn = worker_init_fn,
            )
            for ds in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                ds,
                shuffle        = False,
                batch_size     = self.batch_size.test,
                num_workers    = self.num_workers.test,
                worker_init_fn = worker_init_fn,
            )
            for ds in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (f"ConditionalCrystDataModule("
                f"train={len(self.train_dataset) if self.train_dataset else 'N/A'}, "
                f"props={PROPERTY_KEYS})")