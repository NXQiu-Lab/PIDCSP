import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from typing import Any, Dict, List, Optional

from pidcsp.common.data_utils import lattice_params_to_matrix_torch
from pidcsp.common.conditional_data_utils import (
    PROPERTY_KEYS, DEFAULT_LOG_TRANSFORM_PROPS,
)
from pidcsp.pl_modules.classifier.pidcsp_g_classifier import CrystalClassifier

MAX_ATOMIC_NUM = 100


LOG_TRANSFORM_PROPS = DEFAULT_LOG_TRANSFORM_PROPS


def apply_log_transform(val: torch.Tensor, key: str) -> torch.Tensor:

    if key in LOG_TRANSFORM_PROPS:
        return torch.log1p(val.clamp(min=0.0))
    return val


def inverse_log_transform(val: torch.Tensor, key: str) -> torch.Tensor:

    if key in LOG_TRANSFORM_PROPS:
        return torch.expm1(val)
    return val


class ClassifierTrainer(pl.LightningModule):

    def __init__(
        self,
        pretrained_ckpt_path: str,

        hidden_dim:      int   = 512,
        time_dim:        int   = 256,
        num_freqs:       int   = 128,
        edge_style:      str   = 'fc',
        clf_hidden_dim:  int   = 256,
        n_mp_layers:     int   = 3,

        phys_data_path:  str   = "",
        use_phys_emb:    bool  = True,
        phys_props:      bool  = True,
        phys_pg:         bool  = True,

        num_heads:       int   = 4,
        dropout:         float = 0.0,
        alpha:           float = 0.2,
        phys_edge_weight: float = 0.5,

        use_time_phys_mod: bool = True,
        time_mod_min:    float = 0.7,

        lr:              float = 3e-4,
        weight_decay:    float = 1e-5,

        t_warmup_steps:  int   = 1000,
        low_t_ratio:     float = 0.4,

        prop_keys:       List[str] = None,
        prop_loss_weights: Dict[str, float] = None,
        log_transform_props: List[str] = None,
        use_max_features: bool = False,
        include_b_as_x: bool = False,
        max_role_feature_dim: int = 12,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.prop_keys    = prop_keys or PROPERTY_KEYS
        self.log_transform_props = set(
            LOG_TRANSFORM_PROPS if log_transform_props is None
            else log_transform_props
        )
        self.prop_weights = prop_loss_weights or {
            k: 1.0 for k in self.prop_keys}


        self.backbone = self._load_backbone(pretrained_ckpt_path)
        self._freeze_backbone()

        self.beta_scheduler  = self.backbone.beta_scheduler
        self.sigma_scheduler = self.backbone.sigma_scheduler
        self.timesteps       = self.beta_scheduler.timesteps

        resolved_phys_path = phys_data_path
        if not resolved_phys_path:
            resolved_phys_path = self._find_phys_data_path()


        self.classifier = CrystalClassifier(
            hidden_dim        = hidden_dim,
            time_dim          = time_dim,
            num_freqs         = num_freqs,
            edge_style        = edge_style,
            clf_hidden_dim    = clf_hidden_dim,
            n_mp_layers       = n_mp_layers,
            phys_data_path    = resolved_phys_path,
            use_phys_emb      = use_phys_emb,
            phys_props        = phys_props,
            phys_pg           = phys_pg,
            num_heads         = num_heads,
            dropout           = dropout,
            alpha             = alpha,
            phys_edge_weight  = phys_edge_weight,
            use_time_phys_mod = use_time_phys_mod,
            time_mod_min      = time_mod_min,
            prop_keys         = self.prop_keys,
            log_transform_props = list(self.log_transform_props),
            use_max_features  = use_max_features,
            include_b_as_x    = include_b_as_x,
            max_role_feature_dim = max_role_feature_dim,
        )

        n_clf = sum(p.numel() for p in self.classifier.parameters())
        print(f"[Classifier] 参数量: {n_clf:,}")
        print(f"[Classifier] 预测属性: {self.prop_keys}")
        print(f"[Classifier] log变换属性: {self.log_transform_props & set(self.prop_keys)}")


        self._prop_stats: Dict = {}
        self._global_step_count = 0


    def _find_phys_data_path(self) -> str:
        current = Path(__file__).resolve()
        root    = current.parent.parent.parent.parent
        path    = root / 'embeddings' / 'phys_embedding_data.pt'
        if path.exists():
            return str(path)
        return ""

    def _load_backbone(self, ckpt_path: str):
        from pidcsp.pl_modules.pidcsp_g_diffusion import PIDCSPDiffusion
        import tempfile

        phys_path = self._find_phys_data_path()

        ckpt = torch.load(ckpt_path, map_location='cpu')
        hp   = ckpt.get('hyper_parameters', {})
        self._patch_phys_path(hp, phys_path)
        ckpt['hyper_parameters'] = hp

        fd, tmp = tempfile.mkstemp(suffix='.ckpt')
        try:
            os.close(fd)
            torch.save(ckpt, tmp)
            model = PIDCSPDiffusion.load_from_checkpoint(
                tmp, map_location='cpu', strict=False)
            print(f"[Backbone] 加载成功: {ckpt_path}")
        finally:
            os.unlink(tmp)
        return model

    @staticmethod
    def _patch_phys_path(hp, correct_path):
        if isinstance(hp, dict):
            for k in list(hp.keys()):
                if k == 'phys_data_path':
                    hp[k] = correct_path
                elif isinstance(hp[k], dict):
                    ClassifierTrainer._patch_phys_path(hp[k], correct_path)

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()


    def _preprocess_target(
        self, raw_val: torch.Tensor, key: str
    ) -> torch.Tensor:


        val = raw_val.float()

        val = (torch.log1p(val.clamp(min=0.0))
               if key in self.log_transform_props else val)

        if key in self._prop_stats:
            s    = self._prop_stats[key]
            mean = float(s.get('mean', 0.0))
            std  = max(float(s.get('std', 1.0)), 1e-8)
            val  = (val - mean) / std
        return val

    def _denormalize_pred(
        self, pred_normed: torch.Tensor, key: str
    ) -> torch.Tensor:


        val = pred_normed.float()
        if key in self._prop_stats:
            s    = self._prop_stats[key]
            mean = float(s.get('mean', 0.0))
            std  = max(float(s.get('std', 1.0)), 1e-8)
            val  = val * std + mean
        val = (torch.expm1(val)
               if key in self.log_transform_props else val)
        return val


    def _add_noise(
        self,
        batch,
        times: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = batch.num_atoms.device

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1.0 - alphas_cumprod)
        sigmas = self.sigma_scheduler.sigmas[times]

        lattices    = lattice_params_to_matrix_torch(
            batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l = torch.randn_like(lattices)
        rand_x = torch.randn_like(frac_coords)
        rand_t = torch.randn(
            batch.num_nodes, MAX_ATOMIC_NUM, device=device)

        gt_onehot = F.one_hot(
            batch.atom_types - 1,
            num_classes=MAX_ATOMIC_NUM,
        ).float()

        sigmas_per_atom = sigmas.repeat_interleave(
            batch.num_atoms)[:, None]

        noisy_lattice = (
            c0[:, None, None] * lattices
            + c1[:, None, None] * rand_l
        )
        noisy_coords = (
            frac_coords + sigmas_per_atom * rand_x
        ) % 1.0
        noisy_atom_probs = (
            c0.repeat_interleave(batch.num_atoms)[:, None] * gt_onehot
            + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t
        )

        return {
            'noisy_lattice':    noisy_lattice,
            'noisy_coords':     noisy_coords,
            'noisy_atom_probs': noisy_atom_probs,
        }

    def _sample_times(
        self, B: int, device, is_training: bool
    ) -> torch.Tensor:


        T = self.timesteps

        if is_training and \
                self._global_step_count < self.hparams.t_warmup_steps:
            t_max = max(T // 3, 10)
            return torch.randint(1, t_max + 1, (B,), device=device)

        low_boundary = T // 3
        use_low = torch.rand(B, device=device) < self.hparams.low_t_ratio
        t_low   = torch.randint(1, low_boundary + 1, (B,), device=device)
        t_high  = torch.randint(low_boundary + 1, T + 1, (B,), device=device)
        return torch.where(use_low, t_low, t_high)


    def _compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch,
        times: torch.Tensor,
        device,
    ) -> Dict[str, torch.Tensor]:


        B      = batch.num_graphs
        losses = {}
        t_norm = times.float() / self.timesteps


        t_weights = torch.exp(-3.0 * t_norm)

        for key in self.prop_keys:
            pred       = predictions[key]
            target_val = getattr(batch, key, None)
            miss_flag  = getattr(batch, f'{key}_missing', None)

            if target_val is None:
                losses[key] = torch.tensor(0.0, device=device)
                continue

            target_raw  = target_val.view(B, 1).float().to(device)
            missing = (miss_flag.view(B, 1).float().to(device)
                    if miss_flag is not None
                    else torch.zeros(B, 1, device=device))

            valid = (1.0 - missing).squeeze(-1).bool()

            if valid.sum() == 0:
                losses[key] = torch.tensor(0.0, device=device)
                continue


            target_normed = self._preprocess_target(target_raw, key)

            pred_v   = pred[valid]
            target_v = target_normed[valid]
            w_v      = t_weights[valid].unsqueeze(-1)

            mse          = (pred_v - target_v) ** 2
            weighted_mse = (mse * w_v).mean()

            losses[key] = (
                self.prop_weights.get(key, 1.0) * weighted_mse)

        valid_losses = [
            losses[k] for k in self.prop_keys
            if k in losses and isinstance(losses[k], torch.Tensor)
        ]
        losses['total'] = (
            sum(valid_losses) if valid_losses
            else torch.tensor(0.0, device=device)
        )
        return losses

    def _compute_mae(
        self,
        predictions: Dict[str, torch.Tensor],
        batch,
        device,
    ) -> Dict[str, torch.Tensor]:


        B    = batch.num_graphs
        maes = {}

        for key in self.prop_keys:
            pred       = predictions[key]
            target_val = getattr(batch, key, None)
            miss_flag  = getattr(batch, f'{key}_missing', None)

            if target_val is None:
                continue

            target_raw = target_val.view(B, 1).float().to(device)
            missing    = (miss_flag.view(B, 1).float().to(device)
                         if miss_flag is not None
                         else torch.zeros(B, 1, device=device))
            valid = (1.0 - missing).squeeze(-1).bool()

            if valid.sum() == 0:
                continue


            pred_raw = self._denormalize_pred(pred[valid], key)
            target_v = target_raw[valid]

            maes[f'{key}_mae'] = (pred_raw - target_v).abs().mean()

        return maes


    def _shared_forward(self, batch, is_training: bool) -> Dict:
        B      = batch.num_graphs
        device = batch.num_atoms.device

        times = self._sample_times(B, device, is_training)
        noisy = self._add_noise(batch, times)

        predictions = self.classifier(
            atom_type_probs = noisy['noisy_atom_probs'],
            frac_coords     = noisy['noisy_coords'],
            lattice         = noisy['noisy_lattice'],
            num_atoms       = batch.num_atoms,
            node2graph      = batch.batch,
            t               = times,
        )

        losses = self._compute_loss(predictions, batch, times, device)

        if is_training:
            self._global_step_count += 1

        return losses, predictions


    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.classifier.parameters(),
            lr           = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
            betas        = (0.9, 0.999),
        )

        warmup_epochs = 5

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs

            max_ep = 200
            try:
                if self.trainer is not None and \
                        hasattr(self.trainer, 'max_epochs') and \
                        self.trainer.max_epochs is not None:
                    max_ep = self.trainer.max_epochs
            except Exception:
                pass

            progress = (epoch - warmup_epochs) / max(
                max_ep - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "monitor":   "val_loss",
            }
        }

    def training_step(self, batch: Any, batch_idx: int):
        losses, _ = self._shared_forward(batch, is_training=True)
        loss       = losses['total']

        log_dict = {'train_loss': loss}
        for k in self.prop_keys:
            if k in losses and isinstance(losses[k], torch.Tensor):
                log_dict[f'train_{k}'] = losses[k].detach()

        self.log_dict(log_dict, on_step=True, on_epoch=True,
                      prog_bar=True)

        return loss if not loss.isnan() else None

    def validation_step(self, batch: Any, batch_idx: int):
        losses, predictions = self._shared_forward(batch, is_training=False)
        loss                 = losses['total']

        log_dict = {'val_loss': loss}
        for k in self.prop_keys:
            if k in losses and isinstance(losses[k], torch.Tensor):
                log_dict[f'val_{k}'] = losses[k].detach()


        device = batch.num_atoms.device
        maes   = self._compute_mae(predictions, batch, device)
        for mk, mv in maes.items():
            log_dict[f'val_{mk}'] = mv.detach()

        self.log_dict(log_dict, on_step=False, on_epoch=True,
                      prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        self.backbone.eval()

    def on_fit_start(self):

        try:
            dm = getattr(self.trainer, 'datamodule', None)
            if dm is None:
                print("[Classifier] ⚠️  trainer.datamodule不可用")
                return

            if hasattr(dm, 'property_stats') and dm.property_stats:
                raw_stats = dm.property_stats

                updated_stats = {}
                for key, s in raw_stats.items():


                    stat = dict(s)
                    stat['log_transformed'] = bool(
                        stat.get('log_transformed', key in self.log_transform_props)
                    )
                    stat['space'] = 'transformed'
                    updated_stats[key] = stat
                    print(f"  {key}{' [log空间]' if stat['log_transformed'] else ''}: "
                          f"mean={stat.get('mean', 0.0):.4f}, "
                          f"std={stat.get('std', 1.0):.4f}")

                self._prop_stats = updated_stats
                print(f"\n[Classifier] prop_stats 已初始化（含log变换修正）")
            else:
                print("[Classifier] ⚠️  DataModule未提供prop_stats")
        except Exception as e:
            print(f"[Classifier] ⚠️  获取prop_stats失败: {e}")

    def _compute_log_stats(self, dm, key: str):


        try:
            train_ds = dm.datasets.get('train', None)
            if train_ds is None:
                return 0.0, 1.0

            vals = []
            for data in train_ds:
                v = getattr(data, key, None)
                if v is not None:
                    v = float(v)
                    if not math.isnan(v) and not math.isinf(v):
                        vals.append(math.log1p(max(v, 0.0)))

            if len(vals) < 2:
                return 0.0, 1.0

            mean = sum(vals) / len(vals)
            std  = (sum((x - mean)**2 for x in vals) / len(vals)) ** 0.5
            return mean, max(std, 1e-8)
        except Exception as e:
            print(f"    [WARN] 无法计算 {key} 的log空间统计量: {e}")
            return 0.0, 1.0

    def on_validation_epoch_end(self):
        if self.current_epoch % 10 == 0:
            self._print_diagnostics()

    def _print_diagnostics(self):
        print(f"\n[Epoch {self.current_epoch}] 分类器诊断:")
        for key in self.prop_keys:
            pred = self.classifier.predictors[key]
            last = pred[-1]
            log_flag = "★log" if key in self.log_transform_props else ""
            print(f"  {key}{log_flag}: "
                  f"w_norm={last.weight.data.norm():.4f}, "
                  f"bias={last.bias.data.item():.4f}")

    def save_classifier(self, path: str):


        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'prop_keys':       self.prop_keys,
            'hidden_dim':      self.hparams.hidden_dim,
            'time_dim':        self.hparams.time_dim,
            'num_freqs':       self.hparams.num_freqs,
            'edge_style':      self.hparams.edge_style,
            'clf_hidden_dim':  self.hparams.clf_hidden_dim,
            'n_mp_layers':     self.hparams.n_mp_layers,
            'phys_data_path':  self.hparams.phys_data_path,
            'num_heads':       self.hparams.num_heads,

            'prop_stats':      self._prop_stats,
            'log_transform_props': list(self.log_transform_props & set(self.prop_keys)),
            'use_max_features': bool(getattr(self.classifier, 'use_max_features', False)),
            'include_b_as_x': bool(getattr(self.classifier, 'include_b_as_x', False)),
            'max_role_feature_dim': self.hparams.max_role_feature_dim,
        }, path)
        print(f"[Classifier] 已保存: {path}")
        for k, s in self._prop_stats.items():
            log_flag = " [log空间]" if s.get('log_transformed', False) else ""
            print(f"  {k}{log_flag}: mean={s['mean']:.4f}, std={s['std']:.4f}")
