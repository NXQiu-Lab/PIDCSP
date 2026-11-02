import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import pytorch_lightning as pl
from torch_scatter import scatter
from tqdm import tqdm

from pidcsp.common.data_utils import lattice_params_to_matrix_torch
from pidcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
from pidcsp.pl_modules.property_embeddings import PropertyEmbeddings
from pidcsp.pl_modules.property_predictor import (
    PropertyPredictorHead,
    compute_property_prediction_loss,
    PROPERTY_KEYS,
)

MAX_ATOMIC_NUM = 100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {
            "optimizer":    opt,
            "lr_scheduler": scheduler,
            "monitor":      "val_loss",
        }


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device   = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LightweightAlignmentLoss(nn.Module):


    def __init__(
        self,
        n_pairs:    int   = 32,
        margin:     float = 0.5,
        prop_sigma: float = 1.0,
    ):
        super().__init__()
        self.n_pairs    = n_pairs
        self.margin     = margin
        self.prop_sigma = prop_sigma

    def forward(
        self,
        cond_emb:  torch.Tensor,
        batch,
        prop_keys = PROPERTY_KEYS,
    ) -> torch.Tensor:
        B      = cond_emb.shape[0]
        device = cond_emb.device

        if B < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        prop_vecs = []
        for key in prop_keys:
            val  = getattr(batch, key, None)
            miss = getattr(batch, f'{key}_missing', None)
            if val is None:
                prop_vecs.append(torch.zeros(B, 1, device=device))
            else:
                v = val.view(B, 1).float().to(device)
                m = (miss.view(B, 1).float().to(device)
                     if miss is not None
                     else torch.zeros(B, 1, device=device))
                prop_vecs.append(v * (1.0 - m))
        prop_matrix = torch.cat(prop_vecs, dim=-1)

        n_pairs = min(self.n_pairs, B * (B - 1) // 2)
        idx_i   = torch.randint(0, B, (n_pairs,), device=device)
        idx_j   = torch.randint(0, B, (n_pairs,), device=device)
        same    = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % B

        prop_diff = (prop_matrix[idx_i] - prop_matrix[idx_j]).norm(dim=-1)
        prop_sim  = torch.exp(
            -prop_diff ** 2 / (2 * self.prop_sigma ** 2))

        c_i      = F.normalize(cond_emb[idx_i], dim=-1)
        c_j      = F.normalize(cond_emb[idx_j], dim=-1)
        cond_sim = (c_i * c_j).sum(dim=-1)

        loss_sim  = prop_sim  * F.relu(self.margin - cond_sim)
        loss_diff = (1.0 - prop_sim) * F.relu(cond_sim + self.margin)
        return (loss_sim + loss_diff).mean()


class PIDCSPConditionalDiffusion(BaseModule):


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


        self.prop_embeddings = PropertyEmbeddings(
            hidden_dim=self.hparams.get('cond_hidden_dim', 128),
        )
        self.cond_dim = self.prop_embeddings.total_dim


        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim  = self.hparams.latent_dim + self.hparams.time_dim,
            pred_type   = False,
            smooth      = False,
            cond_dim    = self.cond_dim,
            _recursive_ = False,
        )


        self.prop_predictor = PropertyPredictorHead(
            hidden_dim   = self.hparams.get('prop_hidden_dim', 512),
            prop_keys    = PROPERTY_KEYS,
            dropout_rate = 0.1,
        )


        self.align_loss_fn = LightweightAlignmentLoss(
            n_pairs    = 32,
            margin     = 0.5,
            prop_sigma = 1.0,
        )


        self.beta_scheduler  = hydra.utils.instantiate(
            self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(
            self.hparams.sigma_scheduler)
        self.timesteps = self.beta_scheduler.timesteps


        self.time_dim       = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)


        self.keep_lattice     = self.hparams.cost_lattice < 1e-5
        self.keep_coords      = self.hparams.cost_coord   < 1e-5
        self.p_uncond         = self.hparams.get('p_uncond',         0.1)
        self.guidance_scale   = self.hparams.get('guidance_scale',   2.0)
        self.cost_prop_pred   = self.hparams.get('cost_prop_pred',   0.1)
        self.cost_align       = self.hparams.get('cost_align',       0.01)
        self.prop_pred_warmup = self.hparams.get('prop_pred_warmup', 3000)


        self._grad_check_done  = False
        self._grad_check_step  = 100


    def on_after_backward(self):


        step = self.trainer.global_step


        should_check = (
            (step <= 2000 and step % 500 == 0) or
            (step > 2000  and step % 2000 == 0)
        )
        if not should_check:
            return

        print(f"\n[FiLM梯度检查 @ step {step}]")

        film_grad_norms = []
        prop_grad_norms = []


        for i in range(self.decoder.num_layers):
            layer = self.decoder._modules.get(f'csp_layer_{i}')
            if layer is None:
                continue

            for film_name in ['film_pre', 'film_post']:
                film = getattr(layer, film_name, None)
                if film is None:
                    continue

                g_grad = film.gamma_net[-1].weight.grad
                b_grad = film.beta_net[-1].weight.grad

                g_norm = g_grad.norm().item() if g_grad is not None else 0.0
                b_norm = b_grad.norm().item() if b_grad is not None else 0.0
                film_grad_norms.append((i, film_name, g_norm, b_norm))

                status = "✅" if (g_norm > 1e-6 or b_norm > 1e-6) else "❌零梯度"
                print(f"  Layer{i}.{film_name}: "
                      f"γ_grad={g_norm:.2e}, β_grad={b_norm:.2e} {status}")


        print(f"\n  [属性预测头梯度]")
        for key in self.prop_predictor.prop_keys:
            pred_module = self.prop_predictor.predictors[key]

            last_layer  = pred_module[-1]
            w_grad = last_layer.weight.grad
            w_norm = w_grad.norm().item() if w_grad is not None else 0.0
            prop_grad_norms.append(w_norm)
            status = "✅" if w_norm > 1e-6 else "❌零梯度"
            print(f"  {key}: grad_norm={w_norm:.2e} {status}")


        print(f"\n  [属性嵌入梯度]")
        from pidcsp.pl_modules.property_predictor import PROPERTY_KEYS
        for key in PROPERTY_KEYS:
            emb = self.prop_embeddings.embeddings[key]
            w_grad = emb.mlp[0].weight.grad
            null_grad = emb.null_vec.grad
            w_norm    = w_grad.norm().item()    if w_grad    is not None else 0.0
            null_norm = null_grad.norm().item() if null_grad is not None else 0.0
            print(f"  {key}: mlp_grad={w_norm:.2e}, "
                  f"null_grad={null_norm:.2e}")


        all_film_norms = [max(g, b) for _, _, g, b in film_grad_norms]
        if all_film_norms:
            max_film_grad = max(all_film_norms)
            avg_film_grad = sum(all_film_norms) / len(all_film_norms)
            print(f"\n  FiLM梯度摘要: max={max_film_grad:.2e}, "
                  f"avg={avg_film_grad:.2e}")

            if max_film_grad < 1e-8:
                print("  ❌ 严重：FiLM完全没有梯度！")
                print("     可能原因：")
                print("     1. cond_emb在计算图外（detach了）")
                print("     2. 属性预测损失weight为0")
                print("     3. missing_mask过滤掉了所有样本")
            elif avg_film_grad < 1e-6:
                print("  ⚠️ FiLM梯度极小，条件学习很慢")
            else:
                print("  ✅ FiLM梯度正常")


    def _get_prop_loss_weight(self, is_training: bool) -> float:


        if not is_training:
            return self.cost_prop_pred
        if self.trainer is None or self.prop_pred_warmup <= 0:
            return self.cost_prop_pred
        ratio = min(1.0, self.trainer.global_step / self.prop_pred_warmup)
        return self.cost_prop_pred * ratio

    def _get_align_loss_weight(self, is_training: bool) -> float:

        if not is_training:
            return self.cost_align
        if self.trainer is None or self.prop_pred_warmup <= 0:
            return self.cost_align
        ratio = min(1.0, self.trainer.global_step / (self.prop_pred_warmup * 2))
        return self.cost_align * ratio


    def _forward_decoder(
        self,
        time_emb:           torch.Tensor,
        atom_types:         torch.Tensor,
        input_frac_coords:  torch.Tensor,
        input_lattice:      torch.Tensor,
        num_atoms:          torch.Tensor,
        batch_idx:          torch.Tensor,
        cond_emb:           torch.Tensor,
        need_node_features: bool = False,
    ) -> Dict:


        result = self.decoder(
            time_emb,
            atom_types,
            input_frac_coords,
            input_lattice,
            num_atoms,
            batch_idx,
            cond_emb             = cond_emb,
            return_node_features = need_node_features,
        )

        if need_node_features:
            pred_l, pred_x, node_features = result
            return {
                'pred_l':        pred_l,
                'pred_x':        pred_x,
                'node_features': node_features,
            }
        else:
            pred_l, pred_x = result
            return {'pred_l': pred_l, 'pred_x': pred_x}


    def _shared_forward(
        self,
        batch,
        is_training: bool,
    ) -> Dict[str, torch.Tensor]:


        B      = batch.num_graphs
        device = batch.num_atoms.device


        times          = self.beta_scheduler.uniform_sample_t(B, device)
        time_emb       = self.time_embedding(times)
        t_normalized   = times.float() / self.timesteps

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        c0             = torch.sqrt(alphas_cumprod)
        c1             = torch.sqrt(1.0 - alphas_cumprod)
        sigmas         = self.sigma_scheduler.sigmas[times]
        sigmas_norm    = self.sigma_scheduler.sigmas_norm[times]


        lattices    = lattice_params_to_matrix_torch(
            batch.lengths, batch.angles)
        frac_coords = batch.frac_coords


        rand_l = torch.randn_like(lattices)
        rand_x = torch.randn_like(frac_coords)

        input_lattice = (c0[:, None, None] * lattices
                         + c1[:, None, None] * rand_l)

        sigmas_per_atom      = sigmas.repeat_interleave(
            batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(
            batch.num_atoms)[:, None]
        input_frac_coords    = (
            frac_coords + sigmas_per_atom * rand_x) % 1.

        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices


        p_uncond = self.p_uncond if is_training else 0.0
        cond_emb = self.prop_embeddings(batch, p_uncond=p_uncond)


        prop_pred_weight = self._get_prop_loss_weight(is_training)
        align_weight     = self._get_align_loss_weight(is_training)
        need_features    = prop_pred_weight > 0


        decoder_out = self._forward_decoder(
            time_emb,
            batch.atom_types,
            input_frac_coords,
            input_lattice,
            batch.num_atoms,
            batch.batch,
            cond_emb,
            need_node_features=need_features,
        )

        pred_l = decoder_out['pred_l']
        pred_x = decoder_out['pred_x']


        tar_x = (d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom)
            / torch.sqrt(sigmas_norm_per_atom))

        loss_lattice   = F.mse_loss(pred_l, rand_l)
        loss_coord     = F.mse_loss(pred_x, tar_x)
        loss_diffusion = (
            self.hparams.cost_lattice * loss_lattice
            + self.hparams.cost_coord  * loss_coord
        )


        loss_prop_total    = torch.tensor(0.0, device=device)
        prop_losses_detail = {}

        if need_features and 'node_features' in decoder_out:
            node_features    = decoder_out['node_features']
            prop_predictions = self.prop_predictor(
                node_features, batch.batch, B)

            prop_losses = compute_property_prediction_loss(
                prop_predictions,
                batch,
                prop_keys    = PROPERTY_KEYS,
                t_normalized = t_normalized,
                device       = device,
            )
            loss_prop_total    = prop_pred_weight * prop_losses['total']
            prop_losses_detail = {
                k: v for k, v in prop_losses.items() if k != 'total'
            }


        loss_align = torch.tensor(0.0, device=device)
        if align_weight > 0 and B >= 4:
            loss_align = align_weight * self.align_loss_fn(
                cond_emb, batch)


        loss = loss_diffusion + loss_prop_total + loss_align

        result = {
            'loss':           loss,
            'loss_lattice':   loss_lattice,
            'loss_coord':     loss_coord,
            'loss_diffusion': loss_diffusion,
            'loss_prop':      loss_prop_total,
            'loss_align':     loss_align,
        }
        for k, v in prop_losses_detail.items():
            result[f'loss_prop_{k}'] = v.detach()
        if is_training:
            result['prop_weight']  = torch.tensor(prop_pred_weight)
            result['align_weight'] = torch.tensor(align_weight)

        return result

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        return self._shared_forward(batch, is_training=True)


    def _decode_with_cfg(
        self,
        time_emb:   torch.Tensor,
        atom_types: torch.Tensor,
        x_t:        torch.Tensor,
        l_t:        torch.Tensor,
        num_atoms:  torch.Tensor,
        batch_idx:  torch.Tensor,
        cond_emb:   torch.Tensor,
        w:          float,
    ):


        B      = cond_emb.shape[0]
        N      = atom_types.shape[0]
        device = cond_emb.device

        if w <= 1.0:
            out = self._forward_decoder(
                time_emb, atom_types, x_t, l_t,
                num_atoms, batch_idx, cond_emb,
                need_node_features=False,
            )
            return out['pred_l'], out['pred_x']


        uncond_emb = self.prop_embeddings.get_uncond_embedding(B, device)

        cat_time  = torch.cat([time_emb,   time_emb],    dim=0)
        cat_l     = torch.cat([l_t,        l_t],         dim=0)
        cat_cond  = torch.cat([cond_emb,   uncond_emb],  dim=0)
        cat_atom  = torch.cat([atom_types, atom_types],  dim=0)
        cat_x     = torch.cat([x_t,        x_t],         dim=0)
        cat_na    = torch.cat([num_atoms,   num_atoms],   dim=0)

        cat_batch = torch.cat([batch_idx,  batch_idx + B], dim=0)

        out = self._forward_decoder(
            cat_time, cat_atom, cat_x, cat_l,
            cat_na, cat_batch, cat_cond,
            need_node_features=False,
        )
        pred_l_c = out['pred_l']
        pred_x_c = out['pred_x']


        pred_l = pred_l_c[:B] + (w - 1.0) * (pred_l_c[:B] - pred_l_c[B:])
        pred_x = pred_x_c[:N] + (w - 1.0) * (pred_x_c[:N] - pred_x_c[N:])

        return pred_l, pred_x


    @torch.no_grad()
    def sample(
        self,
        batch,
        step_lr:        float = 1e-5,
        guidance_scale: float = None,
        target_props:   Optional[Dict[str, float]] = None,
    ):


        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        B      = batch.num_graphs
        device = self.device

        if target_props is not None:
            batch = self._inject_target_props(batch, target_props)


        cond_emb   = self.prop_embeddings(batch, p_uncond=0.0)
        atom_types = batch.atom_types


        l_T = torch.randn([B, 3, 3], device=device)
        x_T = torch.rand([batch.num_nodes, 3], device=device)

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        traj = {self.timesteps: {
            'num_atoms':   batch.num_atoms,
            'atom_types':  atom_types,
            'frac_coords': x_T % 1.,
            'lattices':    l_T,
        }}

        for t in tqdm(range(self.timesteps, 0, -1),
                      desc="CSP Conditional Sampling"):

            times    = torch.full((B,), t, device=device)
            time_emb = self.time_embedding(times)

            alphas         = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas_ddpm    = self.beta_scheduler.sigmas[t]
            sigma_x        = self.sigma_scheduler.sigmas[t]
            sigma_norm     = self.sigma_scheduler.sigmas_norm[t]
            adj_sigma_x    = self.sigma_scheduler.sigmas[t - 1]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']


            rand_x = (torch.randn_like(x_T) if t > 1
                      else torch.zeros_like(x_T))
            rand_l = (torch.randn_like(l_T) if t > 1
                      else torch.zeros_like(l_T))

            pred_l, pred_x = self._decode_with_cfg(
                time_emb, atom_types, x_t, l_t,
                batch.num_atoms, batch.batch, cond_emb,
                w=guidance_scale)

            step_size = (step_lr
                         * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)
            pred_x_s = pred_x * torch.sqrt(sigma_norm)

            x_t_half = (x_t - step_size * pred_x_s + std_x * rand_x
                        if not self.keep_coords else x_t)
            l_t_half = l_t


            rand_x = (torch.randn_like(x_T) if t > 1
                      else torch.zeros_like(x_T))
            rand_l = (torch.randn_like(l_T) if t > 1
                      else torch.zeros_like(l_T))

            step_size2 = sigma_x ** 2 - adj_sigma_x ** 2
            std_x2     = torch.sqrt(
                adj_sigma_x ** 2 * step_size2
                / (sigma_x ** 2 + 1e-10))

            pred_l, pred_x = self._decode_with_cfg(
                time_emb, atom_types, x_t_half, l_t_half,
                batch.num_atoms, batch.batch, cond_emb,
                w=guidance_scale)

            pred_x_s = pred_x * torch.sqrt(sigma_norm)

            x_t_m1 = (x_t_half - step_size2 * pred_x_s + std_x2 * rand_x
                      if not self.keep_coords else x_t)
            l_t_m1 = (c0 * (l_t_half - c1 * pred_l) + sigmas_ddpm * rand_l
                      if not self.keep_lattice else l_t)


            if not self.keep_lattice:
                vec_norms = l_t_m1.norm(dim=-1, keepdim=True)
                max_norm  = 20.0
                scale     = torch.clamp(
                    max_norm / vec_norms.clamp(min=1e-6), max=1.0)
                l_t_m1 = l_t_m1 * scale

            traj[t - 1] = {
                'num_atoms':   batch.num_atoms,
                'atom_types':  atom_types,
                'frac_coords': x_t_m1 % 1.,
                'lattices':    l_t_m1,
            }

        traj_stack = {
            'num_atoms':       batch.num_atoms,
            'atom_types':      atom_types,
            'all_frac_coords': torch.stack([
                traj[i]['frac_coords']
                for i in range(self.timesteps, -1, -1)]),
            'all_lattices':    torch.stack([
                traj[i]['lattices']
                for i in range(self.timesteps, -1, -1)]),
        }
        return traj[0], traj_stack

    def _inject_target_props(self, batch, target_props: Dict):
        B      = batch.num_graphs
        device = batch.num_atoms.device
        for key, raw_val in target_props.items():
            if hasattr(self, '_prop_stats') and key in self._prop_stats:
                s      = self._prop_stats[key]
                norm_v = (float(raw_val) - s.get('mean', 0.0)
                          ) / (s.get('std', 1.0) + 1e-8)
            else:
                norm_v = float(raw_val)
            setattr(batch, key,
                    torch.full((B,), norm_v,
                               dtype=torch.float32, device=device))
            setattr(batch, f'{key}_missing',
                    torch.zeros(B, dtype=torch.float32, device=device))
        return batch

    def set_prop_stats(self, stats: dict):
        self._prop_stats = stats


    def training_step(self, batch: Any, batch_idx: int):
        out  = self._shared_forward(batch, is_training=True)
        loss = out['loss']

        log_dict = {
            'train_loss':           loss,
            'train_loss_diffusion': out['loss_diffusion'],
            'train_loss_lattice':   out['loss_lattice'],
            'train_loss_coord':     out['loss_coord'],
            'train_loss_prop':      out['loss_prop'],
            'train_loss_align':     out['loss_align'],
        }
        if 'prop_weight' in out:
            log_dict['prop_weight']  = out['prop_weight']
            log_dict['align_weight'] = out['align_weight']
        for key in PROPERTY_KEYS:
            k = f'loss_prop_{key}'
            if k in out:
                log_dict[f'train_{k}'] = out[k]

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        return None if loss.isnan() else loss

    def validation_step(self, batch: Any, batch_idx: int):
        out = self._shared_forward(batch, is_training=False)
        log_dict, loss = self._compute_stats(out, 'val')
        self.log('val_loss', loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        out            = self._shared_forward(batch, is_training=False)
        log_dict, loss = self._compute_stats(out, 'test')
        self.log_dict(log_dict)
        return loss

    def _compute_stats(self, out: dict, prefix: str):
        log_dict = {
            f'{prefix}_loss':           out['loss'],
            f'{prefix}_loss_diffusion': out['loss_diffusion'],
            f'{prefix}_loss_lattice':   out['loss_lattice'],
            f'{prefix}_loss_coord':     out['loss_coord'],
            f'{prefix}_loss_prop':      out['loss_prop'],
            f'{prefix}_loss_align':     out['loss_align'],
        }
        for key in PROPERTY_KEYS:
            k = f'loss_prop_{key}'
            if k in out:
                log_dict[f'{prefix}_{k}'] = out[k]
        return log_dict, out['loss']


    def on_train_epoch_end(self):
        if self.current_epoch % 10 != 0:
            return

        print(f"\n[PIDCSP-Conditional] Epoch {self.current_epoch}")


        for i in range(self.decoder.num_layers):
            layer = self.decoder._modules.get(f'csp_layer_{i}')
            if layer is None:
                continue
            film = getattr(layer, 'film_pre', None)
            if film is not None:
                g_dev = (film.gamma_net[-1].bias.data - 1.0
                         ).abs().mean().item()
                b_dev = film.beta_net[-1].bias.data.abs().mean().item()
                status = "✓" if (g_dev > 0.01 or b_dev > 0.01) else "⚠"
                print(f"  Layer {i} FiLM: γ_dev={g_dev:.4f}, "
                      f"β_dev={b_dev:.4f} {status}")


        device = next(self.parameters()).device
        B_test = 4

        class _M:
            num_graphs = B_test
            num_atoms  = torch.ones(B_test, dtype=torch.long, device=device)

        mock = _M()
        for key in PROPERTY_KEYS:
            setattr(mock, key,
                    torch.zeros(B_test, device=device))
            setattr(mock, key + '_missing',
                    torch.zeros(B_test, device=device))

        with torch.no_grad():
            ce = self.prop_embeddings(mock, p_uncond=0.0)
            ue = self.prop_embeddings.get_uncond_embedding(B_test, device)
            diff = (ce - ue).norm(dim=-1).mean().item()

        print(f"  CFG嵌入差异: {diff:.4f} "
              f"{'✓有效' if diff > 0.1 else '⚠过小'}")

        pw = self._get_prop_loss_weight(is_training=False)
        aw = self._get_align_loss_weight(is_training=False)
        print(f"  损失权重: prop={pw:.4f}, align={aw:.4f}")
