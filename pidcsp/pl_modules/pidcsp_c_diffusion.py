import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from pidcsp.common.utils import PROJECT_ROOT
from pidcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from pidcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PIDCSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T


            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2

            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t


            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    @torch.no_grad()
    def Energysample(
        self,
        batch,
        step_lr: float = 1e-5,
        energy_config: Dict[str, Any] = None,
    ):


        _DEFAULT_CONFIG = {
            'weight':              0.5,
            'start_step':          950,
            'schedule':            'linear',
            'min_distance':        0.8,
            'min_volume':          10.0,
            'max_volume':          10000.0,
            'max_lattice_norm':    30.0,
            'max_atoms_energy':    150,
            'grad_clip_norm':      10.0,
            'guide_coords':        True,
            'guide_lattice':       False,
            'lattice_guide_weight': 0.1,
            'corrector_guidance':  True,
            'predictor_guidance':  True,
            'use_amp':             True,
            'log_interval':        50,
        }

        cfg = _DEFAULT_CONFIG.copy()
        if energy_config is not None:
            unknown_keys = set(energy_config) - set(_DEFAULT_CONFIG)
            if unknown_keys:
                import warnings
                warnings.warn(
                    f"[EnergyGuide] Unknown config keys ignored: {unknown_keys}"
                )
            cfg.update({k: v for k, v in energy_config.items()
                        if k in _DEFAULT_CONFIG})


        if not hasattr(self, '_chgnet_model') or self._chgnet_model is None:
            try:
                from chgnet.model.model import CHGNet
                self._chgnet_model = CHGNet.load().to(self.device)
                self._chgnet_model.eval()
                for p in self._chgnet_model.parameters():
                    p.requires_grad_(False)
                print("[EnergyGuide] CHGNet loaded successfully on "
                    f"{self.device}")
            except Exception as e:
                print(f"[EnergyGuide] CHGNet unavailable ({e}). "
                    "Falling back to standard sampling.")
                return self.sample(batch, step_lr)

        device     = self.device
        batch_size = batch.num_graphs
        time_start = self.beta_scheduler.timesteps


        guidance_onset_t = time_start - cfg['start_step']

        def _guidance_scale(t: int) -> float:

            if t > guidance_onset_t:
                return 0.0
            progress = 1.0 - t / max(guidance_onset_t, 1)

            schedule = cfg['schedule']
            w        = cfg['weight']

            if schedule == 'constant':
                return w
            elif schedule == 'linear':
                return w * progress
            elif schedule == 'cosine':
                return w * (1.0 - math.cos(progress * math.pi)) / 2.0
            elif schedule == 'warmup':
                warmup_frac = 0.2
                if progress < warmup_frac:
                    return w * (progress / warmup_frac)
                return w
            else:
                return w * progress


        def _lattice_valid(lattice_np: np.ndarray) -> bool:

            try:
                if not np.all(np.isfinite(lattice_np)):
                    return False
                det = np.linalg.det(lattice_np)
                vol = abs(det)
                if vol < cfg['min_volume'] or vol > cfg['max_volume']:
                    return False
                norms = np.linalg.norm(lattice_np, axis=1)
                if np.any(norms > cfg['max_lattice_norm']) or np.any(norms < 0.1):
                    return False
                return True
            except Exception:
                return False

        def _structure_valid(struct) -> bool:

            try:
                dmat = struct.distance_matrix
                np.fill_diagonal(dmat, np.inf)
                return float(np.min(dmat)) > cfg['min_distance']
            except Exception:
                return False


        def _compute_energy_grads(
            frac_coords:  torch.Tensor,
            lattices:     torch.Tensor,
            atom_types:   torch.Tensor,
            t:            int,
        ):


            from ase.data import chemical_symbols
            from pymatgen.core.structure import Structure, Lattice

            scale = _guidance_scale(t)
            if scale < 1e-10:
                return (torch.zeros_like(frac_coords),
                        torch.zeros_like(lattices), None)

            coord_grad_list   = []
            lattice_grad_list = []
            energy_vals       = []
            n_valid           = 0

            for i in range(batch_size):
                n_at  = batch.num_atoms[i].item()
                s_idx = batch.ptr[i].item()
                e_idx = batch.ptr[i + 1].item()

                zero_cg = torch.zeros((n_at, 3), device=device)
                zero_lg = torch.zeros((3, 3),    device=device)

                try:

                    if n_at > cfg['max_atoms_energy']:
                        raise ValueError(f"n_atoms={n_at} exceeds limit")


                    lat_np = lattices[i].detach().cpu().numpy()
                    lat_np = np.clip(lat_np,
                                    -cfg['max_lattice_norm'],
                                    cfg['max_lattice_norm'])
                    if not _lattice_valid(lat_np):
                        raise ValueError("lattice fails validation")


                    types_i = atom_types[s_idx:e_idx]
                    types_i = torch.clamp(types_i, 0, MAX_ATOMIC_NUM - 1)
                    symbols = [chemical_symbols[z.item() + 1] for z in types_i]


                    frac_np = frac_coords[s_idx:e_idx].detach().cpu().numpy()
                    frac_np = frac_np % 1.0


                    struct = Structure(Lattice(lat_np), symbols, frac_np,
                                    coords_are_cartesian=False)

                    if not _structure_valid(struct):
                        raise ValueError("interatomic distance too small")


                    coords_diff = (frac_coords[s_idx:e_idx]
                                .detach().clone() % 1.0)
                    coords_diff.requires_grad_(True)

                    lat_diff = None
                    if cfg['guide_lattice']:
                        lat_diff = lattices[i].detach().clone()
                        lat_diff.requires_grad_(True)


                    graph = self._chgnet_model.graph_converter(struct)
                    for attr_name in vars(graph):
                        val = getattr(graph, attr_name)
                        if isinstance(val, torch.Tensor):
                            setattr(graph, attr_name, val.to(device))
                    graph.coords = coords_diff

                    use_amp = cfg['use_amp'] and device.type == 'cuda'
                    with torch.amp.autocast(
                        device_type=device.type, enabled=use_amp
                    ):
                        with torch.enable_grad():
                            out = self._chgnet_model([graph])

                    if (not isinstance(out, dict) or 'e' not in out
                            or out['e'].isnan().any()):
                        raise ValueError("CHGNet returned invalid output")

                    energy = out['e'].sum()
                    energy_vals.append(energy.item())


                    targets = [coords_diff]
                    if cfg['guide_lattice'] and lat_diff is not None:
                        targets.append(lat_diff)

                    with torch.enable_grad():
                        grads = torch.autograd.grad(
                            -energy,
                            targets,
                            create_graph=False,
                            retain_graph=False,
                            allow_unused=True,
                        )

                    cg = grads[0] if grads[0] is not None else zero_cg

                    gn = torch.norm(cg)
                    if gn > cfg['grad_clip_norm']:
                        cg = cg * (cfg['grad_clip_norm'] / (gn + 1e-8))
                    if torch.isnan(cg).any():
                        raise ValueError("NaN in coord gradient")
                    coord_grad_list.append(cg)

                    if cfg['guide_lattice'] and len(grads) > 1 \
                            and grads[1] is not None:
                        lg = grads[1]
                        lg_n = torch.norm(lg)
                        if lg_n > cfg['grad_clip_norm']:
                            lg = lg * (cfg['grad_clip_norm'] / (lg_n + 1e-8))
                        if torch.isnan(lg).any():
                            lg = zero_lg
                        lattice_grad_list.append(lg)
                    else:
                        lattice_grad_list.append(zero_lg)

                    n_valid += 1

                except Exception as exc:
                    coord_grad_list.append(zero_cg)
                    lattice_grad_list.append(zero_lg)
                    if t % cfg['log_interval'] == 0:
                        print(f"  [EnergyGrad] struct {i} skipped: {exc}")

            coord_grads   = torch.cat(coord_grad_list, dim=0)
            lattice_grads = torch.stack(lattice_grad_list, dim=0)

            info = None
            if energy_vals:
                info = {
                    'mean_energy': float(np.mean(energy_vals)),
                    'std_energy':  float(np.std(energy_vals))
                                if len(energy_vals) > 1 else 0.0,
                    'n_valid':     n_valid,
                    'grad_norm':   float(torch.norm(coord_grads).item()),
                }
                if t % cfg['log_interval'] == 0:
                    print(
                        f"  [Energy] t={t:>4d}  "
                        f"w={scale:.4f}  "
                        f"E={info['mean_energy']:.4f}±"
                        f"{info['std_energy']:.4f} eV/atom  "
                        f"‖∇‖={info['grad_norm']:.4f}  "
                        f"valid={n_valid}/{batch_size}"
                    )

            return coord_grads, lattice_grads, info


        l_T = torch.randn([batch_size, 3, 3], device=device)
        x_T = torch.rand([batch.num_nodes, 3], device=device)

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        traj = {
            time_start: {
                'num_atoms':   batch.num_atoms,
                'atom_types':  batch.atom_types,
                'frac_coords': x_T % 1.,
                'lattices':    l_T,
            }
        }

        energy_trajectory = []


        for t in tqdm(range(time_start, 0, -1),
                    desc="Energy-Guided Sampling"):

            times    = torch.full((batch_size,), t, device=device)
            time_emb = self.time_embedding(times)


            alphas        = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            beta_sigma    = self.beta_scheduler.sigmas[t]
            sigma_x       = self.sigma_scheduler.sigmas[t]
            sigma_norm    = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            g_scale = _guidance_scale(t)


            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)


            if cfg['corrector_guidance'] and g_scale > 1e-10:
                cg_coord, cg_lat, cg_info = _compute_energy_grads(
                    x_t, l_t, batch.atom_types, t
                )
                if cg_info is not None:
                    energy_trajectory.append({
                        't': t, 'phase': 'corrector', **cg_info
                    })
            else:
                cg_coord = torch.zeros_like(x_t)
                cg_lat   = torch.zeros_like(l_t)


            pred_l_c, pred_x_c = self.decoder(
                time_emb, batch.atom_types, x_t, l_t,
                batch.num_atoms, batch.batch
            )


            pred_x_c_scaled = pred_x_c * torch.sqrt(sigma_norm)
            if cfg['guide_coords']:
                pred_x_c_guided = pred_x_c_scaled + g_scale * cg_coord
            else:
                pred_x_c_guided = pred_x_c_scaled

            step_size_c = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x_c     = torch.sqrt(2 * step_size_c)

            x_t_half = (x_t - step_size_c * pred_x_c_guided + std_x_c * rand_x
                        if not self.keep_coords else x_t)


            l_t_half = l_t
            if cfg['guide_lattice'] and not self.keep_lattice \
                    and g_scale > 1e-10:
                lat_step = step_lr * cfg['lattice_guide_weight']
                l_t_half = l_t + lat_step * g_scale * cg_lat


            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)


            if cfg['predictor_guidance'] and g_scale > 1e-10:
                pg_coord, pg_lat, pg_info = _compute_energy_grads(
                    x_t_half, l_t_half, batch.atom_types, t
                )
                if pg_info is not None:
                    energy_trajectory.append({
                        't': t, 'phase': 'predictor', **pg_info
                    })
            else:
                pg_coord = torch.zeros_like(x_t_half)
                pg_lat   = torch.zeros_like(l_t_half)


            pred_l_p, pred_x_p = self.decoder(
                time_emb, batch.atom_types, x_t_half, l_t_half,
                batch.num_atoms, batch.batch
            )


            adj_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size_p = sigma_x ** 2 - adj_sigma_x ** 2
            std_x_p     = torch.sqrt(
                (adj_sigma_x ** 2 * step_size_p) / (sigma_x ** 2 + 1e-10)
            )

            pred_x_p_scaled = pred_x_p * torch.sqrt(sigma_norm)
            if cfg['guide_coords']:
                pred_x_p_guided = pred_x_p_scaled + g_scale * pg_coord
            else:
                pred_x_p_guided = pred_x_p_scaled

            x_t_next = (x_t_half - step_size_p * pred_x_p_guided
                        + std_x_p * rand_x
                        if not self.keep_coords else x_t)


            if not self.keep_lattice:
                pred_l_adj = pred_l_p
                if cfg['guide_lattice'] and g_scale > 1e-10:
                    pred_l_adj = (pred_l_p
                        - g_scale * cfg['lattice_guide_weight'] * pg_lat)
                l_t_next = (c0 * (l_t_half - c1 * pred_l_adj)
                            + beta_sigma * rand_l)
            else:
                l_t_next = l_t


            traj[t - 1] = {
                'num_atoms':   batch.num_atoms,
                'atom_types':  batch.atom_types,
                'frac_coords': x_t_next % 1.,
                'lattices':    l_t_next,
            }


        traj_stack = {
            'num_atoms':      batch.num_atoms,
            'atom_types':     batch.atom_types,
            'all_frac_coords': torch.stack(
                [traj[i]['frac_coords'] for i in range(time_start, -1, -1)]
            ),
            'all_lattices':    torch.stack(
                [traj[i]['lattices'] for i in range(time_start, -1, -1)]
            ),
            'energy_trajectory': energy_trajectory,
        }

        return traj[0], traj_stack


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss