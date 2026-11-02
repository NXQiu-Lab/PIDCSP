import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Optional, Any
import math

from pidcsp.common.data_utils import lattice_params_to_matrix_torch
from pidcsp.common.conditional_data_utils import PROPERTY_KEYS

MAX_ATOMIC_NUM = 100


LOG_TRANSFORM_PROPS = {"band_gap"}


def _apply_log(val: float, key: str) -> float:
    if key in LOG_TRANSFORM_PROPS:
        return math.log1p(max(val, 0.0))
    return val


class ClassifierGuidanceSampler:


    def __init__(
        self,
        backbone,
        classifier:             Any,
        prop_stats:             Dict = None,
        guidance_scale_lattice: float = 1.0,
        guidance_scale_coord:   float = 0.5,
        grad_clip_lattice:      float = 0.3,
        grad_clip_coord:        float = 0.05,
        t_guidance_end_ratio:   float = 0.8,
        step_lr:                float = 1e-5,
        n_corrector_steps:      int   = 1,
        log_transform_props = None,
        verbose:                bool  = True,
    ):
        self.backbone        = backbone
        self.classifier      = classifier
        self.prop_stats      = prop_stats or {}
        self.s_l             = guidance_scale_lattice
        self.s_x             = guidance_scale_coord
        self.clip_l          = grad_clip_lattice
        self.clip_x          = grad_clip_coord


        self.t_guide_end     = t_guidance_end_ratio
        self.step_lr         = step_lr
        self.n_corrector     = n_corrector_steps
        self.verbose         = verbose
        classifier_log_props = getattr(classifier, 'log_transform_props', None)
        if classifier_log_props is None:
            classifier_log_props = [
                k for k, value in self.prop_stats.items()
                if isinstance(value, dict) and value.get('log_transformed', False)
            ]
        if log_transform_props is not None:
            self.log_transform_props = set(log_transform_props)
        elif classifier_log_props:
            self.log_transform_props = set(classifier_log_props)
        else:
            self.log_transform_props = set(LOG_TRANSFORM_PROPS)

        self.beta_scheduler  = backbone.beta_scheduler
        self.sigma_scheduler = backbone.sigma_scheduler
        self.timesteps       = self.beta_scheduler.timesteps
        self.keep_lattice    = backbone.keep_lattice
        self.keep_coords     = backbone.keep_coords

    def _normalize_target(self, key: str, raw_val: float) -> float:


        val = (math.log1p(max(raw_val, 0.0))
               if key in self.log_transform_props else raw_val)
        if key in self.prop_stats:
            s    = self.prop_stats[key]
            mean = float(s.get('mean', 0.0))
            std  = max(float(s.get('std',  1.0)), 1e-8)
            val  = float((val - mean) / std)
        return val

    def _compute_guidance_gradient(
        self,
        atom_type_probs: torch.Tensor,
        frac_coords:     torch.Tensor,
        lattice:         torch.Tensor,
        num_atoms:       torch.Tensor,
        node2graph:      torch.Tensor,
        t_tensor:        torch.Tensor,
        target_props:    Dict[str, float],
        device,
    ) -> Dict[str, Optional[torch.Tensor]]:


        B = lattice.shape[0]

        lat_wg = lattice.detach().clone().requires_grad_(True)
        crd_wg = frac_coords.detach().clone().requires_grad_(True)

        preds = self.classifier(
            atom_type_probs = atom_type_probs.detach(),
            frac_coords     = crd_wg,
            lattice         = lat_wg,
            num_atoms       = num_atoms,
            node2graph      = node2graph,
            t               = t_tensor,
        )

        guide_loss = torch.tensor(0.0, device=device)
        n_active   = 0

        for key, pred in preds.items():
            if key not in target_props:
                continue
            target_normed = self._normalize_target(key, target_props[key])
            target_tensor = torch.full(
                (B, 1), target_normed,
                dtype=torch.float32, device=device)
            guide_loss = guide_loss + F.mse_loss(pred, target_tensor)
            n_active  += 1

        if n_active == 0:
            return {'grad_lattice': None, 'grad_coords': None}

        guide_loss.backward()

        grads = {}

        if lat_wg.grad is not None:
            g       = lat_wg.grad.detach().clone()
            g_norms = g.view(B, -1).norm(dim=-1, keepdim=True).unsqueeze(-1)
            scale   = (self.clip_l / g_norms.clamp(min=1e-6)).clamp(max=1.0)
            grads['grad_lattice'] = g * scale
        else:
            grads['grad_lattice'] = None

        if crd_wg.grad is not None:
            g      = crd_wg.grad.detach().clone()
            g_norm = g.norm()
            if g_norm > self.clip_x:
                g = g * (self.clip_x / g_norm)
            grads['grad_coords'] = g
        else:
            grads['grad_coords'] = None

        del lat_wg, crd_wg, preds, guide_loss
        return grads

    def _apply_guidance_to_sample(self, l_t, x_t, grads, t):
        l_guided = l_t.clone()
        x_guided = x_t.clone()

        if grads['grad_lattice'] is not None and not self.keep_lattice:
            l_guided = l_guided - self.s_l * grads['grad_lattice']

        if grads['grad_coords'] is not None and not self.keep_coords:
            x_guided = x_guided - self.s_x * grads['grad_coords']
            x_guided = x_guided % 1.0

        return l_guided, x_guided

    def _backbone_corrector_step(
        self, time_emb, t_t, x_t, l_t, num_atoms, node2graph,
        sigma_x, sigma_norm, step_lr,
    ):
        with torch.no_grad():
            _, pred_x, _ = self.backbone.decoder(
                time_emb, t_t, x_t, l_t, num_atoms, node2graph)

        step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
        std_x     = (2 * step_size) ** 0.5
        rand_x    = torch.randn_like(x_t)

        pred_x_scaled = pred_x * (sigma_norm ** 0.5)
        x_out = x_t - step_size * pred_x_scaled + std_x * rand_x
        return x_out

    def _backbone_predictor_step(
        self, t, time_emb, t_t, x_t, l_t, num_atoms, node2graph,
        alphas, alphas_bar, sigmas_ddpm,
        sigma_x, adj_sigma_x, sigma_norm,
        l_T, x_T, t_T,
    ):
        c0 = 1.0 / (alphas ** 0.5)
        c1 = (1 - alphas) / ((1 - alphas_bar) ** 0.5)

        rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)

        with torch.no_grad():
            pred_l, pred_x, pred_t = self.backbone.decoder(
                time_emb, t_t, x_t, l_t, num_atoms, node2graph)

        step_size2 = sigma_x ** 2 - adj_sigma_x ** 2
        std_x2     = ((adj_sigma_x ** 2 * step_size2) / (sigma_x ** 2)) ** 0.5

        pred_x_scaled = pred_x * (sigma_norm ** 0.5)
        x_out = x_t - step_size2 * pred_x_scaled + std_x2 * rand_x
        l_out = (c0 * (l_t - c1 * pred_l) + sigmas_ddpm * rand_l
                 if not self.keep_lattice else l_t)
        t_out = c0 * (t_t - c1 * pred_t) + sigmas_ddpm * rand_t

        return x_out, l_out, t_out

    def sample(
        self,
        batch,
        target_props: Dict[str, float],
        device,
        guidance_scale_lattice: float = None,
        guidance_scale_coord:   float = None,
    ) -> tuple:
        if guidance_scale_lattice is not None:
            self.s_l = guidance_scale_lattice
        if guidance_scale_coord is not None:
            self.s_x = guidance_scale_coord

        self.backbone.eval()
        self.classifier.eval()

        B          = batch.num_graphs
        node2graph = batch.batch.to(device)

        l_T = torch.randn([B, 3, 3], device=device)
        x_T = torch.rand([batch.num_nodes, 3], device=device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM], device=device)

        if self.keep_coords:
            x_T = batch.frac_coords.to(device)
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(
                batch.lengths, batch.angles).to(device)

        traj = {self.timesteps: {
            'num_atoms':   batch.num_atoms,
            'atom_types':  t_T,
            'frac_coords': x_T % 1.0,
            'lattices':    l_T,
        }}

        t_guide_start = int(self.timesteps * (1.0 - self.t_guide_end))

        if self.verbose:
            props_str = ', '.join(
                f"{k.split('_')[0]}={v:.3f}"
                for k, v in target_props.items())
            iterator = tqdm(
                range(self.timesteps, 0, -1),
                desc=f"Guided [{props_str}]",
                ncols=100,
            )
        else:
            iterator = range(self.timesteps, 0, -1)

        for t in iterator:
            times    = torch.full((B,), t, device=device)
            time_emb = self.backbone.time_embedding(times)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            alphas         = float(self.beta_scheduler.alphas[t])
            alphas_bar     = float(self.beta_scheduler.alphas_cumprod[t])
            sigmas_ddpm    = float(self.beta_scheduler.sigmas[t])
            sigma_x        = float(self.sigma_scheduler.sigmas[t])
            sigma_norm     = float(self.sigma_scheduler.sigmas_norm[t])
            adj_sigma_x    = float(self.sigma_scheduler.sigmas[t-1])

            if t >= t_guide_start and len(target_props) > 0:
                grads = self._compute_guidance_gradient(
                    atom_type_probs = t_t,
                    frac_coords     = x_t,
                    lattice         = l_t,
                    num_atoms       = batch.num_atoms,
                    node2graph      = node2graph,
                    t_tensor        = times,
                    target_props    = target_props,
                    device          = device,
                )
                l_t_guided, x_t_guided = self._apply_guidance_to_sample(
                    l_t, x_t, grads, t)
            else:
                grads      = {'grad_lattice': None, 'grad_coords': None}
                l_t_guided = l_t
                x_t_guided = x_t

            x_t_m05 = self._backbone_corrector_step(
                time_emb, t_t,
                x_t_guided if not self.keep_coords  else x_t,
                l_t_guided if not self.keep_lattice else l_t,
                batch.num_atoms, node2graph,
                sigma_x, sigma_norm, self.step_lr,
            ) if not self.keep_coords else x_t_guided

            x_t_m1, l_t_m1, t_t_m1 = self._backbone_predictor_step(
                t, time_emb, t_t,
                x_t_m05 if not self.keep_coords  else x_t,
                l_t_guided if not self.keep_lattice else l_t,
                batch.num_atoms, node2graph,
                alphas, alphas_bar, sigmas_ddpm,
                sigma_x, adj_sigma_x, sigma_norm,
                l_T, x_T, t_T,
            )

            traj[t-1] = {
                'num_atoms':   batch.num_atoms,
                'atom_types':  t_t_m1,
                'frac_coords': x_t_m1 % 1.0,
                'lattices':    l_t_m1,
            }

        traj_stacked = {
            'num_atoms':       batch.num_atoms,
            'atom_types':      torch.stack([
                traj[i]['atom_types']
                for i in range(self.timesteps, -1, -1)
            ]).argmax(dim=-1) + 1,
            'all_frac_coords': torch.stack([
                traj[i]['frac_coords']
                for i in range(self.timesteps, -1, -1)
            ]),
            'all_lattices':    torch.stack([
                traj[i]['lattices']
                for i in range(self.timesteps, -1, -1)
            ]),
        }

        return traj[0], traj_stacked
