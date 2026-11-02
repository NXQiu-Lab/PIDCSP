# PIDCSP

PIDCSP is a physics-informed diffusion framework for crystal structure prediction and generation. It combines periodic crystal diffusion with element-level physical descriptors and graph attention, and provides three complete workflows: composition-conditioned prediction (`PIDCSP-C`), de novo generation (`PIDCSP-G`), and property-guided generation (`PIDCSP-G + classifier`).

## Functional modes

| Mode | Known input | Generated variables | Entry configuration | Main script |
| --- | --- | --- | --- | --- |
| PIDCSP-C | composition and atom types | lattice and fractional coordinates | `model=pidcsp_c` | `scripts/evaluate_pidcsp_c.py` |
| PIDCSP-G | atom count or dataset prior | atom types, lattice, and fractional coordinates | `model=pidcsp_g` | `scripts/generate_pidcsp_g.py` |
| PIDCSP-G + classifier | generation prior and target properties | property-guided crystal candidates | PIDCSP-G checkpoint plus classifier checkpoint | `scripts/generate_pidcsp_g_classifier.py` |

The public model name uses hyphens. Python modules and Hydra configuration files use underscores because hyphens are not valid Python identifiers.

## How the three workflows run

### PIDCSP-C

PIDCSP-C predicts a stable structure for a known composition. The data pipeline reads crystal records, constructs periodic graphs, standardizes lattice parameters, and keeps atom types fixed. During training, Gaussian noise is added to lattice matrices and wrapped noise is added to fractional coordinates. `PIDCSPDiffusion` sends the noisy structure, diffusion time, atom types, graph membership, and lattice state to `PIDCSPNet`. The network predicts lattice noise and coordinate score terms. The objective combines lattice and coordinate losses. During sampling, the reverse process alternates lattice denoising and periodic Langevin coordinate updates until a candidate crystal is recovered.

The default PIDCSP-C decoder is the physics-fusion PIDCSP-G backbone. This lets composition-conditioned prediction use the same element descriptors, graph attention, and physical fusion mechanism as the generation branch while leaving atom types fixed.

### PIDCSP-G

PIDCSP-G performs ab initio generation. It follows the same joint lattice-coordinate diffusion process as PIDCSP-C and adds a continuous noisy representation of atom types. The decoder predicts lattice noise, coordinate scores, and atom-type logits. Training therefore adds the atom-type objective to the lattice and coordinate objectives. Reverse diffusion progressively reconstructs all three variables and converts the final type distribution to discrete atomic numbers.

The PIDCSP-G backbone loads cached element properties together with periodic-table group and period information. These descriptors are projected into node and edge features, combined with distance embeddings, and passed through graph-attention message-passing layers. Time-dependent modulation controls how strongly physical information affects the network at each diffusion step.

Three decoder profiles are included:

- `pidcsp_g`: the default PIDCSP-G edge-guided graph-attention decoder.
- `pidcsp_g_fusion`: the PIDCSP-G node-edge fusion decoder used by the default PIDCSP-C configuration.
- `pidcsp_g_mpts52`: the PIDCSP-G-mpts52 decoder for the larger MPTS-52 structures.

### PIDCSP-G + classifier

This workflow separates generative learning from property guidance. First, a trained PIDCSP-G checkpoint is frozen. A time-aware crystal classifier is then trained on noisy structures using the same diffusion schedule and compatible graph features. The current implementation supervises `formation_energy_per_atom` and `band_gap`; band gap is transformed with `log1p` before normalization.

At generation time, every reverse-diffusion step first obtains the ordinary PIDCSP-G update. While guidance is active, the classifier evaluates the noisy candidate and differentiates the selected property objective with respect to lattice and fractional coordinates. The clipped gradients are scaled independently and added to the reverse update. Guidance is disabled near the end of the trajectory according to `t_guidance_end_ratio`, allowing the base model to finish local structural refinement. The CIF workflow writes individual candidates, while the PT workflow supports per-record caching and interrupted-run recovery.

## Repository layout

```text
PIDCSP-main/
├── pidcsp/
│   ├── common/                  data processing, periodic geometry, utilities
│   ├── pl_data/                 standard and conditional data modules
│   ├── pl_modules/
│   │   ├── classifier/          classifier training and guided sampling
│   │   ├── pidcsp_c_diffusion.py
│   │   ├── pidcsp_g_diffusion.py
│   │   ├── pidcsp_g.py
│   │   ├── pidcsp_g_fusion.py
│   │   └── pidcsp_g_mpts52.py
│   ├── prop_models/             pretrained property-model assets
│   └── run.py                   Hydra training entry point
├── conf/                        datasets, models, schedulers, and training options
├── scripts/                     training, sampling, evaluation, and metrics
├── cgcnn/                       external property evaluation utilities
├── environment.yaml
└── requirements.txt
```

## Installation

The reference environment uses Python 3.8.13, PyTorch 1.9.0, PyTorch Geometric 1.7.2, PyTorch Lightning 1.3.8, and pymatgen 2023.8.10.

```bash
conda env create -f environment.yaml
conda activate pidcsp
```

Alternatively, install the pinned packages with:

```bash
pip install -r requirements.txt
```

Copy the environment template and adjust paths only when the repository is not launched from its root:

```bash
cp .env.example .env
```

The default values are:

```dotenv
PROJECT_ROOT=.
HYDRA_JOBS=./hydra
WANDB_DIR=./wandb
```

Prepare the physical-property cache before using a physics-informed decoder:

```bash
python scripts/prepare_pidcsp_physics.py
```

This creates `embeddings/phys_embedding_data.pt`.

## Data

Place each dataset under `data/<dataset>/` with `train.csv`, `val.csv`, and `test.csv`. Available Hydra dataset configurations are `perov_5`, `mp_20`, `mpts_52`, `carbon_24`, `mp_20_condition`, and `max`. Processed tensors and property statistics are cached beside the CSV files.

## Training

Train PIDCSP-C:

```bash
python pidcsp/run.py data=mp_20 model=pidcsp_c expname=pidcsp_c_mp20
```

Train PIDCSP-G:

```bash
python pidcsp/run.py data=mp_20 model=pidcsp_g expname=pidcsp_g_mp20
```

Train the MPTS-52 profile:

```bash
python pidcsp/run.py data=mpts_52 model=pidcsp_g model/decoder=pidcsp_g_mpts52 expname=pidcsp_g_mpts52
```

Train the property classifier after PIDCSP-G:

```bash
python scripts/train_pidcsp_g_classifier.py \
  --backbone_ckpt hydra/singlerun/<date>/<run>/last.ckpt \
  --data_root data/mp_20 \
  --save_dir classifier_ckpts/pidcsp_g \
  --properties formation_energy_per_atom,band_gap
```

Hydra writes checkpoints, scalers, and resolved `hparams.yaml` files under `HYDRA_JOBS`.

## Sampling and evaluation

Evaluate PIDCSP-C:

```bash
python scripts/evaluate_pidcsp_c.py --model_path <run_or_checkpoint> --dataset mp_20
python scripts/compute_metrics.py --root_path <run_directory> --tasks csp --gt_file data/mp_20/test.csv
```

For multiple samples per composition, pass `--num_evals 20` to the evaluation command and `--multi_eval` to the metrics command.

Generate with PIDCSP-G:

```bash
python scripts/generate_pidcsp_g.py --model_path <run_or_checkpoint> --dataset mp_20
python scripts/compute_metrics.py --root_path <run_directory> --tasks gen --gt_file data/mp_20/test.csv
```

Evaluate the MPTS-52 profile:

```bash
python scripts/evaluate_pidcsp_g_mpts52.py \
  --model_path <run_or_checkpoint> \
  --dataset mpts_52 \
  --num_evals 20
```

Run classifier-guided generation and write CIF candidates:

```bash
python scripts/generate_pidcsp_g_classifier.py \
  --model_path <pidcsp_g_checkpoint> \
  --classifier_ckpt <classifier_checkpoint> \
  --test_csv data/mp_20/test.csv \
  --save_path guided_results \
  --properties formation_energy_per_atom,band_gap
```

Run the resumable PT-output workflow:

```bash
python scripts/generate_pidcsp_g_classifier_pt.py \
  --model_path <pidcsp_g_checkpoint> \
  --classifier_ckpt <classifier_checkpoint> \
  --test_csv data/mp_20/test.csv \
  --save_path guided_pt_results \
  --props formation_energy_per_atom band_gap
```

## Reproducibility notes

- A decoder checkpoint must be loaded with the same decoder profile and dimensions used for training.
- Classifier `hidden_dim`, `time_dim`, `num_freqs`, edge settings, and physical-data settings must match its PIDCSP-G backbone.
- Keep the saved `lattice_scaler.pt`, `prop_scaler.pt`, and classifier property-statistics cache with each checkpoint.
- Generation guidance scales and clipping thresholds affect property strength and structural validity; report them with evaluation results.

## Acknowledgments

PIDCSP is developed from the open-source [DiffCSP](https://github.com/jiaor17/DiffCSP) implementation and retains its joint periodic diffusion foundation. The original framework also builds on [CDVAE](https://github.com/txie-93/cdvae). Perov-5, Carbon-24, and MP-20 originate from CDVAE, while MPTS-52 follows the MP time-split dataset.

Please cite the upstream DiffCSP work when this inherited foundation is used:

```bibtex
@article{jiao2023crystal,
  title={Crystal structure prediction by joint equivariant diffusion},
  author={Jiao, Rui and Huang, Wenbing and Lin, Peijia and Han, Jiaqi and Chen, Pin and Lu, Yutong and Liu, Yang},
  journal={arXiv preprint arXiv:2309.04475},
  year={2023}
}
```

Add the PIDCSP paper citation here when its public bibliographic record is available.
