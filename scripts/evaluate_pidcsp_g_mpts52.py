import time
import argparse
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Batch

from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr


SAMPLING_CONFIGS = {
    1: {
        'sampling_temperature': 0.8,
        'cross_attn_temperature': 0.9,
        'step_lr_scale': 1.0,
        'description': 'Single sampling - High precision mode'
    },
    10: {
        'sampling_temperature': 1.2,
        'cross_attn_temperature': 1.05,
        'step_lr_scale': 1.1,
        'description': 'Medium sampling - Balanced mode'
    },
    20: {
        'sampling_temperature': 1.4,
        'cross_attn_temperature': 1.1,
        'step_lr_scale': 1.2,
        'description': 'Multi sampling - High diversity mode'
    },
}


def get_sampling_config(num_evals):


    if num_evals in SAMPLING_CONFIGS:
        return SAMPLING_CONFIGS[num_evals]


    keys = sorted(SAMPLING_CONFIGS.keys())
    selected = keys[0]
    for k in keys:
        if num_evals >= k:
            selected = k

    return SAMPLING_CONFIGS[selected]


def configure_model_for_sampling(model, num_evals, manual_temperature=None):


    config = get_sampling_config(num_evals)
    decoder = model.decoder

    print(f"\n{'='*60}")
    print(f"Configuring model for {num_evals} samples")
    print(f"Mode: {config['description']}")
    print(f"{'='*60}")


    if manual_temperature is not None and manual_temperature > 0:

        sampling_temp = manual_temperature
        print(f"[Manual] sampling_temperature: {sampling_temp}")
    else:
        sampling_temp = config['sampling_temperature']
        print(f"[Auto] sampling_temperature: {sampling_temp}")


    if hasattr(decoder, 'sampling_temperature'):
        decoder.sampling_temperature = sampling_temp
    elif hasattr(decoder, 'set_sampling_temperature'):
        decoder.set_sampling_temperature(sampling_temp)
    else:
        print("Warning: Model does not support sampling_temperature, skipping...")


    cross_attn_temp = config['cross_attn_temperature']


    layers_configured = 0
    for i in range(getattr(decoder, 'num_layers', 6)):
        layer = decoder._modules.get(f"csp_layer_{i}")
        if layer is not None and hasattr(layer, 'cross_attention'):
            if hasattr(layer.cross_attention, 'temperature'):
                layer.cross_attention.temperature = cross_attn_temp
                layers_configured += 1

    if layers_configured > 0:
        print(f"[Auto] cross_attn_temperature: {cross_attn_temp} (applied to {layers_configured} layers)")


    print(f"[Auto] step_lr_scale: {config['step_lr_scale']}")
    print(f"{'='*60}\n")

    return config


def add_sampling_noise(model, base_temperature, noise_scale=0.1):


    noise = noise_scale * (np.random.rand() - 0.5) * 2
    actual_temperature = base_temperature * (1 + noise)
    actual_temperature = max(0.5, actual_temperature)

    decoder = model.decoder
    if hasattr(decoder, 'sampling_temperature'):
        decoder.sampling_temperature = actual_temperature

    return actual_temperature


def sample_pidcsp_g_mpts52(loader, model, num_evals, step_lr=1e-5, add_noise=True):


    base_temperature = getattr(model.decoder, 'sampling_temperature', 1.0)

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()

        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []

        for eval_idx in range(num_evals):
            print(f'batch {idx}/{len(loader)}, sample {eval_idx + 1}/{num_evals}', end='')


            if add_noise and num_evals > 1:
                actual_temp = add_sampling_noise(model, base_temperature, noise_scale=0.1)
                print(f' (temp={actual_temp:.3f})')
            else:
                print()


            outputs, traj = model.sample(batch, step_lr=step_lr)


            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())


            if add_noise and num_evals > 1:
                model.decoder.sampling_temperature = base_temperature

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))
        input_data_list = input_data_list + batch.to_data_list()


    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch
    )


def main(args):

    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(model_path, load_data=True)

    if torch.cuda.is_available():
        model.to('cuda')

    print('='*60)
    print('PIDCSP-G Diffusion Evaluation')
    print('='*60)


    config = configure_model_for_sampling(
        model,
        args.num_evals,
        manual_temperature=args.temperature if args.temperature > 0 else None
    )


    if args.step_lr >= 0:
        base_step_lr = args.step_lr
    else:

        lr_key = 'csp' if args.num_evals == 1 else 'csp_multi'
        base_step_lr = recommand_step_lr[lr_key][args.dataset]


    step_lr = base_step_lr * config['step_lr_scale']
    print(f"Final step_lr: {step_lr:.2e} (base: {base_step_lr:.2e}, scale: {config['step_lr_scale']})")


    start_time = time.time()

    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch) = sample_pidcsp_g_mpts52(
        test_loader,
        model,
        args.num_evals,
        step_lr,
        add_noise=(not args.no_noise)
    )

    elapsed_time = time.time() - start_time


    if args.label == '':
        diff_out_name = 'eval_diff.pt'
    else:
        diff_out_name = f'eval_diff_{args.label}.pt'

    save_dict = {
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles,
        'time': elapsed_time,

        'sampling_config': {
            'num_evals': args.num_evals,
            'sampling_temperature': config['sampling_temperature'],
            'cross_attn_temperature': config['cross_attn_temperature'],
            'step_lr_scale': config['step_lr_scale'],
            'final_step_lr': step_lr,
        }
    }

    torch.save(save_dict, model_path / diff_out_name)

    print(f"\n{'='*60}")
    print(f"Results saved to: {model_path / diff_out_name}")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIDCSP-G-mpts52 diffusion evaluation')


    parser.add_argument('--model_path', required=True, type=str,
                        help='Path to the trained model')
    parser.add_argument('--dataset', required=True, type=str,
                        help='Dataset name (e.g., mp20, carbon, perovskite)')


    parser.add_argument('--num_evals', default=1, type=int, choices=[1, 10, 20],
                        help='Number of samples per structure (1, 10, or 20)')
    parser.add_argument('--step_lr', default=-1, type=float,
                        help='Step learning rate (-1 for auto)')


    parser.add_argument('--temperature', default=-1, type=float,
                        help='Manual sampling temperature (-1 for auto based on num_evals)')
    parser.add_argument('--no_noise', action='store_true',
                        help='Disable temperature noise for multi-sampling')


    parser.add_argument('--label', default='', type=str,
                        help='Label for output file')

    args = parser.parse_args()
    main(args)
