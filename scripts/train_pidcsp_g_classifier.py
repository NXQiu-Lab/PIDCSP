import argparse
import os
import sys
import math

sys.path.insert(0, '.')

import torch
import pytorch_lightning as pl


PL_VER = pl.__version__
PL_MAJOR, PL_MINOR = (int(x) for x in PL_VER.split('.')[:2])
print(f"[ENV] PyTorch Lightning : {PL_VER}")
print(f"[ENV] PyTorch           : {torch.__version__}")
print(f"[ENV] Python            : {sys.version.split()[0]}")
print(f"[ENV] CUDA 可用         : {torch.cuda.is_available()}")

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from pidcsp.pl_modules.classifier.pidcsp_g_classifier_trainer import ClassifierTrainer
from pidcsp.pl_data.datamodule_conditional import ConditionalCrystDataModule


def build_trainer(args, callbacks) -> pl.Trainer:


    use_gpu = torch.cuda.is_available()


    trainer_kwargs = dict(
        max_epochs              = args.max_epochs,
        callbacks               = callbacks,
        gradient_clip_val       = 1.0,
        log_every_n_steps       = 20,
        check_val_every_n_epoch = 1,


        progress_bar_refresh_rate = 50,

        weights_summary           = None,
    )

    if use_gpu:
        trainer_kwargs['gpus'] = 1


    print(f"[Trainer] PL {PL_VER}，GPU={'是' if use_gpu else '否'}")
    print(f"[Trainer] 参数: "
          f"{ {k: v for k, v in trainer_kwargs.items() if k != 'callbacks'} }")

    return pl.Trainer(**trainer_kwargs)


def build_datamodule(args) -> ConditionalCrystDataModule:

    from omegaconf import OmegaConf

    prop_keys = [x.strip() for x in args.properties.split(',') if x.strip()]
    log_transform_props = [
        x.strip() for x in args.log_transform_properties.split(',') if x.strip()
    ]

    common = dict(
        prop                 = prop_keys[0],
        niggli               = True,
        primitive            = False,
        graph_method         = 'crystalnn',
        tolerance            = 0.1,
        lattice_scale_method = 'scale_length',
        preprocess_workers   = args.preprocess_workers,
        use_space_group      = False,
        use_pos_index        = False,
        use_conditions       = True,
        prop_stats_path      = None,
        prop_keys            = prop_keys,
        log_transform_props  = log_transform_props,
    )

    datasets_cfg = OmegaConf.create({
        'train': {
            '_target_': (
                'pidcsp.pl_data.dataset_conditional'
                '.ConditionalCrystDataset'
            ),
            'name':      'Conditional train',
            'path':      f"{args.data_root}/train.csv",
            'save_path': f"{args.data_root}/train_ori.pt",
            **common,
        },
        'val': [{
            '_target_': (
                'pidcsp.pl_data.dataset_conditional'
                '.ConditionalCrystDataset'
            ),
            'name':      'Conditional val',
            'path':      f"{args.data_root}/val.csv",
            'save_path': f"{args.data_root}/val_ori.pt",
            **common,
        }],
        'test': [{
            '_target_': (
                'pidcsp.pl_data.dataset_conditional'
                '.ConditionalCrystDataset'
            ),
            'name':      'Conditional test',
            'path':      f"{args.data_root}/test.csv",
            'save_path': f"{args.data_root}/test_ori.pt",
            **common,
        }],
    })

    num_workers_cfg = OmegaConf.create(
        {'train': 8, 'val': 4, 'test': 4})
    batch_size_cfg = OmegaConf.create({
        'train': args.batch_size,
        'val':   min(args.batch_size, 128),
        'test':  min(args.batch_size, 128),
    })

    prop_stats_save = os.path.join(args.save_dir, 'prop_stats_cache')

    dm = ConditionalCrystDataModule(
        datasets             = datasets_cfg,
        num_workers          = num_workers_cfg,
        batch_size           = batch_size_cfg,
        prop_stats_save_path = prop_stats_save,
    )
    return dm


def main(args):
    pl.seed_everything(42)

    os.makedirs(args.save_dir, exist_ok=True)


    print("\n[Setup] 构建DataModule...")
    dm = build_datamodule(args)


    print("\n[Setup] 构建ClassifierTrainer...")
    trainer_module = ClassifierTrainer(
        pretrained_ckpt_path = args.backbone_ckpt,
        hidden_dim           = 512,
        time_dim             = 256,
        num_freqs            = 128,
        edge_style           = 'fc',
        clf_hidden_dim       = 256,
        n_mp_layers          = 3,
        num_heads            = 4,
        dropout              = 0.0,
        alpha                = 0.2,
        phys_edge_weight     = 0.5,
        use_time_phys_mod    = True,
        time_mod_min         = 0.7,
        lr                   = args.lr,
        weight_decay         = 1e-5,
        t_warmup_steps       = 1000,
        low_t_ratio          = 0.4,
        prop_keys = [x.strip() for x in args.properties.split(',') if x.strip()],
        prop_loss_weights = {k.strip(): 1.0 for k in args.properties.split(',') if k.strip()},
        log_transform_props = [
            x.strip() for x in args.log_transform_properties.split(',') if x.strip()
        ],
        use_max_features = args.use_max_features,
        include_b_as_x = args.include_b_as_x,
    )


    callbacks = [
        ModelCheckpoint(
            dirpath    = args.save_dir,
            filename   = 'clf-{epoch:03d}-val{val_loss:.4f}',
            monitor    = 'val_loss',
            mode       = 'min',
            save_top_k = 3,
            save_last  = True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor  = 'val_loss',
            patience = 30,
            mode     = 'min',
        ),
    ]


    trainer = build_trainer(args, callbacks)


    print(f"\n[Train] 开始训练，最大epoch={args.max_epochs}")
    trainer.fit(trainer_module, datamodule=dm)


    final_path = os.path.join(args.save_dir, 'classifier_final.pt')
    trainer_module.save_classifier(final_path)

    print(f"\n[Done]")
    print(f"  分类器:     {final_path}")
    print(f"  prop_stats: "
          f"{args.save_dir}/prop_stats_cache/prop_stats.pt")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--backbone_ckpt',
                   required=True,
                   help='预训练扩散模型checkpoint路径')
    p.add_argument('--data_root',
                   required=True,
                   help='数据集根目录（含train/val/test.csv）')
    p.add_argument('--save_dir',
                   required=True,
                   help='分类器checkpoint保存目录')
    p.add_argument('--max_epochs',
                   type=int,   default=200)
    p.add_argument('--batch_size',
                   type=int,   default=512)
    p.add_argument('--lr',
                   type=float, default=3e-4)
    p.add_argument('--preprocess_workers',
                   type=int,   default=4)
    p.add_argument('--properties', default='formation_energy_per_atom,band_gap',
                   help='逗号分隔的监督属性列名')
    p.add_argument('--log-transform-properties', dest='log_transform_properties',
                   default='band_gap', help='需要log1p后标准化的属性列名；可为空')
    p.add_argument('--use-max-features', action='store_true',
                   help='启用可微M/A/X角色与211/312/413特征')
    p.add_argument('--include-b-as-x', action='store_true',
                   help='将B作为扩展MAX体系的X位元素')
    main(p.parse_args())
