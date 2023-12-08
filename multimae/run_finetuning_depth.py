from rgb_d_trainer import RgbDepthTrainer
import torch
import numpy as np
import os
import torch.nn as nn
from functools import partial
from input_adapters import PatchedInputAdapter
from dataset import CustomDataset
from torchvision import transforms
from output_adapters import DPTOutputAdapter, ConvNeXtAdapter
from multimae import multivit_base
from pos_embed_multi import interpolate_pos_embed_multimae
from pathlib import Path
from datetime import datetime
import logging



if __name__ == '__main__':
    run_name = "multimae-test"
    log_dir = './logs/'
    run_name += f"_{datetime.now():%F_%T}"

    ### SETUP LOGGING ###
    ch = logging.StreamHandler()
    cons_lvl = getattr(logging, 'DEBUG')
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{run_name}.log"
    fh = logging.FileHandler(log_file)
    file_lvl = getattr(logging, 'DEBUG')
    fh.setLevel(file_lvl)
    ffmt = logging.Formatter(
        "{levelname:8} - {process: ^6} - {name: ^16} - {asctime} - {message}",
        style="{",
    )
    fh.setFormatter(ffmt)

    logger = logging.getLogger("MultiMAE")
    logger.setLevel(min(file_lvl, cons_lvl))
    logger.addHandler(ch)
    logger.addHandler(fh)


    device = 'cuda'
    DOMAIN_CONF = {
        'rgb': {
            'channels': 3,
            'stride_level': 1,
            'input_adapter': partial(PatchedInputAdapter, num_channels=3),
            'aug_type': 'image',
        },
        'depth': {
            'channels': 1,
            'stride_level': 1,
            'input_adapter': partial(PatchedInputAdapter, num_channels=1),
            'aug_type': 'mask',
        },
        'mask_valid': {
            'stride_level': 1,
            'aug_type': 'mask',
        },
    }


    in_domains = ['rgb']
    out_domains = ['depth']
    all_domains = list(set(in_domains) | set(out_domains))

    patch_size = 16
    input_size = 224

    input_adapters = {
            domain: DOMAIN_CONF[domain]['input_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=patch_size,
                image_size=input_size,
            )
            for domain in in_domains
        }

    decoder_main_tasks = ['rgb']

    additional_targets = {domain: DOMAIN_CONF[domain]['aug_type'] for domain in all_domains}

    # DPT settings are fixed for ViT-B. Modify them if using a different backbone.

    adapters_dict = {
        'dpt': DPTOutputAdapter,
        'convnext': partial(ConvNeXtAdapter, preds_per_patch=64),
    }

    output_adapter = 'dpt'

    output_adapters = {
        domain: adapters_dict[output_adapter](
            num_classes=DOMAIN_CONF[domain]['channels'],
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size=patch_size,
            main_tasks=decoder_main_tasks
        )
        for domain in out_domains
    }

    model_name = 'multivit_base'
    drop_path_encoder = 0.0
    model = multivit_base(input_adapters=input_adapters, output_adapters=output_adapters)


    finetune_path = '../data/mae-b_dec512d8b_1600e_multivit-c477195b.pth'
    checkpoint = torch.load(finetune_path, map_location='cpu')

    checkpoint_model = checkpoint['model']

    # # Remove keys for semantic segmentation
    # for k in list(checkpoint_model.keys()):
    #     if "semseg" in k:
    #         del checkpoint_model[k]

    # Remove output adapters
    for k in list(checkpoint_model.keys()):
        if "output_adapters" in k:
            del checkpoint_model[k]

    # Interpolate position embedding
    interpolate_pos_embed_multimae(model, checkpoint_model)

    # Load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    lr = 3e-4
    decay = 1e-4
    opt = torch.optim.Adam(
        model.parameters(), lr, weight_decay=decay
    )


    # Example usage:
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

    # Create instances of the dataset for training and validation
    train_dataset = CustomDataset(root_dir='../data/clevr_complex/train', transform=transform)
    val_dataset = CustomDataset(root_dir='../data/clevr_complex/val', transform=transform)

    bs = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)


    total_epochs = 5
    trainer = RgbDepthTrainer(
        model=model,
        optimizer=opt,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=run_name,
        ckpt_dir=Path(log_dir).parent / "ckpt",
        ckpt_replace=True,
        ckpt_resume=None,
        ckpt_track_metric='loss',
        metrics_on_train=False,
        total_epochs=total_epochs,
        device=device,
    )

    trainer.run()