from rgb_d_trainer import RgbDepthTrainer
import torch
from functools import partial
from input_adapters import PatchedInputAdapter, SemSegInputAdapter
from dataset import MultiModalDataset
from transforms import (
    MultiRandomRotate,
    MultiHorizontalFlip,
    MultiVerticalFlip,
    DepthNormalizer,
    LongTransform,
    FirstChannelTransform,
)
from torchvision import transforms
from output_adapters import DPTOutputAdapter, ConvNeXtAdapter, LinearDepthAdapter
from multimae import multivit_base
from pos_embed_multi import interpolate_pos_embed_multimae
from pathlib import Path
from datetime import datetime
import logging
from configs.depth import depth_configs
from randomness import seed_everything

config = depth_configs()

seed_everything(seed=config.seed)


if __name__ == "__main__":
    # SETUP LOGGING ###

    run_name = config.run_name
    log_dir = config.log_dir
    run_name += f"_{datetime.now():%F_%T}"

    ch = logging.StreamHandler()
    cons_lvl = getattr(logging, config.cons_lvl)
    ch.setLevel(cons_lvl)
    cfmt = logging.Formatter("{levelname:8} - {asctime} - {message}", style="{")
    ch.setFormatter(cfmt)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{run_name}.log"
    fh = logging.FileHandler(log_file)
    file_lvl = getattr(logging, config.file_lvl)
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

    device = config.device

    # SETUP DOMAIN ADAPTERS ###

    DOMAIN_CONF = {
        "rgb": {
            "channels": 3,
            "stride_level": 1,
            "input_adapter": partial(PatchedInputAdapter, num_channels=3),
            "aug_type": "image",
        },
        "depth": {
            "channels": 1,
            "stride_level": 1,
            "input_adapter": partial(PatchedInputAdapter, num_channels=1),
            "aug_type": "mask",
        },
        "mask_valid": {
            "stride_level": 1,
            "aug_type": "mask",
        },
        "semseg": {
            "stride_level": 1,
            "aug_type": "mask",
            "input_adapter": partial(
                SemSegInputAdapter,
                num_classes=config.semseg_num_classes,
                dim_class_emb=32,
                interpolate_class_emb=False,
                emb_padding_idx=config.semseg_num_classes,
            ),
        },
    }

    in_domains = config.in_domains
    out_domains = config.out_domains
    all_domains = list(set(in_domains) | set(out_domains))

    patch_size = config.patch_size
    input_size = config.input_size

    # INPUT ADAPTERS ###

    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=patch_size,
            image_size=input_size,
        )
        for domain in in_domains
    }

    # MAKE OUTPUT ADAPTERS ###

    decoder_main_tasks = config.decoder_main_tasks

    additional_targets = {
        domain: DOMAIN_CONF[domain]["aug_type"] for domain in all_domains
    }

    # DPT settings are fixed for ViT-B. Modify them if using a different backbone.

    adapters_dict = {
        "dpt": DPTOutputAdapter,
        "convnext": partial(ConvNeXtAdapter, preds_per_patch=64),
        "linear_depth": partial(LinearDepthAdapter,
                                input_dim=768,
                                hidden_dims=config.linear_hidden_dims,
                                output_size=input_size, 
                                aggregation=config.linear_aggregation,
                                use_norm=config.linear_use_norm
                                )
    }

    output_adapter = config.output_adapter

    output_adapters = {
        domain: adapters_dict[output_adapter](
            num_classes=DOMAIN_CONF[domain]["channels"],
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size=patch_size,
            main_tasks=decoder_main_tasks,
        )
        for domain in out_domains
    }

    # SET MODEL ###
    model_name = "multivit_base"
    drop_path_encoder = 0.0
    model = multivit_base(
        input_adapters=input_adapters, output_adapters=output_adapters
    )

    # LOAD CHECKPOINT ###
    finetune_path = config.fine_tune_path
    checkpoint = torch.load(finetune_path, map_location="cpu")

    checkpoint_model = checkpoint["model"]

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

    # MAKE TRANSFORMS ###

    train_transforms = {
        "rgb": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.4753, 0.4713, 0.4645],
                    std=[0.0903, 0.0872, 0.0869],
                ),
            ]
        ),
        "semseg": transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.NEAREST
                ),
                FirstChannelTransform(),
                LongTransform(),
            ]
        ),
    }
    target_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((224, 224)), DepthNormalizer()]
    )

    multimodal_transforms = [
        MultiHorizontalFlip(0.5),
        MultiVerticalFlip(0.5),
        MultiRandomRotate(0.5, 90),
    ]

    # MAKE DATASETS ###

    train_dataset = MultiModalDataset(
        root_dir=config.train_dir,
        input_tasks=in_domains,
        output_task=out_domains[0],
        train_transform=train_transforms,
        target_transform=target_transform,
        multimodal_augmentations=multimodal_transforms,
        training=True,
        subset_idx=config.subset_idx
    )
    val_dataset = MultiModalDataset(
        root_dir=config.val_dir,
        input_tasks=in_domains,
        output_task=out_domains[0],
        train_transform=train_transforms,
        target_transform=target_transform,
        training=False,
    )

    bs = config.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, shuffle=False, drop_last=False
    )

    # SET OPTIMIZER ###

    lr = config.lr
    decay = config.weight_decay
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=decay)
    total_iterations = len(train_loader) * config.total_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_iterations, eta_min=1e-7
    )

    # SETUP TRAINER ###

    total_epochs = config.total_epochs
    trainer = RgbDepthTrainer(
        model=model,
        optimizer=opt,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=run_name,
        ckpt_dir=Path(log_dir).parent / "ckpt",
        ckpt_replace=not config.save_every_epoch,
        ckpt_resume=None,
        ckpt_track_metric="rmse",
        metrics_on_train=True,
        total_epochs=total_epochs,
        device=device,
        return_all_layers= (output_adapter == 'dpt')
    )

    trainer.run()
