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
from output_adapters import DPTOutputAdapter, ConvNeXtAdapter
from multimae import multivit_base
from pos_embed_multi import interpolate_pos_embed_multimae
from pathlib import Path
from datetime import datetime
import logging
from configs.depth import embedding_configs
from randomness import seed_everything
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def obtain_embeddings(model, loader, device, aggregation='none'):
    all_embeddings = []

    for i, (inp, gt) in tqdm(enumerate(loader), total=len(loader)):
        inp, gt = inp, gt.to(device)

        task_dict = {k: v.to(device) for k, v in inp.items()}

        # pred = model(
        #     task_dict, return_all_layers=True
        # ) 
        input_tokens, input_info = model.process_input(task_dict)
        encoder_tokens = model.encoder(input_tokens).cpu()
        if aggregation == 'none':
            encoder_tokens = encoder_tokens.reshape(encoder_tokens.size(0), -1).numpy()
        elif aggregation == 'sum':
            encoder_tokens = encoder_tokens.sum(dim=1).numpy()
        elif aggregation == 'mean':
            encoder_tokens = encoder_tokens.mean(dim=1).numpy()
        else:
            raise AttributeError(f'Unknown aggregation {aggregation}')
        all_embeddings.append(encoder_tokens)

    return np.vstack(all_embeddings)

@torch.no_grad()
def obtain_head_embeddings(model, loader, device, saving_path):

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)


    for i, (inp, gt) in tqdm(enumerate(loader), total=len(loader)):
        inp, gt = inp, gt.to(device)

        task_dict = {k: v.to(device) for k, v in inp.items()}

        out = model(task_dict, return_all_layers=True, return_embedding=True)
        out = out['depth'][0].cpu()
        
        saving_name = f'{saving_path}/batch_{i}.pt'
        torch.save(out, saving_name)


config = embedding_configs()

seed_everything(seed=config.seed)


if __name__ == "__main__":
    device = config.device
    embedding_type = config.embedding_type

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
    }

    output_adapter = "dpt"

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
    ).to(device)

    # LOAD CHECKPOINT ###
    finetune_path = config.fine_tune_path
    checkpoint = torch.load(finetune_path, map_location="cpu")

    checkpoint_model = checkpoint["model"]

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
    # MAKE DATASETS ###

    train_dataset = MultiModalDataset(
        root_dir=config.train_dir,
        input_tasks=in_domains,
        output_task=out_domains[0],
        train_transform=train_transforms,
        target_transform=target_transform,
        training=False,
    )

    bs = config.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=False, drop_last=False
    )

    if embedding_type == 'transformer_out':
        embeds = obtain_embeddings(model, train_loader, device, aggregation=config.aggregation)
        np.save(config.embed_save_path, embeds)

    elif embedding_type == 'head_out':
        obtain_head_embeddings(model, train_loader, device, saving_path=config.saving_path)



