import sys 

sys.path.append('../multimae/')

import torch
import numpy as np
import os
import torch.nn as nn
from functools import partial
from input_adapters import PatchedInputAdapter, SemSegInputAdapter
from dataset import MultiModalDataset, LongTransform, FirstChannelTransform
from torchvision import transforms
from output_adapters import DPTOutputAdapter, ConvNeXtAdapter
from multimae import multivit_base
from pos_embed_multi import interpolate_pos_embed_multimae
from pathlib import Path
from datetime import datetime
import logging
from configs.depth import depth_configs

import matplotlib.pyplot as plt
from PIL import Image


def init_rgb_model():

    device = 'cuda:0'

    ### SETUP DOMAIN ADAPTERS ###

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
    }

    in_domains = ['rgb']
    out_domains = ['depth']
    all_domains = list(set(in_domains) | set(out_domains))

    patch_size = 16
    input_size = 224

    ### INPUT ADAPTERS ###

    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=patch_size,
            image_size=input_size,
        )
        for domain in in_domains
    }

    ### MAKE OUTPUT ADAPTERS ###

    decoder_main_tasks = ['rgb']

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

    ### SET MODEL ###
    model_name = "multivit_base"
    drop_path_encoder = 0.0
    model = multivit_base(
        input_adapters=input_adapters, output_adapters=output_adapters
    )

    ckpt_path = '/home/MultiModalCoreset/multimae/ckpt/multimae-test_2023-12-08_09:34:44/epoch__0003_-_loss__315.9.ckpt'

    ### LOAD CHECKPOINT ###
    finetune_path = ckpt_path
    checkpoint = torch.load(finetune_path, map_location="cpu")

    checkpoint_model = checkpoint["model"]

    # Interpolate position embedding
    interpolate_pos_embed_multimae(model, checkpoint_model)

    # Load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model.eval()

    return model

def prepare_image(img):

    train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        )
    
    x = train_transforms(img)
    sample_dict = {'rgb': x.unsqueeze(0)}

    return sample_dict

def inference(img, model):

    sample_dict = prepare_image(img)
    
    with torch.no_grad():
        out = model(sample_dict, return_all_layers=True)

    pred = out['depth'][0][0].numpy()

    return pred
    
def save_predictions(pred, path):
    plt.imshow(pred)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)