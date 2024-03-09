import sys 

sys.path.append('../multimae/')
from rgb_d_trainer import RgbDepthTrainer
import torch
import numpy as np
import os
from functools import partial
from input_adapters import PatchedInputAdapter, SemSegInputAdapter
from transforms import MultiRandomRotate, MultiHorizontalFlip, MultiVerticalFlip, DepthNormalizer, LongTransform, FirstChannelTransform
from torchvision import transforms
from output_adapters import DPTOutputAdapter, ConvNeXtAdapter
from multimae import multivit_base
from pos_embed_multi import interpolate_pos_embed_multimae

multi_path = "../multimae/ckpt/epoch__0015_-_rmse__0.005769.ckpt"
rgb_path = "../multimae/ckpt/rgb-augmented-epoch25_rmse_534.ckpt"
semseg_path = "../multimae/ckpt/semseg-epoch__0026_-_rmse__491.7.ckpt"

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

device = 'cuda:0'
semseg_num_classes = 256

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
            num_classes=semseg_num_classes,
            dim_class_emb=32,
            interpolate_class_emb=False,
            emb_padding_idx=semseg_num_classes,
        ),
    },
}

out_domains = ['depth']

patch_size = 16
input_size = 224

def init_rgb_model():


    ### SETUP DOMAIN ADAPTERS ###

    in_domains = ['rgb']
    all_domains = list(set(in_domains) | set(out_domains))

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


    ### LOAD CHECKPOINT ###
    finetune_path = rgb_path
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

    x = train_transforms['rgb'](img)
    sample_dict = {'rgb': x.unsqueeze(0)}

    return sample_dict

def inference(img, model):

    sample_dict = prepare_image(img)
    
    with torch.no_grad():
        out = model(sample_dict, return_all_layers=True)

    pred = out['depth'][0][0].numpy()

    return pred
    


def init_rgb_semseg_model():

    ### SETUP DOMAIN ADAPTERS ###

    in_domains = ["rgb", "semseg"]
    out_domains = ["depth"]
    all_domains = list(set(in_domains) | set(out_domains))


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

    decoder_main_tasks = ["rgb", "semseg"]

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


    ### LOAD CHECKPOINT ###
    finetune_path = multi_path
    checkpoint = torch.load(finetune_path, map_location="cpu")

    checkpoint_model = checkpoint["model"]

    # Interpolate position embedding
    interpolate_pos_embed_multimae(model, checkpoint_model)

    # Load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model.eval()

    return model


def prepare_multi_image(img, semseg):

    x = train_transforms["rgb"](img)
    semseg = train_transforms["semseg"](semseg)
    sample_dict = {"rgb": x.unsqueeze(0), "semseg": semseg.unsqueeze(0)}

    return sample_dict


def multi_inference(img, semseg, model):
    sample_dict = prepare_multi_image(img, semseg)

    with torch.no_grad():
        out = model(sample_dict, return_all_layers=True)

    pred = out["depth"][0][0].numpy()

    return pred



def init_semseg_model():

    ### SETUP DOMAIN ADAPTERS ###
    in_domains = ["semseg"]
    out_domains = ["depth"]
    all_domains = list(set(in_domains) | set(out_domains))

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

    decoder_main_tasks = ["semseg"]

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


    ### LOAD CHECKPOINT ###
    finetune_path = semseg_path
    checkpoint = torch.load(finetune_path, map_location="cpu")

    checkpoint_model = checkpoint["model"]

    # Interpolate position embedding
    interpolate_pos_embed_multimae(model, checkpoint_model)

    # Load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model.eval()

    return model


def prepare_semseg_image(semseg):

    semseg = train_transforms["semseg"](semseg)
    sample_dict = {"semseg": semseg.unsqueeze(0)}

    return sample_dict


def semseg_inference(semseg, model):
    sample_dict = prepare_semseg_image(semseg)

    with torch.no_grad():
        out = model(sample_dict, return_all_layers=True)

    pred = out["depth"][0][0].numpy()

    return pred

def masked_berhu_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)

    diff = preds - target
    diff[~mask_valid] = 0
    with torch.no_grad():
        c = max(torch.abs(diff).max() * 0.2, 1e-5)

    l1_loss = torch.abs(diff)
    l2_loss = (torch.square(diff) + c**2) / 2.0 / c
    berhu_loss = (
        l1_loss[torch.abs(diff) < c].sum() + l2_loss[torch.abs(diff) >= c].sum()
    )

    return berhu_loss / mask_valid.sum()