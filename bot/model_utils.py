import sys
import torch
from functools import partial
from torchvision import transforms
import matplotlib.pyplot as plt

sys.path.append("../multimae/")
from input_adapters import PatchedInputAdapter  # noqa:E402
from dataset import LongTransform, FirstChannelTransform  # noqa:E402
from output_adapters import DPTOutputAdapter, ConvNeXtAdapter  # noqa:E402
from multimae import multivit_base  # noqa:E402
from pos_embed_multi import interpolate_pos_embed_multimae  # noqa:E402


def init_rgb_model():
    # SETUP DOMAIN ADAPTERS #

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

    in_domains = ["rgb"]
    out_domains = ["depth"]

    patch_size = 16
    input_size = 224

    # INPUT ADAPTERS #

    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=patch_size,
            image_size=input_size,
        )
        for domain in in_domains
    }

    # MAKE OUTPUT ADAPTERS #

    decoder_main_tasks = ["rgb"]

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

    # SET MODEL #
    model = multivit_base(
        input_adapters=input_adapters, output_adapters=output_adapters
    )

    ckpt_path = "./model_ckpt/epoch__0003_-_loss__315.9.ckpt"

    # LOAD CHECKPOINT #
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
    sample_dict = {"rgb": x.unsqueeze(0)}

    return sample_dict


def inference(img, model):
    sample_dict = prepare_image(img)

    with torch.no_grad():
        out = model(sample_dict, return_all_layers=True)

    pred = out["depth"][0][0].numpy()

    return pred


def save_predictions(pred, path):
    plt.imshow(pred)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)


def init_rgb_semseg_model():
    # SETUP DOMAIN ADAPTERS #

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
        # "semseg": {
        #     "stride_level": 1,
        #     "aug_type": "mask",
        #     "input_adapter": partial(
        #         SemSegInputAdapter,
        #         num_classes=config.semseg_num_classes,
        #         dim_class_emb=32,
        #         interpolate_class_emb=False,
        #         emb_padding_idx=config.semseg_num_classes,
        #     ),
        # },
        # not working
    }

    in_domains = ["rgb", "semseg"]
    out_domains = ["depth"]

    patch_size = 16
    input_size = 224

    # INPUT ADAPTERS #

    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=patch_size,
            image_size=input_size,
        )
        for domain in in_domains
    }

    # MAKE OUTPUT ADAPTERS #

    decoder_main_tasks = ["rgb", "semseg"]

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

    # SET MODEL #

    model = multivit_base(
        input_adapters=input_adapters, output_adapters=output_adapters
    )

    ckpt_path = "/home/MultiModalCoreset/multimae/ckpt/long-rgb-semseg_2023-12-23_21:46:00/epoch__0005_-_loss__281.8.ckpt"  # noqa: 501

    # LOAD CHECKPOINT #
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


def prepare_multi_image(img, semseg):
    train_transforms = {
        "rgb": transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        ),
        "semseg": transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize((224, 224)),
                FirstChannelTransform(),
                LongTransform(),
            ]
        ),
    }

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
