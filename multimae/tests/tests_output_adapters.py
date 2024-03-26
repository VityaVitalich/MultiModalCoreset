import torch
import pytest
from ..output_adapters import (
    DPTOutputAdapter,
)


@pytest.fixture
def input_tokens():
    # Mock input tokens from encoder with size (batch_size, num_tokens, dim_tokens)
    # DPT is hierarchical, suppose we have 4 layers
    return [
        torch.randn(2, 196, 768),
        torch.randn(2, 196, 768),
        torch.randn(2, 196, 768),
        torch.randn(2, 196, 768),
    ]


@pytest.fixture
def input_info():
    # Mock input information
    return {
        "image_size": (224, 224),
        "tasks": {
            "rgb": {"start_idx": 0, "end_idx": 768, "has_2d_posemb": True},
        },
        "num_task_tokens": 196,
        "num_global_tokens": 1,
    }


@pytest.fixture
def output_adapter():
    # Init with Mocks
    return DPTOutputAdapter(
        num_classes=1,
        stride_level=1,
        patch_size=16,
        main_tasks=["rgb"],
        hooks=[0, 1, 2, 3],
        layer_dims=[96, 192, 384, 768],
        feature_dim=256,
        use_bn=False,
        dim_tokens_enc=768,
    )


# text for asserions made with chat gpt again, corrected manually after


def test_output_adapter_initialization(output_adapter):
    """
    Test for ensuring that the output adapter is initialized correctly.
    """
    assert (
        output_adapter.num_channels == 1
    ), "Number of channels should be initialized to 1."
    assert output_adapter.stride_level == 1, "Stride level should be initialized to 1."
    assert output_adapter.patch_size == (
        16,
        16,
    ), "Patch size should be correctly set to (16, 16)."
    assert (
        output_adapter.dim_tokens_enc is not None
    ), "Dimension of encoded tokens should not be None."


def test_output_adapter_forward(output_adapter, input_tokens, input_info):
    """
    Test for ensuring that the forward pass of the output adapter works correctly.
    """
    output = output_adapter(input_tokens, input_info)

    print(output.size())
    assert isinstance(output, torch.Tensor), "Output should be a PyTorch tensor."
    assert (
        output.shape[2:] == input_info["image_size"]
    ), "Output image size should match the input information image size."
    assert (
        output.shape[1] == output_adapter.num_channels
    ), "Output channels should match the number of channels in the output adapter."
