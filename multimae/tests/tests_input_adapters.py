import torch
import pytest
from ..input_adapters import PatchedInputAdapter, SemSegInputAdapter


@pytest.fixture
def patch_input_adapter():
    """
    Simulation of input adapter as it is during training. imgs are also reshaped to 224
    """
    adapter = PatchedInputAdapter(
        num_channels=3,
        stride_level=1,
        patch_size_full=16,
        dim_tokens=768,
        sincos_pos_emb=True,
        learnable_pos_emb=False,
        image_size=224,
    )
    return adapter


@pytest.fixture
def semseg_input_adapter():
    """
    Simulation of input adapter as it is during training. imgs are also reshaped to 224
    """
    adapter = SemSegInputAdapter(
        num_classes=10,
        stride_level=1,
        patch_size_full=16,
        dim_tokens=768,
        sincos_pos_emb=True,
        learnable_pos_emb=False,
        image_size=224,
        dim_class_emb=64,
        interpolate_class_emb=True,
        emb_padding_idx=0,
    )
    return adapter


# chat gpt helped me to write text of assertions (string after assertion), i checked that it is correct
# just a bit lazy for writing obvious things
# the prompt was "suppose i have the following pytest tests,
# write a text for assertions, output the code with text of assertions"


def test_patched_input_adapter_init(patch_input_adapter):
    """
    Simple init tests for Patched Input Adapter.
    """
    assert (
        patch_input_adapter.num_channels == 3
    ), "Number of input channels should be 3."
    assert (
        patch_input_adapter.stride_level == 1
    ), "Stride level should be initialized to 1."
    assert patch_input_adapter.patch_size_full == (
        16,
        16,
    ), "Patch size should be correctly set to (16, 16)."
    assert (
        patch_input_adapter.dim_tokens == 768
    ), "Dimension of tokens should be set to 768."


def test_semseg_input_adapter_init(semseg_input_adapter):
    """
    Simple init tests for Semantic Segmentation Input Adapter.
    """
    assert semseg_input_adapter.num_classes == 11, "Number of classes should be 11."
    assert (
        semseg_input_adapter.stride_level == 1
    ), "Stride level should be initialized to 1."
    assert semseg_input_adapter.patch_size_full == (
        16,
        16,
    ), "Patch size should be correctly set to (16, 16)."
    assert (
        semseg_input_adapter.dim_tokens == 768
    ), "Dimension of tokens should be set to 768."


def test_patched_input_adapter_forward(patch_input_adapter):
    """
    Forward simulation for Patched Input Adapter
    """
    input_tensor = torch.rand((1, 3, 224, 224))
    output = patch_input_adapter(input_tensor)
    assert output.shape == (
        1,
        196,
        768,
    ), "Output shape should be (1, 196, 768) indicating correct batch size, number of patches, and dimension of tokens."


def test_semseg_input_adapter_forward(semseg_input_adapter):
    """
    Forward simulation for Semantic Segmentation Input Adapter.
    """
    input_tensor = torch.randint(0, 10, (1, 224, 224))
    output = semseg_input_adapter(input_tensor)
    assert output.shape == (
        1,
        196,
        768,
    ), "Output shape should be (1, 196, 768) indicating correct batch size, number of patches, and dimension of tokens."
