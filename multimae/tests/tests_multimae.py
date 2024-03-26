import pytest
import torch
from ..multimae import MultiMAE, MultiViT, pretrain_multimae_base, multivit_base


# Mock for an adapters. should return something that looks like visual tokens
class AdapterMock(torch.nn.Module):
    def init(*args, **kwargs):
        return

    def forward(x, *args, **kwargs):
        return torch.rand(2, 197, 768)


@pytest.fixture
def sample_input():
    # random image-like array
    return {"rgb": torch.randn(2, 3, 224, 224), "semseg": torch.randn(2, 224, 224)}


@pytest.fixture
def input_adapters():
    # Mocks
    return {"rgb": AdapterMock(), "semseg": AdapterMock()}


@pytest.fixture
def output_adapters():
    # Mocks
    return {"rgb": AdapterMock(), "semseg": AdapterMock()}


@pytest.fixture
def multimae_model(input_adapters, output_adapters):
    # Initialize a MultiMAE default with Mock adapters
    return MultiMAE(input_adapters=input_adapters, output_adapters=output_adapters)


@pytest.fixture
def multivit_model(input_adapters, output_adapters):
    # Initialize a MultiViT model with mock adatapters
    return multivit_base(input_adapters=input_adapters, output_adapters=output_adapters)


# MULTIMAE PART
def test_multimae_initialization(multimae_model):
    # mostly checks the correctness of init
    assert isinstance(
        multimae_model, MultiMAE
    ), "The model should be an instance of MultiMAE"


def test_multimae_forward(multimae_model, sample_input):
    # mostly checks correct forward
    preds, _ = multimae_model(sample_input, mask_inputs=True)
    assert isinstance(preds, dict), "Model should output a dict"
    assert (
        "rgb" in preds and "semseg" in preds
    ), "Output dictionary should contain keys for all input tasks"


def test_multimae_masking(multimae_model, sample_input):
    _, task_masks = multimae_model(sample_input, mask_inputs=True)
    assert isinstance(task_masks, dict), "Task masks should be a dict"
    assert all(
        torch.is_tensor(mask) for mask in task_masks.values()
    ), "Each task mask should be a tensor"


# MultiViT class
def test_multivit_initialization(multivit_model):
    assert isinstance(
        multivit_model, MultiViT
    ), "The model should be an instance of MultiViT."


def test_multivit_forward(multivit_model, sample_input):
    preds = multivit_model(sample_input)
    assert isinstance(preds, dict), "Model should output a dictionary of predictions."
    assert (
        "rgb" in preds and "semseg" in preds
    ), "Output dictionary should contain keys for all input tasks."


# Additional tests for predefined models
def test_pretrain_multimae_base_initialization(input_adapters, output_adapters):
    model = pretrain_multimae_base(
        input_adapters=input_adapters, output_adapters=output_adapters
    )
    assert isinstance(
        model, MultiMAE
    ), "pretrain_multimae_base should return a MultiMAE instance."


def test_multivit_base_initialization(input_adapters, output_adapters):
    model = multivit_base(
        input_adapters=input_adapters, output_adapters=output_adapters
    )
    assert isinstance(
        model, MultiViT
    ), "multivit_base should return a MultiViT instance."
