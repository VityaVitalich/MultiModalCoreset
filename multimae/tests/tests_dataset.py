import pytest
from ..dataset import MultiModalDataset
from torchvision.transforms import Compose, ToTensor
import numpy as np
from PIL import Image
import os
import torch

tmp_path = "tmp"


@pytest.fixture
def random_image(tmp_path):
    """
    Create temporary image. unlink at the end
    """
    img_path = tmp_path / "image.png"
    array = np.random.rand(256, 256, 3) * 255
    img = Image.fromarray(array.astype("uint8")).convert("RGB")
    img.save(img_path)
    yield img_path
    img_path.unlink()


@pytest.fixture
def sample_transform():
    return Compose([ToTensor()])


@pytest.fixture
def multimodal_dataset(tmp_path, random_image, sample_transform):
    root_dir = tmp_path
    os.mkdir(root_dir / "rgb")
    os.mkdir(root_dir / "depth")
    os.link(random_image, root_dir / "rgb" / "image.png")
    os.link(random_image, root_dir / "depth" / "image.png")
    return MultiModalDataset(
        root_dir=str(root_dir),  # Use the temporary directory
        train_transform={"rgb": sample_transform},
        target_transform=sample_transform,
        input_tasks=["rgb"],
        output_task="depth",
        training=True,
    )


class MockAugmentation:
    def set_param(self):
        pass

    def __call__(self, img, modality=None):
        return img.convert("L")


@pytest.fixture
def augmented_multimodal_dataset(multimodal_dataset):
    multimodal_dataset.multimodal_augmentations = [MockAugmentation()]
    return multimodal_dataset


# same thing, chat gpt helped in writting text for assertions, cuz im too lazy for those things :)
# the prompt was "suppose i have the following pytest tests, write a text for assertions,
# output the code with text of assertions"


def test_dataset_init(multimodal_dataset):
    """
    Testing initialization of dataset
    """
    assert (
        multimodal_dataset.root_dir is not None
    ), "The root_dir should be initialized and not None"
    assert (
        "rgb" in multimodal_dataset.input_tasks
    ), "The dataset should contain 'rgb' in its input_tasks"
    assert (
        multimodal_dataset.output_task == "depth"
    ), "The output task should be set to 'depth'"
    assert hasattr(
        multimodal_dataset, "rgb_files"
    ), "The dataset should have an attribute 'rgb_files'"


def test_dataset_len(multimodal_dataset):
    """
    Just a simple check for Len function correctness
    """
    assert len(multimodal_dataset) == 1, "The length of the dataset should be exactly 1"


def test_dataset_getitem(multimodal_dataset):
    """
    Check for correct formats of files
    """
    item, target = multimodal_dataset[0]
    assert isinstance(item, dict), "The item should be of type dict"
    assert "rgb" in item, "The item should contain 'rgb' key"
    assert isinstance(
        item["rgb"], torch.Tensor
    ), "The 'rgb' item should be a torch.Tensor"
    assert isinstance(target, torch.Tensor), "The target should be a torch.Tensor"


def test_augmentations(augmented_multimodal_dataset):
    """
    Simple checks for augmentations
    """
    sample_dict, y = {"rgb": Image.new("RGB", (100, 100))}, Image.new("L", (100, 100))
    (
        augmented_sample_dict,
        augmented_y,
    ) = augmented_multimodal_dataset.make_augmentations(sample_dict, y)

    assert isinstance(
        augmented_sample_dict, dict
    ), "The augmented sample should be of type dict"
    assert (
        "rgb" in augmented_sample_dict
    ), "The augmented sample should contain 'rgb' key"
    assert (
        augmented_sample_dict["rgb"].mode == "L"
    ), "The 'rgb' item in augmented sample should be in 'L' mode (grayscale)"
    assert (
        augmented_y.mode == "L"
    ), "The augmented target image should be in 'L' mode (grayscale)"
