import pytest
from unittest.mock import MagicMock
from ..base_trainer import BaseTrainer, _CyclicalLoader
from torch.utils.data import DataLoader
import torch


class Trainer_Mock(BaseTrainer):
    def compute_score(self, *args, **kwargs):
        return {"accuracy": torch.tensor(0.9)}

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0.1, requires_grad=True)


class DatasetMock(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return {"rgb": torch.randn(10, 3, 224, 224)}, torch.randn(10, 224, 224)


@pytest.fixture
def dummy_dataset():
    return DatasetMock()


@pytest.fixture
def dummy_dataloader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=2)


@pytest.fixture
def mock_model():
    # helps mocking methods
    # proposed by ChatGPT when asked: "how could i mock a method with pytest the easiest way"
    model = MagicMock()
    model.return_value = {"depth": torch.randn(2, 3)}
    model.loss = {"loss": torch.tensor(0.1)}
    return model


@pytest.fixture
def mock_optimizer():
    return MagicMock()


@pytest.fixture
def base_trainer(mock_model, mock_optimizer, dummy_dataloader):
    trainer = Trainer_Mock(
        model=mock_model,
        optimizer=mock_optimizer,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        total_epochs=1,
        device="cpu",
    )
    return trainer


def test_trainer_initialization(base_trainer):
    assert base_trainer is not None, "Base trainer should not be none"
    assert base_trainer._total_epochs == 1, "Total epochs does not match with init"


def test_cyclical_loader(dummy_dataloader):
    cyclical_loader = _CyclicalLoader(dummy_dataloader)
    cyclical_loader.set_iters_per_epoch(5)
    assert (
        len(cyclical_loader) == 5
    ), "Cyclical loader length should match set iters_per_epoch"

    data = list(iter(cyclical_loader))
    assert (
        len(data) == 5
    ), "Cyclical loader should iterate over the set number of iterations"


def test_train_loop(base_trainer):
    # Testing a single train iteration
    base_trainer._cyc_train_loader.set_iters_per_epoch(5)
    base_trainer.train(1)
    # Check if last_iter was updated correctly
    assert (
        base_trainer._last_iter == 1
    ), "Last iter should be updated after training step"


def test_validation_loop(base_trainer):
    # Testing a single validation iteration
    base_trainer.validate()
    assert base_trainer._metric_values == {
        "accuracy": torch.tensor(0.9)
    }, "Metric values should be as in Mock"
