import os
import pytest
import pandas as pd
import numpy as np
import torch
from PIL import Image
from unittest.mock import patch
from src.models.model_architecture import SimpleCNN
from src.utils.utils import (
    SignLanguageDataset,
    get_train_val_datasets,
    evaluate,
    plot_metrics,
    get_device,
    load_model,
)


@pytest.fixture(scope="module")
def create_csv(tmp_path_factory):
    # Create a temporary directory and CSV file
    tmp_dir = tmp_path_factory.mktemp("data")
    test_csv_path = tmp_dir / "test_data.csv"

    # Create dummy dataset: 3 samples, 28x28 images (784 pixels) + label
    header = ["label"] + [f"pixel{i}" for i in range(1, 785)]
    data = [
        [0] + [0] * 784,
        [1] + [255] * 784,
        [2] + [128] * 784,
    ]
    df = pd.DataFrame(data, columns=header)
    df.to_csv(test_csv_path, index=False, header=True)

    yield str(test_csv_path)  # Return as string path for compatibility


def test_sign_language_dataset_len_and_getitem(create_csv):
    dataset = SignLanguageDataset(create_csv)

    # Length should be 3 rows
    assert len(dataset) == 3

    # Test __getitem__ returns PIL Image and correct label
    img, label = dataset[1]
    assert isinstance(img, Image.Image)
    assert label == 1


def test_sign_language_dataset_with_transform(create_csv):
    # Dummy transform that converts image to grayscale numpy array
    def dummy_transform(img):
        return np.array(img)

    dataset = SignLanguageDataset(create_csv, transform=dummy_transform)
    img, _ = dataset[0]
    assert isinstance(img, np.ndarray)
    assert img.shape == (28, 28)


def test_get_train_val_datasets(create_csv):
    train_ds, val_ds = get_train_val_datasets(create_csv, val_split=0.33)
    total_len = len(train_ds) + len(val_ds)
    assert total_len == 3
    # Validate sizes approximately
    assert abs(len(val_ds) - 1) <= 1
    assert abs(len(train_ds) - 2) <= 1


def test_evaluate():
    # Simple dummy model returning fixed outputs
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Return batch_size x 3 logits tensor
            return torch.tensor([[0.1, 0.7, 0.2]] * x.size(0))

    # Dummy criterion: always return fixed loss
    def dummy_criterion(outputs, labels):
        return torch.tensor(0.5)

    # Dummy data loader
    dummy_data = [(torch.randn(2, 1, 28, 28), torch.tensor([1, 2]))]
    device = torch.device("cpu")

    model = DummyModel()
    loss, acc = evaluate(model, dummy_data, dummy_criterion, device)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 100


def test_plot_metrics(tmp_path):
    train_losses = [0.5, 0.4, 0.3]
    val_losses = [0.6, 0.5, 0.4]
    val_acc = [70, 75, 80]

    # Change working directory to temp
    cwd = os.getcwd()
    os.chdir(tmp_path)

    plot_metrics(train_losses, val_losses, val_acc)

    assert (tmp_path / "training_plot.png").exists()

    os.chdir(cwd)


def test_get_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda"]


@pytest.fixture
def dummy_model_file(tmp_path):
    model = SimpleCNN()
    dummy_path = tmp_path / "dummy_model.pth"
    torch.save(model.state_dict(), dummy_path)
    return dummy_path


def test_load_model_eval_mode(dummy_model_file):
    model = load_model(str(dummy_model_file))  # Default is eval_mode=True
    assert isinstance(model, SimpleCNN)
    assert not model.training


def test_load_model_train_mode(dummy_model_file):
    model = load_model(str(dummy_model_file), eval_mode=False)
    assert isinstance(model, SimpleCNN)
    assert model.training


@patch("torch.cuda.is_available", return_value=True)
def test_get_device_cuda(mock_cuda):
    assert str(get_device()) == "cuda"
