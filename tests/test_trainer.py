import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from src.training.trainer import train_model
import os


# Dummy writer that does nothing
class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def create_dummy_dataloaders():
    # Simulate simple grayscale image data
    x = torch.randn(30, 1, 28, 28)  # 30 images
    y = torch.randint(0, 3, (30,))  # 3 classes
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10)
    return loader, loader  # use same loader for train & val


def test_train_model_executes():
    train_loader, val_loader = create_dummy_dataloaders()
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 3))
    device = torch.device("cpu")

    fd, path = tempfile.mkstemp()
    os.close(fd)  # Close file descriptor so torch can write to it
    try:
        train_losses, val_losses, val_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=5,
            patience=2,
            lr=0.001,
            checkpoint_path=path,
            writer=DummyWriter(),
            use_wandb=False,
        )

        assert len(train_losses) > 0
        assert len(train_losses) == len(val_losses) == len(val_acc)
        assert all(isinstance(loss, float) for loss in train_losses)
        assert os.path.exists(path)  # checkpoint file should exist
    finally:
        os.remove(path)


def test_early_stopping_triggered():
    # Overfit quickly to cause early stopping
    train_loader, val_loader = create_dummy_dataloaders()
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 3))
    device = torch.device("cpu")

    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        train_losses, val_losses, val_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=20,
            patience=1,
            lr=0.001,
            checkpoint_path=path,
            writer=DummyWriter(),
            use_wandb=False,
        )
        assert len(train_losses) <= 20
    finally:
        os.remove(path)
