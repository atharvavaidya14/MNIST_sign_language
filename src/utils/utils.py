import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
import torch
from typing import Tuple


class SignLanguageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_val_datasets(
    csv_path, transform=None, val_split=0.2
) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and validation sets."""
    full_dataset = SignLanguageDataset(csv_path, transform=transform)
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    return random_split(full_dataset, [train_size, val_size])


def evaluate(model, dataloader, criterion, device) -> Tuple[float, float]:
    """
    Evaluate the model on the validation or test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total


def plot_metrics(train_losses, val_losses, val_accuracies):
    """
    Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy over Epochs")

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

