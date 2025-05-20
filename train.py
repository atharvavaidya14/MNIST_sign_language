import torch
from torchvision import transforms
from model import SimpleCNN
from trainer import train_model
from utils import get_train_val_datasets, SignLanguageDataset, evaluate, plot_metrics
from torch.utils.data import DataLoader

# Configs
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Data
train_dataset, val_dataset = get_train_val_datasets("sign_mnist_train.csv", transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataset = SignLanguageDataset("sign_mnist_test.csv", transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model
model = SimpleCNN().to(DEVICE)

# Train
train_losses, val_losses, val_accuracies = train_model(
    model,
    train_loader,
    val_loader,
    device=DEVICE,
    lr=LR,
    epochs=EPOCHS,
    checkpoint_path="trained_models/sign_cnn_best.pth",
    log_dir="runs/sign_cnn",
)

# Plot
plot_metrics(train_losses, val_losses, val_accuracies)

# Evaluate on test
criterion = torch.nn.CrossEntropyLoss()
test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
