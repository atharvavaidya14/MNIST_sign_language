import argparse
import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.model_architecture import SimpleCNN
from src.training.trainer import train_model
from src.utils.utils import (
    get_train_val_datasets,
    SignLanguageDataset,
    evaluate,
    plot_metrics,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def main(args):
    # Dataset version logging
    dataset_version = os.path.basename(args.train_csv).split("_")[-1].split(".")[0]
    print(f"📦 Using dataset version: {dataset_version}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"🚀 Using device: {device}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # WandB setup
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="sign-language-cnn",
            config={
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "dataset": dataset_version,
            },
        )

    # Transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Datasets
    train_dataset, val_dataset = get_train_val_datasets(args.train_csv, transform)
    test_dataset = SignLanguageDataset(args.test_csv, transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    model = SimpleCNN().to(device)

    # Train
    train_losses, val_losses, val_accuracies = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
        writer=writer,
        use_wandb=args.use_wandb,
    )

    # Plot metrics
    plot_metrics(train_losses, val_losses, val_accuracies)

    # Evaluate on test set
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"🧪 Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
        wandb.finish()

    # Tracking for dvc
    metrics = {
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "val_accuracy": val_accuracies[-1],
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on Sign Language MNIST")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/sign_mnist_train_v1.csv",
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/sign_mnist_test_v1.csv",
        help="Path to testing CSV file",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="trained_models/sign_cnn_best.pth",
        help="Path to save best model",
    )
    parser.add_argument(
        "--log_dir", type=str, default="runs/sign_cnn", help="TensorBoard log directory"
    )
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable logging to Weights & Biases (wandb)",
    )

    args = parser.parse_args()
    main(args)
