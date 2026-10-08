from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from app.cnn_model import AssignmentCNN, CIFAR10_CLASSES, CNN_INPUT_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Assignment 2 CIFAR-10 CNN.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--model-path", type=Path, default=Path("./models/cifar10_cnn.pth"))
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Use only the first N training examples for a quick smoke test.",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=0,
        help="Use only the first N test examples for a quick smoke test.",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(CNN_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    if args.limit_train:
        train_dataset = Subset(train_dataset, range(args.limit_train))
    if args.limit_test:
        test_dataset = Subset(test_dataset, range(args.limit_test))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


def evaluate(model: AssignmentCNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total else 0.0


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = build_loaders(args)
    model = AssignmentCNN(num_classes=len(CIFAR10_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=120)
        for batch_number, (inputs, labels) in enumerate(progress, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

            progress.set_postfix(
                {
                    "loss": f"{running_loss / batch_number:.4f}",
                    "accuracy": f"{running_correct / max(1, running_total):.3f}",
                }
            )

        test_accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}: test accuracy = {test_accuracy:.2f}%")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CIFAR10_CLASSES,
            "input_size": CNN_INPUT_SIZE,
            "epochs": args.epochs,
            "test_accuracy": evaluate(model, test_loader, device),
        },
        args.model_path,
    )
    print(f"Saved model checkpoint to {args.model_path}")


if __name__ == "__main__":
    main()
