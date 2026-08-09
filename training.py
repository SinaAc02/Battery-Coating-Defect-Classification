from pathlib import Path
import json
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


# ------------------------------------------------------------
# Fixed config
# ------------------------------------------------------------
random_seed = 42
batch_size = 32
learning_rate = 0.0003
n_epochs = 12
patience = 3
image_size = 224

labels = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data" / "raw" / "CoatingVision" / "classification"
output_dir = base_dir / "model_output"
output_dir.mkdir(exist_ok=True)

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class CoatingDataset(Dataset):
    def __init__(self, data, image_folder, transform):
        self.data = data.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = Image.open(self.image_folder / row["file_name"]).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(row[labels].values.astype(np.float32))
        return image, target, row["file_name"]


def get_frame_name(file_name):
    # Several patches can come from the same camera frame.
    return re.sub(r"_patch_\d+\.[^.]+$", "", file_name)


def make_split(data):
    # Grouped splitting prevents patches from one frame appearing in two splits.
    first_split = GroupShuffleSplit(
        n_splits=1, train_size=0.70, random_state=random_seed
    )
    train_index, remaining_index = next(
        first_split.split(data, groups=data["frame_name"])
    )

    train_data = data.iloc[train_index]
    remaining_data = data.iloc[remaining_index]

    second_split = GroupShuffleSplit(
        n_splits=1, train_size=0.50, random_state=random_seed
    )
    validation_index, test_index = next(
        second_split.split(remaining_data, groups=remaining_data["frame_name"])
    )

    validation_data = remaining_data.iloc[validation_index]
    test_data = remaining_data.iloc[test_index]
    return train_data, validation_data, test_data


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probabilities = []
    all_file_names = []

    with torch.no_grad():
        for images, targets, file_names in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            loss = criterion(output, targets)

            total_loss += loss.item() * images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_probabilities.append(torch.sigmoid(output).cpu().numpy())
            all_file_names.extend(file_names)

    return (
        total_loss / len(data_loader.dataset),
        np.concatenate(all_targets),
        np.concatenate(all_probabilities),
        all_file_names,
    )


def find_thresholds(targets, probabilities):
    # Find one threshold per label using only validation data.
    thresholds = []
    for column in range(len(labels)):
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(0.10, 0.91, 0.05):
            predictions = probabilities[:, column] >= threshold
            _, _, f1, _ = precision_recall_fscore_support(
                targets[:, column], predictions, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds.append(best_threshold)

    return np.array(thresholds)


def main():
    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    labels_file = data_dir / "labels.csv"
    image_folder = data_dir / "images"

    if not labels_file.exists():
        raise FileNotFoundError("Dataset not found. Run download_data.py first.")

    data = pd.read_csv(labels_file)
    data["frame_name"] = data["original_file_name"].apply(get_frame_name)
    train_data, validation_data, test_data = make_split(data)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
    print(
        f"Train/validation/test images: "
        f"{len(train_data)}/{len(validation_data)}/{len(test_data)}"
    )

    split_file = pd.concat([
        train_data.assign(split="train"),
        validation_data.assign(split="validation"),
        test_data.assign(split="test"),
    ])
    split_file.to_csv(output_dir / "data_split.csv", index=False)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_loader = DataLoader(
        CoatingDataset(train_data, image_folder, train_transform),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        CoatingDataset(validation_data, image_folder, test_transform),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        CoatingDataset(test_data, image_folder, test_transform),
        batch_size=batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model = model.to(device)

    # Rare labels receive a larger weight in the loss.
    positive_count = train_data[labels].sum().values.astype(np.float32)
    negative_count = len(train_data) - positive_count
    positive_weights = torch.tensor(
        negative_count / positive_count, dtype=torch.float32, device=device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0001
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    train_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        validation_loss, _, _, _ = evaluate(
            model, validation_loader, criterion
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print(
            f"Epoch {epoch:02d}: train loss={train_loss:.4f}, "
            f"validation loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_state)

    # ------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------
    _, validation_targets, validation_probabilities, _ = evaluate(
        model, validation_loader, criterion
    )
    thresholds = find_thresholds(validation_targets, validation_probabilities)

    test_loss, test_targets, test_probabilities, test_file_names = evaluate(
        model, test_loader, criterion
    )
    test_predictions = test_probabilities >= thresholds

    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, test_predictions, average=None, zero_division=0
    )
    micro_f1 = precision_recall_fscore_support(
        test_targets, test_predictions, average="micro", zero_division=0
    )[2]
    macro_f1 = f1.mean()
    macro_average_precision = average_precision_score(
        test_targets, test_probabilities, average="macro"
    )

    print()
    print("Final test result")
    print("-" * 65)
    print(f"Test loss:               {test_loss:.4f}")
    print(f"Micro F1:                {micro_f1:.4f}")
    print(f"Macro F1:                {macro_f1:.4f}")
    print(f"Macro average precision: {macro_average_precision:.4f}")
    print()

    result_rows = []
    for index, label in enumerate(labels):
        average_precision = average_precision_score(
            test_targets[:, index], test_probabilities[:, index]
        )
        result_rows.append({
            "label": label,
            "precision": precision[index],
            "recall": recall[index],
            "f1": f1[index],
            "average_precision": average_precision,
            "support": int(support[index]),
            "threshold": thresholds[index],
        })
        print(
            f"{label:16s} precision={precision[index]:.3f} "
            f"recall={recall[index]:.3f} f1={f1[index]:.3f} "
            f"support={int(support[index])}"
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "test_results.csv", index=False)

    predictions = pd.DataFrame({"file_name": test_file_names})
    for index, label in enumerate(labels):
        predictions[f"true_{label}"] = test_targets[:, index].astype(int)
        predictions[f"probability_{label}"] = test_probabilities[:, index]
        predictions[f"predicted_{label}"] = test_predictions[:, index].astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    torch.save({
        "model_state": model.state_dict(),
        "labels": labels,
        "thresholds": thresholds,
        "image_size": image_size,
    }, output_dir / "coating_defect_model.pth")

    summary = {
        "test_loss": test_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "macro_average_precision": macro_average_precision,
        "train_images": len(train_data),
        "validation_images": len(validation_data),
        "test_images": len(test_data),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, marker="o", label="Train loss")
    plt.plot(validation_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=160)
    plt.close()

    results.set_index("label")[["precision", "recall", "f1"]].plot.bar(
        figsize=(8, 4)
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Test results")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "test_results.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()

