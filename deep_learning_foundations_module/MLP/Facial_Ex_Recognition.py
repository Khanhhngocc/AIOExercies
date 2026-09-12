from pathlib import Path
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 59

IMG_HEIGHT = 128
IMG_WIDTH = 128

TRAIN_RATIO = 0.8
BATCH_SIZE = 64
HIDDEN_DIMS = 128
LEARNING_RATE = 1e-3
EPOCHS = 20

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")


# =========================================================
# DATA PATHS
# =========================================================

current_dir = Path(__file__).resolve().parent /"datasets" / "[Data]_Ex3_FER-2013"

train_dir = current_dir / "train"
test_dir = current_dir / "test"

if not train_dir.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục train:\n{train_dir}"
    )

if not test_dir.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục test:\n{test_dir}"
    )


# =========================================================
# CLASS LABELS
# =========================================================

classes = sorted([
    folder.name
    for folder in train_dir.iterdir()
    if folder.is_dir()
])

if not classes:
    raise ValueError(
        f"Không tìm thấy thư mục class nào trong:\n{train_dir}"
    )

label2idx = {
    class_name: index
    for index, class_name in enumerate(classes)
}

idx2label = {
    index: class_name
    for class_name, index in label2idx.items()
}

print("Classes:", classes)
print("label2idx:", label2idx)


# =========================================================
# READ IMAGE PATHS
# =========================================================

def collect_image_files(
    image_directory,
    class_mapping
):
    image_paths = []
    image_labels = []

    for class_name in class_mapping:
        class_directory = image_directory / class_name

        if not class_directory.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục class:\n"
                f"{class_directory}"
            )

        for file_path in sorted(class_directory.iterdir()):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            image_paths.append(file_path)
            image_labels.append(class_name)

    return image_paths, image_labels


all_train_paths, all_train_labels = collect_image_files(
    train_dir,
    label2idx
)

test_paths, test_labels = collect_image_files(
    test_dir,
    label2idx
)

if not all_train_paths:
    raise ValueError("Không tìm thấy ảnh trong thư mục train.")

if not test_paths:
    raise ValueError("Không tìm thấy ảnh trong thư mục test.")


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(
        all_train_paths,
        all_train_labels,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=all_train_labels
    )
)

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Test samples: {len(test_paths)}")


# =========================================================
# IMAGE DATASET
# =========================================================

class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        image_labels,
        label_mapping,
        image_size=(128, 128),
        normalize=True
    ):
        self.image_paths = list(image_paths)
        self.image_labels = list(image_labels)
        self.label_mapping = label_mapping
        self.normalize = normalize

        self.resize = Resize(
            image_size,
            antialias=True
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        class_name = self.image_labels[index]

        # Luôn đọc ảnh dưới dạng grayscale: [1, H, W]
        image = read_image(
            str(image_path),
            mode=ImageReadMode.GRAY
        )

        # Resize về [1, IMG_HEIGHT, IMG_WIDTH]
        image = self.resize(image)

        # uint8 → float32
        image = image.float()

        # Pixel [0, 255] → [-1, 1]
        if self.normalize:
            image = image / 127.5 - 1.0

        label = self.label_mapping[class_name]

        return image, label


# =========================================================
# DATASETS
# =========================================================

train_dataset = ImageDataset(
    image_paths=train_paths,
    image_labels=train_labels,
    label_mapping=label2idx,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    normalize=True
)

val_dataset = ImageDataset(
    image_paths=val_paths,
    image_labels=val_labels,
    label_mapping=label2idx,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    normalize=True
)

test_dataset = ImageDataset(
    image_paths=test_paths,
    image_labels=test_labels,
    label_mapping=label2idx,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    normalize=True
)


# =========================================================
# DATA LOADERS
# =========================================================

data_generator = torch.Generator()
data_generator.manual_seed(RANDOM_STATE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=data_generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# DISPLAY SAMPLE IMAGES
# =========================================================

image_batch, label_batch = next(iter(train_loader))

number_of_images = min(9, len(image_batch))

plt.figure(figsize=(10, 10))

for index in range(number_of_images):
    plt.subplot(3, 3, index + 1)

    image = image_batch[index]

    # Đưa ảnh từ [-1, 1] về [0, 1] để hiển thị
    image = (image + 1.0) / 2.0

    # [1, H, W] → [H, W]
    image = image.squeeze(0).numpy()

    label_index = label_batch[index].item()
    class_name = idx2label[label_index]

    plt.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# MLP MODEL
# =========================================================

class MLP(nn.Module):
    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims
    ):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(
            input_dims,
            hidden_dims * 4
        )

        self.linear2 = nn.Linear(
            hidden_dims * 4,
            hidden_dims * 2
        )

        self.linear3 = nn.Linear(
            hidden_dims * 2,
            hidden_dims
        )

        self.output = nn.Linear(
            hidden_dims,
            output_dims
        )

    def forward(self, x):
        x = self.flatten(x)

        x = self.linear1(x)
        x = torch.relu(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.linear3(x)
        x = torch.relu(x)

        return self.output(x)


sample_image, _ = train_dataset[0]

input_dims = sample_image.numel()
output_dims = len(classes)

model = MLP(
    input_dims=input_dims,
    hidden_dims=HIDDEN_DIMS,
    output_dims=output_dims
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print(model)
print(f"Input dimensions: {input_dims}")
print(f"Output dimensions: {output_dims}")


# =========================================================
# METRIC FUNCTIONS
# =========================================================

def compute_accuracy(logits, targets):
    predicted_classes = torch.argmax(
        logits,
        dim=1
    )

    correct_predictions = (
        predicted_classes == targets
    ).sum().item()

    return correct_predictions / len(targets)


def evaluate_model(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += (
                loss.item() * images.size(0)
            )

            all_predictions.append(
                outputs.cpu()
            )

            all_targets.append(
                labels.cpu()
            )

    average_loss = (
        total_loss / len(data_loader.dataset)
    )

    all_predictions = torch.cat(
        all_predictions,
        dim=0
    )

    all_targets = torch.cat(
        all_targets,
        dim=0
    )

    accuracy = compute_accuracy(
        all_predictions,
        all_targets
    )

    return average_loss, accuracy


# =========================================================
# TRAINING
# =========================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_loss = float("inf")
best_model_state = None


for epoch in range(EPOCHS):
    model.train()

    total_train_loss = 0.0
    train_predictions = []
    train_targets = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        total_train_loss += (
            loss.item() * images.size(0)
        )

        train_predictions.append(
            outputs.detach().cpu()
        )

        train_targets.append(
            labels.cpu()
        )

    average_train_loss = (
        total_train_loss
        / len(train_loader.dataset)
    )

    train_predictions = torch.cat(
        train_predictions,
        dim=0
    )

    train_targets = torch.cat(
        train_targets,
        dim=0
    )

    train_accuracy = compute_accuracy(
        train_predictions,
        train_targets
    )

    val_loss, val_accuracy = evaluate_model(
        model,
        val_loader,
        criterion
    )

    train_losses.append(
        average_train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(
            model.state_dict()
        )

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train loss: {average_train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.4f} | "
        f"Val accuracy: {val_accuracy:.4f}"
    )


# =========================================================
# LOAD BEST MODEL
# =========================================================

if best_model_state is not None:
    model.load_state_dict(
        best_model_state
    )


# =========================================================
# TRAINING HISTORY
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

axes[0].plot(
    train_losses,
    label="Training loss"
)

axes[0].plot(
    val_losses,
    label="Validation loss"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()


axes[1].plot(
    train_accuracies,
    label="Training accuracy"
)

axes[1].plot(
    val_accuracies,
    label="Validation accuracy"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training and Validation Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()


# =========================================================
# FINAL EVALUATION
# =========================================================

val_loss, val_accuracy = evaluate_model(
    model,
    val_loader,
    criterion
)

test_loss, test_accuracy = evaluate_model(
    model,
    test_loader,
    criterion
)

print("\nEvaluation on validation set:")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {val_accuracy:.4f}")

print("\nEvaluation on test set:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")