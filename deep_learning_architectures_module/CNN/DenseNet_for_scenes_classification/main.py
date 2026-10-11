# =========================================================
# 1. PROBLEM DEFINITION
#    Classify a scene image into one of the available scene
#    categories using a DenseNet-121 architecture.
# =========================================================


# =========================================================
# 2. IMPORT LIBRARIES
# =========================================================
from pathlib import Path
from collections import Counter
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# =========================================================
# 3. CONFIGURATION
# =========================================================
SEED = 59

# Expected dataset structure:
# scenes_classification/
# ├── train/
# │   ├── class_1/
# │   ├── class_2/
# │   └── ...
# └── val/
#     ├── class_1/
#     ├── class_2/
#     └── ...

ROOT_DIR = "scenes_classification"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "val")

IMAGE_SIZE = (224, 224)
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 8

LEARNING_RATE = 1e-2
NUM_EPOCHS = 15

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "scenes_densenet_best.pt"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# =========================================================
# 5. DATA LOADING
# =========================================================
def load_image_paths(train_dir, test_dir):
    classes = {
        label_idx: class_name
        for label_idx, class_name in enumerate(
            sorted(
                class_name
                for class_name in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, class_name))
            )
        )
    }

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for dataset_path in [train_dir, test_dir]:
        for label_idx, class_name in classes.items():
            class_dir = os.path.join(dataset_path, class_name)

            if not os.path.isdir(class_dir):
                continue

            for image_filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_filename)

                if "train" in dataset_path:
                    X_train.append(image_path)
                    y_train.append(label_idx)
                else:
                    X_test.append(image_path)
                    y_test.append(label_idx)

    return X_train, y_train, X_test, y_test, classes

X_train, y_train, X_test, y_test, classes = load_image_paths(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
)

print(f"Number of training images: {len(X_train)}")
print(f"Number of test images    : {len(X_test)}")
print(f"Classes                  : {classes}")


# =========================================================
# 6. DATA CLEANING
# =========================================================
def is_valid_image(image_path: str) -> bool:
    """Check whether an image file can be opened safely."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError):
        return False

def remove_invalid_images(image_paths, labels):
    """Remove corrupted or unreadable image files."""
    valid_paths = []
    valid_labels = []

    for image_path, label in zip(image_paths, labels):
        if is_valid_image(image_path):
            valid_paths.append(image_path)
            valid_labels.append(label)

    return valid_paths, valid_labels


X_train, y_train = remove_invalid_images(X_train, y_train)
X_test, y_test = remove_invalid_images(X_test, y_test)

# =========================================================
# 7. EXPLORATORY DATA ANALYSIS
# =========================================================
def plot_class_distribution(labels, classes, title):
    """Visualize the number of images in each class."""
    counts = Counter(labels)

    class_names = [classes[index] for index in sorted(classes.keys())]
    class_counts = [counts[index] for index in sorted(classes.keys())]

    plt.figure(figsize=(12, 5))
    plt.bar(class_names, class_counts)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_sample_images(image_paths, labels, classes, n_images=12):
    """Display sample images with their class names."""
    n_images = min(n_images, len(image_paths))
    selected_indices = np.random.choice(
        len(image_paths),
        size=n_images,
        replace=False,
    )

    cols = 4
    rows = int(np.ceil(n_images / cols))

    plt.figure(figsize=(14, 3 * rows))

    for plot_index, image_index in enumerate(selected_indices, start=1):
        image = Image.open(image_paths[image_index]).convert("RGB")

        plt.subplot(rows, cols, plot_index)
        plt.imshow(image)
        plt.title(classes[labels[image_index]])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

plot_class_distribution(
    labels=y_train,
    classes=classes,
    title="Training Dataset Class Distribution",
)

plot_sample_images(
    image_paths=X_train,
    labels=y_train,
    classes=classes,
    n_images=12,
)

# =========================================================
# 8. TRAIN / VALIDATION SPLIT
# =========================================================
val_size = 0.2
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=val_size,
    random_state=SEED,
    shuffle=is_shuffle
)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Test samples       : {len(X_test)}")

# =========================================================
# 9. IMAGE PREPROCESSING
# =========================================================
def transform(img, img_size=(224, 244)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3]
    img = torch.tensor(img).permute(2, 0, 1).float()
    normalized_img = img / 255.0

    return normalized_img


# =========================================================
# 10. PYTORCH DATASET
# =========================================================
class ScenesDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.img_paths = X
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label


# =========================================================
# 11. DATASET OBJECTS
# =========================================================
train_dataset = ScenesDataset(
    image_paths=X_train,
    labels=y_train,
    transform=transform,
)

val_dataset = ScenesDataset(
    image_paths=X_val,
    labels=y_val,
    transform=transform,
)

test_dataset = ScenesDataset(
    image_paths=X_test,
    labels=y_test,
    transform=transform,
)

# =========================================================
# 12. DATALOADERS
# =========================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")


# =========================================================
# 13. DENSENET BOTTLENECK BLOCK
# =========================================================
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            4 * growth_rate,
            kernel_size=1,
            bias=False
        )

        self.batch_norm2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # Keep the input features for DenseNet concatenation.
        res = x.clone()

        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv2(x)

        # Concatenate old and new feature maps on channel axis.
        x = torch.cat([res, x], 1)

        return x

# =========================================================
# 14. DENSE BLOCK
# =========================================================
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()

        layers = []

        for layer_index in range(num_layers):
            layers.append(
                BottleneckBlock(
                    in_channels=in_channels + layer_index * growth_rate,
                    growth_rate=growth_rate,
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# =========================================================
# 15. DENSENET MODEL
# =========================================================
class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=2 * growth_rate,
            kernel_size=7,
            padding=3,
            stride=2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.dense_blocks = nn.ModuleList()

        in_channels = 2 * growth_rate

        for block_index, num_layers in enumerate(num_blocks):
            self.dense_blocks.append(
                DenseBlock(
                    num_layers=num_layers,
                    in_channels=in_channels,
                    growth_rate=growth_rate,
                )
            )
            in_channels += num_layers * growth_rate

            if block_index != len(num_blocks) - 1:
                out_channels = in_channels // 2

                self.dense_blocks.append(
                    nn.Sequential(
                        nn.BatchNorm2d(in_channels),

                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=1,
                            bias=False
                        ),
                        nn.AvgPool2d(kernel_size=2)
                    )
                )

                in_channels = out_channels

        self.bn2 =  nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        # Adaptive pooling works safely with image size changes.
        self.pool2 = nn.AvgPool2d(kernel_size=7)

        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for block in self.dense_blocks:
            x = block(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


# =========================================================
# 16. MODEL INITIALIZATION
# =========================================================
N_CLASSES = len(list(classes.keys()))

model = DenseNet(
    [6, 12, 24, 16],
    growth_rate=32,
    num_classes=N_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
)

print(f"Using device: {DEVICE}")
print(model)

# =========================================================
# 17. TRAINING FOR ONE EPOCH
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy

# =========================================================
# 18. EVALUATION FUNCTION
# =========================================================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy

# =========================================================
# 19. MODEL TRAINING AND CHECKPOINT SAVING
# =========================================================
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    model_path,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "num_blocks": [6, 12, 24, 16],
                    "growth_rate": 32,
                    "num_classes": N_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                model_path,
            )

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

    return history

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    model_path=MODEL_PATH,
)

# =========================================================
# 21. TRAINING HISTORY VISUALIZATION
# =========================================================
def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Training")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_training_history(history)


# =========================================================
# 22. LOAD THE BEST MODEL
# =========================================================
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

print(f"Best model saved at epoch: {checkpoint['epoch']}")
print(f"Best validation accuracy : {checkpoint['val_acc']:.2%}")


# =========================================================
# 23. FINAL TEST EVALUATION
# =========================================================
test_loss, test_acc = evaluate(
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    device=DEVICE,
)

print("\nFinal evaluation on the test dataset")
print(f"Test loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_acc:.2%}")

# =========================================================
# 24. INFERENCE FUNCTION
# =========================================================
@torch.no_grad()
def predict_scene(image_path, model, classes, device):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = classes[predicted_index.item()]

    return predicted_class, confidence.item()

# =========================================================
# 25. INFERENCE EXAMPLE
# =========================================================
example_image_path = X_test[0]

predicted_class, confidence = predict_scene(
    image_path=example_image_path,
    model=model,
    classes=classes,
    device=DEVICE,
)

print("\nInference example")
print(f"Image path      : {example_image_path}")
print(f"Predicted class : {predicted_class}")
print(f"Confidence      : {confidence:.2%}")

image = Image.open(example_image_path).convert("RGB")
plt.figure(figsize=(6, 4))
plt.imshow(image)
plt.title(f"Prediction: {predicted_class} ({confidence:.2%})")
plt.axis("off")
plt.show()

