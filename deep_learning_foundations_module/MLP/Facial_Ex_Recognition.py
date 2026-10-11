# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
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
# 2. CONFIGURATION
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


# =========================================================
# 3. SET RANDOM SEEDS AND DEVICE
# =========================================================
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Use CUDA when available, otherwise Apple MPS, then CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")


# =========================================================
# 4. DEFINE DATA PATHS
# =========================================================
current_dir = (
    Path(__file__).resolve().parent
    / "datasets"
    / "[Data]_Ex3_FER-2013"
)

train_dir = current_dir / "train"
test_dir = current_dir / "test"

if not train_dir.exists():
    raise FileNotFoundError(
        f"Train directory was not found:\n{train_dir}"
    )

if not test_dir.exists():
    raise FileNotFoundError(
        f"Test directory was not found:\n{test_dir}"
    )


# =========================================================
# 5. CREATE CLASS-TO-INDEX MAPPINGS
# =========================================================
# Each subfolder in the training directory represents one class.
classes = sorted(
    [
        folder.name
        for folder in train_dir.iterdir()
        if folder.is_dir()
    ]
)

if not classes:
    raise ValueError(
        f"No class directories were found in:\n{train_dir}"
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
# 6. COLLECT IMAGE FILE PATHS
# =========================================================
def collect_image_files(image_directory, class_mapping):
    """
    Collect valid image paths and their class labels
    from a directory structured by class folders.
    """
    image_paths = []
    image_labels = []

    for class_name in class_mapping:
        class_directory = image_directory / class_name

        if not class_directory.exists():
            raise FileNotFoundError(
                f"Class directory was not found:\n"
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
    label2idx,
)

test_paths, test_labels = collect_image_files(
    test_dir,
    label2idx,
)

if not all_train_paths:
    raise ValueError(
        "No images were found in the train directory."
    )

if not test_paths:
    raise ValueError(
        "No images were found in the test directory."
    )


# =========================================================
# 7. CREATE TRAIN / VALIDATION SPLIT
# =========================================================
# Keep the provided test folder untouched for final evaluation.
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_train_paths,
    all_train_labels,
    train_size=TRAIN_RATIO,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=all_train_labels,
)

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Test samples: {len(test_paths)}")


# =========================================================
# 8. CREATE A CUSTOM IMAGE DATASET
# =========================================================
class ImageDataset(Dataset):
    """
    Load grayscale images, resize them, normalize pixel values,
    and return image-label pairs.
    """

    def __init__(
        self,
        image_paths,
        image_labels,
        label_mapping,
        image_size=(128, 128),
        normalize=True,
    ):
        self.image_paths = list(image_paths)
        self.image_labels = list(image_labels)
        self.label_mapping = label_mapping
        self.normalize = normalize

        self.resize = Resize(
            image_size,
            antialias=True,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        class_name = self.image_labels[index]

        # Read images in grayscale format: [1, height, width].
        image = read_image(
            str(image_path),
            mode=ImageReadMode.GRAY,
        )

        # Resize every image to a fixed input size.
        image = self.resize(image)

        # Convert uint8 pixel values into float32.
        image = image.float()

        # Normalize pixel values from [0, 255] to [-1, 1].
        if self.normalize:
            image = image / 127.5 - 1.0

        label = self.label_mapping[class_name]

        return image, label


# =========================================================
# 9. CREATE DATASETS AND DATALOADERS
# =========================================================
train_dataset = ImageDataset(
    image_paths=train_paths,
    image_labels=train_labels,
    label_mapping=label2idx,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
)

val_dataset = ImageDataset(
    image_paths=val_paths,
    image_labels=val_labels,
    label_mapping=label2idx,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
)

test_dataset = ImageDataset(
    image_paths=test_paths,
    image_labels=test_labels,
    label_mapping=label2idx,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
)

# Use a seeded generator for reproducible training-data shuffling.
data_generator = torch.Generator()
data_generator.manual_seed(RANDOM_STATE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=data_generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# =========================================================
# 10. DISPLAY SAMPLE TRAINING IMAGES
# =========================================================
image_batch, label_batch = next(iter(train_loader))

number_of_images = min(9, len(image_batch))

plt.figure(figsize=(10, 10))

for index in range(number_of_images):
    plt.subplot(3, 3, index + 1)

    # Convert normalized values from [-1, 1] back to [0, 1].
    image = (image_batch[index] + 1.0) / 2.0

    # Convert [1, H, W] into [H, W] for matplotlib.
    image = image.squeeze(0).numpy()

    label_index = label_batch[index].item()
    class_name = idx2label[label_index]

    plt.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# 11. BUILD THE MLP MODEL
# =========================================================
class MLP(nn.Module):
    """
    A fully connected neural network for image classification.

    Images are flattened before being passed through
    three hidden layers and one output layer.
    """

    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims,
    ):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(
            input_dims,
            hidden_dims * 4,
        )

        self.linear2 = nn.Linear(
            hidden_dims * 4,
            hidden_dims * 2,
        )

        self.linear3 = nn.Linear(
            hidden_dims * 2,
            hidden_dims,
        )

        self.output = nn.Linear(
            hidden_dims,
            output_dims,
        )

    def forward(self, x):
        x = self.flatten(x)

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))

        # Return raw class scores (logits).
        return self.output(x)


sample_image, _ = train_dataset[0]

input_dims = sample_image.numel()
output_dims = len(classes)

model = MLP(
    input_dims=input_dims,
    hidden_dims=HIDDEN_DIMS,
    output_dims=output_dims,
).to(device)

# CrossEntropyLoss expects raw logits and integer class indices.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print(model)
print(f"Input dimensions: {input_dims}")
print(f"Output dimensions: {output_dims}")


# =========================================================
# 12. EVALUATION FUNCTION
# =========================================================
def evaluate_model(model, data_loader, criterion):
    """
    Calculate average loss and accuracy for a complete dataset.
    """
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


# =========================================================
# 13. TRAIN THE MODEL
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

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * images.size(0)

    average_train_loss = (
        total_train_loss
        / len(train_loader.dataset)
    )

    # Evaluate the updated model on the complete train and validation sets.
    _, train_accuracy = evaluate_model(
        model,
        train_loader,
        criterion,
    )

    val_loss, val_accuracy = evaluate_model(
        model,
        val_loader,
        criterion,
    )

    train_losses.append(average_train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Keep the model state that achieved the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


# =========================================================
# 14. LOAD THE BEST VALIDATION CHECKPOINT
# =========================================================
if best_model_state is not None:
    model.load_state_dict(best_model_state)


# =========================================================
# 15. VISUALIZE TRAINING HISTORY
# =========================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5),
)

axes[0].plot(
    train_losses,
    label="Training Loss",
)

axes[0].plot(
    val_losses,
    label="Validation Loss",
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    train_accuracies,
    label="Training Accuracy",
)

axes[1].plot(
    val_accuracies,
    label="Validation Accuracy",
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training and Validation Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# 16. FINAL EVALUATION
# =========================================================
val_loss, val_accuracy = evaluate_model(
    model,
    val_loader,
    criterion,
)

test_loss, test_accuracy = evaluate_model(
    model,
    test_loader,
    criterion,
)

print("\nEvaluation on Validation Set:")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {val_accuracy:.4f}")

print("\nEvaluation on Test Set:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")