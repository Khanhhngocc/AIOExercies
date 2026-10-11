# =========================================================
# 1. PROBLEM DEFINITION
#    Classify a weather image into one of the available
#    weather categories using a ResNet model.
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
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 3. CONFIGURATION
# =========================================================
SEED = 59

ROOT_DIR = 'weather-dataset/dataset'

IMAGE_SIZE = (224, 224)

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64

LEARNING_RATE = 1e-2
EPOCHS = 25

MODEL_DIR = Path('./model')
MODEL_PATH = MODEL_DIR / 'weather_resnet_best.pt'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# =========================================================
# 4. SET RANDOM SEED
# =========================================================
def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================================================
# 5. DATA LOADING
# =========================================================
classes = {
    label_idx: class_name
    for label_idx, class_name in enumerate(
        sorted(
            class_name
            for class_name in os.listdir(ROOT_DIR)
            if os.path.isdir(
                os.path.join(ROOT_DIR, class_name)
            )
        )
    )
}

img_paths = []
labels = []

for label_idx, class_name in classes.item():
    class_dir = os.path.join(ROOT_DIR, class_name)

    for img_filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_filename)

        img_paths.append(img_path)

        labels.append(label_idx)


# =========================================================
# 6. DATA CLEANING:
#    Remove invalid or corrupted image files
# =========================================================
def is_valid_image(img_path):
    try:
        with Image.open(img_path) as image:
            image.verify()

        return True

    except (OSError, SyntaxError):
        return False

valid_img_paths = []
valid_labels = []

for img_path, label in zip(img_paths, labels):
    if is_valid_image(img_path):
        valid_img_paths.append(img_path)

        valid_labels.append(label)

img_paths = valid_img_paths
labels = valid_labels

# =========================================================
# 7. DATA UNDERSTANDING / EDA
# =========================================================
print('Number of classes:', len(classes))
print('\nClass mapping:')
print(classes)
print('\nNumber of valid images:', len(img_paths))

label_counts = Counter(labels)

print('\nClass distribution:')

for label_idx, class_name in classes.items():
    print(
        f'{class_name}: '
        f'{label_counts[label_idx]} images'
    )

plt.figure(figsize=(8, 4))
plt.bar(
    [classes[label_idx] for label_idx in classes],
    [label_counts[label_idx] for label_idx in classes],
    color='steelblue'
)

plt.title('Weather Class Distribution')

plt.xlabel('Weather Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# =========================================================
# 8. VISUALIZE SAMPLE RAW IMAGES
# =========================================================
def plot_sample_images(img_paths, labels, classes, n_images=9):
    selected_indices = random.sample(range(len(img_paths)), n_images)

    fig = plt.figure(figsize=(10, 10))

    for plot_idx, data_idx in enumerate(selected_indices):
        image = Image.open(img_paths[data_idx]).convert('RGB')

        label = labels[data_idx]

        ax = fig.add_subplot(3, 3, plot_idx + 1)

        ax.imshow(image)

        ax.set_title(classes[label])

        ax.axis('off')

    plt.tight_layout()

    plt.show()

plot_sample_images(img_paths, labels, classes)

# =========================================================
# 9. TRAIN / VALIDATION / TEST SPLIT
# =========================================================
val_size = 0.2
test_size = 0.125
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    img_paths, labels,
    test_size=val_size,
    random_state=SEED,
    shuffle=is_shuffle
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
    test_size=test_size,
    random_state=SEED,
    shuffle=is_shuffle
)

print('\nSplit sizes:')
print('Train:', len(X_train))
print('Validation:', len(X_val))
print('Test:', len(X_test))

# =========================================================
# 10. IMAGE TRANSFORM:
#     Resize, convert to tensor, and normalize pixel values
# =========================================================
def transform(img, img_size=(224,224)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3]
    img = torch.tensor(img).permute(2, 0, 1).float()

    normalized_img = img / 225.0

    return normalized_img


# =========================================================
# 11. CREATE CUSTOM PYTORCH DATASET
# =========================================================
class WeatherDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.img_paths = X,
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
# 12. CREATE DATASETS AND DATALOADERS
# =========================================================
train_dataset = WeatherDataset(
    X_train, y_train,
    transform=transform
)
val_dataset = WeatherDataset(
    X_val, y_val,
    transform=transform
)
test_dataset = WeatherDataset(
    X_test, y_test,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False
)

# DATALOADER SANITY CHECK
inputs, labels = next(iter(train_loader))

print('\nInput batch shape:', inputs.shape)
print('Label batch shape:', labels.shape)

# Expected input shape: [batch_size, 3, 224, 224]
# Expected label shape: [batch_size]


# =========================================================
# 13. BUILD THE RESIDUAL BLOCK
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),

                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)

        x += self.downsample(shortcut)
        x = self.relu(x)

        return x

# =========================================================
# 14. BUILD THE RESNET MODEL
# =========================================================
class ResNet(nn.Module):
    def __init__(self, residual_block, n_block_list, n_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.batch_norm1 = nn.BatchNorm2d(64)

        self.relu = nn.RelU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv2 = self.create_layer(residual_block, 64, 64, n_block_list[0], 1)
        self.conv3 = self.create_layer(residual_block, 64, 128, n_block_list[1], 2)
        self.conv4 = self.create_layer(residual_block, 128, 256, n_block_list[2], 2)
        self.conv5 = self.create_layer(residual_block, 236, 512, n_block_list[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, n_classes)


    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        blocks = []

        first_block = residual_block(in_channels, out_channels, stride)

        blocks.append(first_block)

        for idx in range (1, n_blocks):
            block = residual_block(out_channels, out_channels, stride=1)

            blocks.append(block)

        block_sequential = nn.Sequential(*blocks)

        return block_sequential

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x

# =========================================================
# 15. INITIALIZE MODEL
# =========================================================
n_classes = len(list(classes.keys()))

model = ResNet(
    residual_block=ResidualBlock,
    n_block_list=[2, 2, 2, 2],
    n_classes=n_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)

print('\nUsing device:', DEVICE)
sample_outputs = model(inputs.to(DEVICE))
print('Model output shape:', sample_outputs.shape)

# =========================================================
# 16. TRAINING FUNCTION
# =========================================================
def train_one_epoch (model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
                outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc


# =========================================================
# 17. VALIDATION / EVALUATION FUNCTION
# =========================================================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
                outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc

# =========================================================
# 18. TRAINING LOOP AND BEST MODEL CHECKPOINT
# =========================================================
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_losses = []
train_accs = []
val_losses = []
val_accs = []

best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Save the model when validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'classes': classes,
                'n_blocks_lst': [2, 2, 2, 2],
                'n_classes': n_classes,
                'image_size': IMAGE_SIZE
            },
            MODEL_PATH
        )

    print(
        f'Epoch {epoch:02d}/{EPOCHS} | '
        f'Train Loss: {train_loss:.4f} | '
        f'Train Acc: {train_acc:.2%} | '
        f'Val Loss: {val_loss:.4f} | '
        f'Val Acc: {val_acc:.2%}'
    )


# =========================================================
# 19. PLOT TRAINING HISTORY
# =========================================================
def plot_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5)
    )

    axes[0].plot(
        epochs,
        train_losses,
        label='Training'
    )
    axes[0].plot(
        epochs,
        val_losses,
        label='Validation'
    )
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        train_accs,
        label='Training'
    )
    axes[1].plot(
        epochs,
        val_accs,
        label='Validation'
    )
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    plt.show()

plot_history(
    train_losses,
    val_losses,
    train_accs,
    val_accs
)


# =========================================================
# 20. LOAD THE BEST MODEL
# =========================================================
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint['model_state_dict']
)

model.eval()

# =========================================================
# 21. FINAL TEST EVALUATION
# =========================================================
test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion,
    DEVICE
)

print('\nFinal evaluation on test dataset')
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.2%}')

# =========================================================
# 22. INFERENCE ON A NEW WEATHER IMAGE
# =========================================================
def predict_weather(img_path, model, classes):
    image = Image.open(img_path).convert('RGB')

    image = transform(image)

    # Add the batch dimension:
    # [3, 224, 224] -> [1, 3, 224, 224]
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_label = torch.max(probabilities, dim=1)

    predicted_class = classes[predicted_label.item()]

    return predicted_class, confidence.item()
