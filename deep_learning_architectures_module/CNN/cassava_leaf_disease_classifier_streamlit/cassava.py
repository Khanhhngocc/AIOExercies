# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

# =========================================================
# 2. CONFIGURATION
# =========================================================
DATA_PATHS = {
    "train": "./train",
    "valid": "./validation",
    "test": "./test",
}

IMAGE_SIZE = 150
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 2e-4

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "lenet_cassava_best.pt"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# =========================================================
# 3. IMAGE LOADER
# =========================================================
def loader(path):
    return Image.open(path).convert("RGB")


# =========================================================
# 4. IMAGE PREPROCESSING
# =========================================================
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

valid_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =========================================================
# 5. CREATE DATASETS WITH ImageFolder
# =========================================================
train_data = datasets.ImageFolder(
    root=DATA_PATHS["train"],
    loader=loader,
    transform=train_transforms,
)

valid_data = datasets.ImageFolder(
    root=DATA_PATHS["valid"],
    loader=loader,
    transform=valid_test_transforms,
)

test_data = datasets.ImageFolder(
    root=DATA_PATHS["test"],
    loader=loader,
    transform=valid_test_transforms,
)

# =========================================================
# 6. DATASET SANITY CHECK
# =========================================================
print("Classes:", train_data.classes)
print("Class-to-index mapping:", train_data.class_to_idx)

print("Number of training images:", len(train_data))
print("Number of validation images:", len(valid_data))
print("Number of test images:", len(test_data))

# Verify that all splits use the same class-label mapping
assert train_data.class_to_idx == valid_data.class_to_idx
assert train_data.class_to_idx == test_data.class_to_idx

sample_image, sample_label = train_data[0]

print("Sample image shape:", sample_image.shape)
print("Sample image label:", sample_label)
print("Sample image class:", train_data.classes[sample_label])


# =========================================================
# 7. DATA EXPLORATION
# =========================================================
def plot_images(images, labels, classes):
    n_images = len(images)
    rows = int(n_images ** 0.5)
    cols = int(n_images ** 0.5)

    fig = plt.figure(figsize=(10, 10))

    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)

        # Convert [C, H, W] to [H, W, C] for Matplotlib
        image = images[i].permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(classes[labels[i]])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


N_IMAGES = 25

sample_images = [train_data[i][0] for i in range(N_IMAGES)]
sample_labels = [train_data[i][1] for i in range(N_IMAGES)]

plot_images(sample_images, sample_labels, train_data.classes)

# =========================================================
# 8. CREATE DATALOADERS
# =========================================================
train_dataloader = data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

valid_dataloader = data.DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_dataloader = data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# =========================================================
# 9. DATALOADER SANITY CHECK
# =========================================================
inputs, labels = next(iter(train_dataloader))

print("Input batch shape:", inputs.shape)
print("Label batch shape:", labels.shape)

# Expected input shape: [batch_size, 3, 150, 150]

# =========================================================
# 10. BUILD THE LENET MODEL
# =========================================================
class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Input: [batch_size, 3, 150, 150]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5,
            padding='same'
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        # Shape after first pooling: [batch_size, 6, 75, 75]
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        # Shape after second pooling: [batch_size, 16, 35, 35]
        self.flattern = nn.Flattern()

        self.fc_1 = nn.Linear(16 * 35 * 35, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)

        outputs = self.flatten(outputs)

        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)

        return outputs

# =========================================================
# 11. TRAINING FUNCTION
# =========================================================
def train(model, criterion, optimizer, train_dataloader, device, epoch=0, log_interval=50):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = criterion(predictions, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_acc, epoch_loss

# =========================================================
# 12. EVALUATION FUNCTION
# =========================================================
@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        predictions = model(inputs)
        loss = criterion(predictions, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_acc, epoch_loss


# =========================================================
# 13. INITIALIZE MODEL, LOSS FUNCTION, AND OPTIMIZER
# =========================================================
num_classes = len(train_data.classes)

print("Using device:", DEVICE)
print("Number of classes:", num_classes)

lenet_model = LeNetClassifier(num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    lenet_model.parameters(),
    lr=LEARNING_RATE,
)

# Run one forward pass to verify the model output shape
sample_predictions = lenet_model(inputs.to(DEVICE))
print("Prediction tensor shape:", sample_predictions.shape)

# Expected output shape: [batch_size, number_of_classes]


# =========================================================
# 14. TRAINING LOOP
# =========================================================
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_accs, train_losses = [], []
eval_accs, eval_losses = [], []

best_loss_eval = float("inf")

for epoch in range (1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()

    train_acc, train_loss = train(
        lenet_model,
        criterion,
        optimizer,
        train_dataloader,
        DEVICE
    )

    train_accs.append(train_acc)
    train_losses.append(train_loss)

    # Evaluate on the validation set
    eval_acc, eval_loss = evaluate(
        lenet_model,
        criterion,
        valid_dataloader
    )

    eval_accs.append(eval_acc)
    eval_losses.append(eval_loss)

    # Save the model only when validation loss improves
    if eval_loss < best_loss_eval:
        best_loss_eval = eval_loss

        torch.save(
            lenet_model.state_dict(),
            MODEL_DIR
        )

    elapsed = time.time() - epoch_start_time

    print("-" * 60)
    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Time: {elapsed:.2f}s | "
        f"Train Accuracy: {train_acc:.3%} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Valid Accuracy: {eval_acc:.3%} | "
        f"Valid Loss: {eval_loss:.4f}"
    )
    print("-" * 60)

# =========================================================
# 15. LOAD THE BEST VALIDATION MODEL
# =========================================================
lenet_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

lenet_model.eval()

# =========================================================
# 16. TEST EVALUATION
# =========================================================
test_acc, test_loss = evaluate(
    lenet_model,
    criterion,
    test_dataloader,
    DEVICE,
)

print("\nFinal test result")
print(f"Test Accuracy: {test_acc:.3%}")
print(f"Test Loss: {test_loss:.4f}")

# =========================================================
# 17. PLOT TRAINING HISTORY
# =========================================================
def plot_result(train_accs, eval_accs, train_losses, eval_losses):
    epochs = list(range(1, len(train_accs) + 1))

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    # Plot accuracy
    axs[0].plot(epochs, train_accs, label="Training")
    axs[0].plot(epochs, eval_accs, label="Validation")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Training and Validation Accuracy")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Plot loss
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Validation")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Training and Validation Loss")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_result(
    train_accs,
    eval_accs,
    train_losses,
    eval_losses,
)

# =========================================================
# 18. INFERENCE ON A NEW CASSAVA LEAF IMAGE
# =========================================================
def load_model(model_path, num_classes):
    model = LeNetClassifier(num_classes)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model


def inference(img_path, model, classes):
    # Load an external image
    image = Image.open(img_path).convert("RGB")

    # Apply the same preprocessing used during training
    img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    image_tensor = img_transform(image)

    # Add a batch dimension: [3, 150, 150] -> [1, 3, 150, 150]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Get prediction logits
    with torch.no_grad():
        predictions = model(image_tensor)

    # Convert logits into probabilities
    probabilities = torch.softmax(predictions, dim=1)

    # Select the class with the highest probability
    confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = classes[predicted_index.item()]

    return predicted_class, confidence.item()