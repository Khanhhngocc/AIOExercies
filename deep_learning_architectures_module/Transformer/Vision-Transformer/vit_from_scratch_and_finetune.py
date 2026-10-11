import os
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image


# ============================================================
# 1. SETUP
# ============================================================
random_state = 59

random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


device = get_device()

print("Device:", device)


# ============================================================
# 2. LOAD DATASET
# ============================================================
data_path = "./flower_photos"

base_dataset = ImageFolder(root=data_path)

num_samples = len(base_dataset)
classes = base_dataset.classes
num_classes = len(classes)

print("Number of samples:", num_samples)
print("Classes:", classes)
print("Number of classes:", num_classes)


# ============================================================
# 3. TRAIN / VALID / TEST SPLIT
# ============================================================
train_ratio = 0.8
valid_ratio = 0.1

n_train = int(num_samples * train_ratio)
n_valid = int(num_samples * valid_ratio)
n_test = num_samples - n_train - n_valid

generator = torch.Generator().manual_seed(random_state)

train_subset, valid_subset, test_subset = random_split(
    base_dataset,
    [n_train, n_valid, n_test],
    generator=generator
)

print("Train:", len(train_subset))
print("Valid:", len(valid_subset))
print("Test :", len(test_subset))


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================
IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 5. DATASET WITH SEPARATE TRANSFORMS
# ============================================================
class ImageSubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = ImageSubsetWithTransform(
    train_subset,
    train_transforms
)

valid_dataset = ImageSubsetWithTransform(
    valid_subset,
    eval_transforms
)

test_dataset = ImageSubsetWithTransform(
    test_subset,
    eval_transforms
)


# ============================================================
# 6. DATALOADER
# ============================================================
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


images, labels = next(iter(train_loader))

print("Image batch:", images.shape)
print("Label batch:", labels.shape)


# ============================================================
# 7. PATCH + POSITION EMBEDDING
# ============================================================
class PatchPositionEmbedding(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        dropout=0.1
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.num_patches = (image_size // patch_size) ** 2

        # Split image into patches + linear projection
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # +1 because of CLS token
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.cls_token, std=0.02)

        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)

        # [B, 3, 224, 224]
        # -> [B, embed_dim, 14, 14]
        x = self.patch_embedding(x)

        # -> [B, embed_dim, 196]
        x = x.flatten(2)

        # -> [B, 196, embed_dim]
        x = x.transpose(1, 2)

        # [1, 1, embed_dim]
        # -> [B, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # CLS + patch embeddings
        # -> [B, 197, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.position_embedding

        x = self.dropout(x)

        return x


# ============================================================
# 8. TRANSFORMER ENCODER BLOCK
# ============================================================
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout=0.1
    ):
        super().__init__()

        self.norm_1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout_1 = nn.Dropout(dropout)

        self.norm_2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self-Attention
        norm_x = self.norm_1(x)

        attention_output, _ = self.attention(
            norm_x,
            norm_x,
            norm_x,
            need_weights=False
        )

        x = x + self.dropout_1(attention_output)

        # Feed Forward Network
        ffn_output = self.ffn(self.norm_2(x))

        x = x + self.dropout_2(ffn_output)

        return x


# ============================================================
# 9. TRANSFORMER ENCODER
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


# ============================================================
# 10. VISION TRANSFORMER
# ============================================================
class VisionTransformerCls(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        num_classes,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = PatchPositionEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            dropout=dropout
        )

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim,128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Patch + CLS + Position Embedding
        x = self.embedding(x)

        # Transformer Encoder
        x = self.encoder(x)

        # Final normalization
        x = self.norm(x)

        # CLS token
        cls_output = x[:, 0]

        # Classification head
        logits = self.classifier(cls_output)

        return logits


# ============================================================
# 11. TRAIN ONE EPOCH
# ============================================================
def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):
    model.train()

    total_loss = 0
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

        predictions = logits.argmax(dim=1)

        total_correct += predictions.eq(labels).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples

    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


# ============================================================
# 12. EVALUATE
# ============================================================

def evaluate_epoch(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)

            total_correct += predictions.eq(labels).sum().item()

            total_samples += batch_size

    epoch_loss = total_loss / total_samples

    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


# ============================================================
# 13. MODEL CONFIG
# ============================================================
IMAGE_SIZE = 224
PATCH_SIZE = 16

EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 4

DROPOUT = 0.1

model = VisionTransformerCls(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
)

model = model.to(device)

print(model)


# ============================================================
# 14. LOSS + OPTIMIZER
# ============================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)


NUM_EPOCHS = 50

MODEL_PATH = "vit_flowers_scratch.pt"


# ============================================================
# 15. TRAINING LOOP
# ============================================================
train_losses = []
train_accuracies = []

valid_losses = []
valid_accuracies = []

best_valid_loss = float("inf")

patience = 7
epochs_without_improvement = 0


for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()

    train_loss, train_accuracy = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    valid_loss, valid_accuracy = evaluate_epoch(
        model,
        valid_loader,
        criterion,
        device
    )

    train_losses.append(train_loss)

    train_accuracies.append(train_accuracy)

    valid_losses.append(valid_loss)

    valid_accuracies.append(valid_accuracy)

    print(
        f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
        f"| Time: {time.time() - start_time:.2f}s "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Valid Loss: {valid_loss:.4f} "
        f"| Valid Acc: {valid_accuracy:.4f}"
    )

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("-> Best model saved")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# 16. LOAD BEST MODEL
# ============================================================
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# 17. TEST
# ============================================================
test_loss, test_accuracy = evaluate_epoch(
    model,
    test_loader,
    criterion,
    device
)

print(f"\nTest Loss: {test_loss:.4f}")

print(f"Test Accuracy: {test_accuracy:.4f}")


# ============================================================
# 18. PLOT RESULT
# ============================================================

epochs = range(1,len(train_losses) + 1)

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_losses, label="Training")
plt.plot(epochs, valid_losses, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(epochs, train_accuracies, label="Training")
plt.plot(epochs, valid_accuracies, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.show()



# ============================================================
# 19. INFERENCE
# ============================================================
def predict_image(image_path, model, transform, classes, device):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    model.eval()

    with torch.inference_mode():
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    predicted_class = probabilities.argmax(dim=1).item()

    confidence = probabilities[0, predicted_class].item()

    return classes[predicted_class], confidence


label, confidence = predict_image(
    image_path="flower.jpg",
    model=model,
    transform=eval_transforms,
    classes=classes,
    device=device
)

print("Prediction:", label)
print(f"Confidence: {confidence:.4f}")



# ============================================================
# 20. PRETRAINED MODEL
# ============================================================
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)

model_name = "google/vit-base-patch16-224-in21k"

image_processor = AutoImageProcessor.from_pretrained(model_name)


# ============================================================
# 21. PRETRAINED IMAGE TRANSFORMS
# ============================================================
image_mean = image_processor.image_mean
image_std = image_processor.image_std

if "height" in image_processor.size:
    pretrained_image_size = image_processor.size["height"]

else:
    pretrained_image_size = image_processor.size["shortest_edge"]


normalize = transforms.Normalize(
    mean=image_mean,
    std=image_std
)


pretrained_train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(pretrained_image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


pretrained_eval_transforms = transforms.Compose([
    transforms.Resize((pretrained_image_size, pretrained_image_size)),
    transforms.ToTensor(),
    normalize
])

# ============================================================
# 22. PRETRAINED DATASETS
# ============================================================
pretrained_train_dataset = ImageSubsetWithTransform(
    train_subset,
    pretrained_train_transforms
)


pretrained_valid_dataset = ImageSubsetWithTransform(
    valid_subset,
    pretrained_eval_transforms
)


pretrained_test_dataset = ImageSubsetWithTransform(
    test_subset,
    pretrained_eval_transforms
)


# ============================================================
# 23. LABEL MAPPING
# ============================================================
id2label = {
    i: label
    for i, label in enumerate(classes)
}

label2id = {
    label: i
    for i, label in enumerate(classes)
}

print(id2label)


# ============================================================
# 24. LOAD PRETRAINED VIT
# ============================================================
pretrained_model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id
)

# ============================================================
# 25. COLLATE FUNCTION
# ============================================================
def collate_fn(examples):
    pixel_values = torch.stack([
        image
        for image, _ in examples
    ])

    labels = torch.tensor([
        label
        for _, label in examples
    ])

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


# ============================================================
# 26. METRIC
# ============================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = (predictions == labels).mean()

    return {
        "accuracy": accuracy
    }


# ============================================================
# 27. TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir="./vit_flowers_pretrained",

    learning_rate=2e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=5,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",
    greater_is_better=True,

    save_total_limit=1,

    remove_unused_columns=False,

    report_to="none",

    seed=random_state
)


# ============================================================
# 28. TRAINER
# ============================================================
trainer = Trainer(
    model=pretrained_model,
    args=training_args,

    train_dataset=pretrained_train_dataset,

    eval_dataset=pretrained_valid_dataset,

    data_collator=collate_fn,

    compute_metrics=compute_metrics,

    processing_class=image_processor
)


# ============================================================
# 29. TRAIN
# ============================================================
trainer.train()


# ============================================================
# 30. TEST
# ============================================================
test_results = trainer.evaluate(pretrained_test_dataset)

print("\nTest Results:")

for metric_name, value in test_results.items():
    print(f"{metric_name}: {value}")


# ============================================================
# 31. SAVE MODEL
# ============================================================
save_path = "./vit_flowers_best"

trainer.save_model(save_path)

image_processor.save_pretrained(save_path)

print(f"Model saved at: {save_path}")


# ============================================================
# 32. PRETRAINED INFERENCE
# ============================================================
def predict_pretrained_image(image_path, model, image_processor):
    image = Image.open(image_path).convert("RGB")

    inputs = image_processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.inference_mode():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    predicted_class = probabilities.argmax(dim=1).item()

    confidence = probabilities[0, predicted_class].item()

    predicted_label = (
        model.config.id2label[
            predicted_class
        ]
    )

    return predicted_label, confidence


label, confidence = (
    predict_pretrained_image(
        image_path="flower.jpg",
        model=trainer.model,
        image_processor=image_processor
    )
)

print("Prediction:", label)

print(f"Confidence: {confidence:.4f}")