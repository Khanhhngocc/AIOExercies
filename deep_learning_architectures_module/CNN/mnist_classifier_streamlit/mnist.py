# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import os
import random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchsummary import summary

import matplotlib.pyplot as plt
from PIL import Image

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# =========================================================
# 2. DOWNLOAD / LOAD MNIST DATASET
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, "data")

train_data = datasets.MNIST(
    root=ROOT,
    train=True,
    download=True
)

test_data = datasets.MNIST(
    root=ROOT,
    train=False,
    download=True
)

# =========================================================
# 3. PREPROCESSING:
#    Split train/validation + normalize + create DataLoader
# =========================================================
VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(
    train_data,
    [n_train_examples, n_valid_examples]
)

# Compute mean and std for normalization
mean = train_data.dataset.data.float().mean() / 255
std = train_data.dataset.data.float().std() / 255

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

train_data.dataset.transform = train_transforms
valid_data.dataset.transform = test_transforms


# Create DataLoader
BATCH_SIZE = 256

train_dataloader = data.DataLoader(
    train_data,
    shuffle=True,
    batch_size=BATCH_SIZE
)

valid_dataloader = data.DataLoader(
    valid_data,
    batch_size=BATCH_SIZE
)

# =========================================================
# 4. DATA EXPLORATION: Visualize sample MNIST images
# =========================================================
def plot_images(images):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure()

    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(images[i], cmap='bone')
        ax.axis('off')

    plt.show()

N_IMAGES = 25

images = [
    image
    for image, label in [train_data[i] for i in range(N_IMAGES)]
]

plot_images(images)

# DATA LOADER SANITY CHECK: Get one batch and inspect its tensor shape
inputs, labels = next(iter(train_dataloader))

print("Input batch shape:", inputs.shape)
print("Label batch shape:", labels.shape)


# =========================================================
# 5. BUILD MODEL: LeNet CNN
# =========================================================
class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding='same'
        )

        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )

        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
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
# 6. TRAINING FUNCTION:
# =========================================================
def train(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):

    model.train()

    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = criterion(predictions, labels)
        losses.append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time

            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_acc / total_count
                )
            )

            total_acc, total_count = 0, 0
            start_time = time.time()

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)

    return epoch_acc, epoch_loss


# =========================================================
# 7. EVALUATION FUNCTION:
# =========================================================
def evaluate(model, criterion, valid_dataloader, device):
    model.eval()

    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)

    return epoch_acc, epoch_loss


# =========================================================
# 8. SET UP TRAINING:
# =========================================================
num_classes = len(train_data.dataset.classes)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

lenet_model = LeNetClassifier(num_classes)
lenet_model.to(device)

# =========================================================
# 9. MODEL SANITY CHECK: Run one forward pass before training
# =========================================================
sample_predictions = lenet_model(inputs.to(device))
print("Prediction tensor shape:", sample_predictions.shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet_model.parameters())

num_epochs = 10
save_model = './model'

train_accs, train_losses = [], []
eval_accs, eval_losses = [], []

best_loss_eval = 100

# =========================================================
# 10. TRAINING LOOP:
# =========================================================
for epoch in range(1, num_epochs+1):
    epoch_start_time = time.time()

    # Training
    train_acc, train_loss = train(
        lenet_model,
        optimizer,
        criterion,
        train_dataloader,
        device,
        epoch
    )

    train_accs.append(train_acc)
    train_losses.append(train_loss)

    # Evaluation
    eval_acc, eval_loss = evaluate(
        lenet_model,
        criterion,
        valid_dataloader,
        device
    )

    eval_accs.append(eval_acc)
    eval_losses.append(eval_loss)

    # Save best model
    if eval_loss < best_loss_eval:
        torch.save(
            lenet_model.state_dict(),
            save_model + '/lenet_model.pt'
        )

    # Print loss, acc end epoch
    print("-" * 59)

    print(
        "| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f} | Train Loss {:8.3f} "
        "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
            epoch,
            time.time() - epoch_start_time,
            train_acc,
            train_loss,
            eval_acc,
            eval_loss
        )
    )
    print("-" * 59)

# =========================================================
# 11. LOAD BEST MODEL AFTER TRAINING
# =========================================================
lenet_model.load_state_dict(
    torch.load(save_model + '/lenet_model.pt', weights_only=True)
)

lenet_model.eval()

# =========================================================
# 12. TRAINING HISTORY VISUALIZATION: Plot training and validation accuracy/loss
# =========================================================
def plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses):
    epochs = list(range(num_epochs))

    fig, axs = plt.subplots(nrows = 1, ncols =2 , figsize = (12,6))

    # Accuracy chart
    axs[0].plot(epochs, train_accs, label = "Training")
    axs[0].plot(epochs, eval_accs, label = "Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Training and Validation Accuracy")
    axs[0].legend()

    # Loss chart
    axs[1].plot(epochs, train_losses, label = "Training")
    axs[1].plot(epochs, eval_losses, label = "Evaluation")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Training and Validation Loss")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses)


# =========================================================
# 13. TEST EVALUATION
# =========================================================
test_data.transform = test_transforms

test_dataloader = data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE
)

test_acc, test_loss = evaluate(
    lenet_model,
    criterion,
    test_dataloader,
    device
)

# =========================================================
# 14. CNN FILTER VISUALIZATION: See how the first convolution layer responds to images
# =========================================================
def plot_filtered_images(images, filters):
    images = torch.cat(
        [i.unsqueeze(0) for i in images],
        dim=0
    ).cpu()

    filters = filters.cpu()

    n_images = images.shape[0]
    n_filters = filters.shape[0]

    # Apply all Conv1 filters to the input images
    filtered_images = F.conv2d(images, filters)

    fig = plt.figure(figsize=(20, 10))

    for i in range(n_images):
        # Show original image
        ax = fig.add_subplot(
            n_images,
            n_filters + 1,
            i + 1 + (i * n_filters)
        )

        ax.imshow(images[i].squeeze(0), cmap='bone')
        ax.set_title('Original')
        ax.axis('off')

        # Show the output of each filter
        for j in range(n_filters):
            image = filtered_images[i][j]

            ax = fig.add_subplot(
                n_images,
                n_filters + 1,
                i + 1 + (i * n_filters) + j + 1
            )

            ax.imshow(image.numpy(), cmap='bone')
            ax.set_title(f'Filter {j + 1}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

N_IMAGES = 5

images = [
    image
    for image, label in [test_data[i] for i in range(N_IMAGES)]
]

# Conv1 has 6 learned filters
filters = lenet_model.conv1.weight.data

plot_filtered_images(images, filters)

# =========================================================
# 15. INFERENCE: Load the saved model and predict a new external image
# =========================================================
def load_model(model_path, num_classes=10):
    model = LeNetClassifier(num_classes)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    return model


def inference(img_path, model):
    # Load the external image
    image = Image.open(img_path)

    # Crop the image into a square if needed
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)

    # Apply the same preprocessing used for MNIST images
    img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    img_new = img_transform(image)

    # [1, 28, 28] -> [1, 1, 28, 28]
    # Add a batch dimension for the model
    img_new = img_new.expand(1, 1, 28, 28)
    #or
    # img_new = img_new.unsqueeze(0).to(device)

    # Predict without calculating gradients
    with torch.no_grad():
        predictions = model(img_new)

    # Convert logits into probabilities
    probabilities = nn.Softmax(dim=1)(predictions)

    # Get the most likely class and its probability
    p_max, yhat = torch.max(probabilities, dim=1)

    return p_max.item(), yhat.item()
