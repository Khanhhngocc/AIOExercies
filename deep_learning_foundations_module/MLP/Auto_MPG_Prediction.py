import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2 READ DATASET
dataset_path = 'datasets/[Data]_Ex1_Auto_MPG_data.csv'
dataset = pd.read_csv(dataset_path)

# 3 PREPROCESSING
X, y = dataset.drop(columns="MPG").values, dataset["MPG"].values

val_size = 0.2
test_size = 0.125
is_shuffle = True
random_state = 59

# 3.1 SPLIT DATA
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)

# 3.2 NORMALIZATION

normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_val = normalizer.transform(X_val)
X_test = normalizer.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 4 CREATE PYTORCH DATALOADER
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=True)


# 5 Build MLP network
class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        out = self.output(x)
        return out.squeeze(1)

input_dims = X_train.shape[1]
output_dims = 1
hidden_dims = 64
lr = 1e-2

model = MLP(input_dims, hidden_dims, output_dims)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 6 Training
def r_squared(y_true, y_pred):
    y_true = torch.Tensor(y_true)
    y_pred = torch.Tensor(y_pred)
    mean_true = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_true) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

epochs = 100
train_losses = []
val_losses = []
train_r2 = []
val_r2 = []

for epoch in range(epochs):
    # ==================== TRAIN ====================
    model.train()

    train_loss = 0.0
    train_target = []
    train_predict = []

    for X_samples, y_samples in train_loader:
        optimizer.zero_grad()

        outputs = model(X_samples)
        loss = criterion(outputs, y_samples)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_samples.size(0)

        train_predict.append(outputs.detach().cpu())
        train_target.append(y_samples.detach().cpu())

    train_predict = torch.cat(train_predict)
    train_target = torch.cat(train_target)

    train_loss /= len(train_loader.dataset)

    train_losses.append(train_loss)
    train_r2.append(r_squared(train_target, train_predict))


    # ================= VALIDATION ==================
    model.eval()
    val_loss = 0.0
    val_target = []
    val_predict = []

    with torch.no_grad():
        for X_samples, y_samples in val_loader:
            outputs = model(X_samples)

            loss = criterion(outputs, y_samples)

            val_loss += loss.item() * X_samples.size(0)

            val_predict.extend(outputs.cpu().tolist())
            val_target.extend(y_samples.cpu().tolist())

    val_loss /= len(val_loader.dataset)

    val_losses.append(val_loss)
    val_r2.append(r_squared(val_target, val_predict))

    print(
        f"EPOCH {epoch + 1}: "
        f"Training loss: {train_loss:.3f} | "
        f"Validation loss: {val_loss:.3f} | "
        f"Train R²: {train_r2[-1]:.3f} | "
        f"Validation R²: {val_r2[-1]:.3f}"
    )

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].plot(train_losses, color='green')
ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
ax[0, 0].set_title('Training Loss')

ax[0, 1].plot(val_losses, color='orange')
ax[0, 1].set(xlabel='Epoch', ylabel='Loss')
ax[0, 1].set_title('Validation Loss')

ax[1, 0].plot(train_r2, color='green')
ax[1, 0].set(xlabel='Epoch', ylabel='R2')
ax[1, 0].set_title('Training R2')

ax[1, 1].plot(val_r2, color='orange')
ax[1, 1].set(xlabel='Epoch', ylabel='R2')
ax[1, 1].set_title('Validation R2')

plt.show()


# EVALUATION
model.eval()
with torch.no_grad():
    y_hat = model(X_val)
    val_set_r2 = r_squared(y_hat, y_val)
    print('Evaluation on validation set:')
    print(f'R2: {val_set_r2}')

model.eval()
with torch.no_grad():
    y_hat = model(X_test)
    test_set_r2 = r_squared(y_hat, y_test)
    print('Evaluation on test set:')
    print(f'R2: {test_set_r2}')