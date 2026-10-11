# =========================================================
# 1. PROBLEM DEFINITION
#    Predict the temperature of the next hour using
#    the temperatures from the previous 6 hours.
# =========================================================


# =========================================================
# 2. IMPORT LIBRARIES
# =========================================================
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# =========================================================
# 3. CONFIGURATION
# =========================================================
SEED = 1

DATASET_PATH = 'dataset/weatherHistory.csv'
MODEL_DIR = Path('./model')
MODEL_PATH = MODEL_DIR / 'weather_rnn_best.pt'

INPUT_SIZE = 6
LABEL_SIZE = 1
OFFSET = 1

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128

EMBEDDING_DIM = 1
HIDDEN_SIZE = 8
N_LAYERS = 3
DROPOUT_PROB = 0.2

LEARNING_RATE = 1e-3
EPOCHS = 50

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
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# DATA LOADING
df = pd.read_csv(DATASET_PATH)
univariate_df = df[['Formatted Date', 'Temperature (C)']].copy()


# =========================================================
# 5. DATA UNDERSTANDING / EDA
# =========================================================
print('Dataset shape:', univariate_df.shape)

print('\nFirst five rows:')
print(univariate_df.head())

print('\nMissing values:')
print(univariate_df.isnull().sum())

print('\nTemperature statistics:')
print(univariate_df['Temperature (C)'].describe())


# =========================================================
# 6. DATA CLEANING AND PREPROCESSING
# =========================================================
univariate_df = univariate_df.dropna()

univariate_df['Formatted Date'] = pd.to_datetime(
    univariate_df['Formatted Date'],
    utc=True
)

univariate_df = univariate_df.sort_values(
    by='Formatted Date'
)

univariate_df = univariate_df.reset_index(drop=True)

temperatures = univariate_df['Temperature (C)'].to_numpy(
    dtype=np.float32
)

timestamps = univariate_df['Formatted Date'].to_numpy()

# =========================================================
# 7. VISUALIZE THE TEMPERATURE TIME SERIES
# =========================================================
plt.figure(figsize=(20, 6))

plt.plot(
    timestamps[:500],
    temperatures[:500],
    color='steelblue'
)

plt.title('Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# =========================================================
# 8. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# =========================================================
dataset_length = len(temperatures)

train_end_idx = int(TRAIN_RATIO * dataset_length)

val_end_idx = int(
    (TRAIN_RATIO + VAL_RATIO) * dataset_length
)

train_temperatures = temperatures[:train_end_idx]

val_temperatures = temperatures[
    train_end_idx:val_end_idx
]

test_temperatures = temperatures[val_end_idx:]

print('Train size:', len(train_temperatures))
print('Validation size:', len(val_temperatures))
print('Test size:', len(test_temperatures))


# =========================================================
# 9. NORMALIZE TEMPERATURES USING TRAINING STATISTICS
# =========================================================
train_mean = train_temperatures.mean()

train_std = train_temperatures.std()

train_temperatures = (
    train_temperatures - train_mean
) / train_std

val_temperatures = (
    val_temperatures - train_mean
) / train_std

test_temperatures = (
    test_temperatures - train_mean
) / train_std


# =========================================================
# 10. CREATE SLIDING WINDOWS
# =========================================================
def slicing_window(values, input_size, label_size, offset):
    features = []
    labels = []

    window_size = input_size + offset

    end_idx = len(values) - window_size + 1

    for idx in range(end_idx):
        features_end_idx = idx + input_size
        label_start_idx = idx + window_size - label_size

        label_end_idx = idx + window_size

        feature = values[idx:features_end_idx]

        label = values[label_start_idx:label_end_idx]

        features.append(feature)
        labels.append(label)

    features = np.expand_dims(np.array(features), axis=-1)
    labels = np.array(labels)

    return features, labels

X_train, y_train = slicing_window(
    train_temperatures,
    input_size=INPUT_SIZE,
    label_size=LABEL_SIZE,
    offset=OFFSET
)

X_val, y_val = slicing_window(
    val_temperatures,
    input_size=INPUT_SIZE,
    label_size=LABEL_SIZE,
    offset=OFFSET
)

X_test, y_test = slicing_window(
    test_temperatures,
    input_size=INPUT_SIZE,
    label_size=LABEL_SIZE,
    offset=OFFSET
)

print('\nWindowed data shapes:')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_val:', X_val.shape)
print('y_val:', y_val.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)


# =========================================================
# 11. CREATE CUSTOM PYTORCH DATASET
# =========================================================
class WeatherForecast(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)

        X = torch.tensor(X, dtype=torch.float32)

        y = torch.tensor(y, dtype=torch.float32)

        return X, y

# =========================================================
# 12. CREATE DATASETS AND DATALOADERS
# =========================================================
train_dataset = WeatherForecast(X_train, y_train)

val_dataset = WeatherForecast(X_val, y_val)

test_dataset = WeatherForecast(X_test, y_test)

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

# Input shape: [batch_size, 6, 1]
# Label shape: [batch_size, 1]


# =========================================================
# 13. BUILD THE RNN FORECASTING MODEL
# =========================================================
class WeatherForecastor(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers, dropout_prob):
        super(WeatherForecastor, self)

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            n_layers,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Input shape: [batch_size, sequence_length, 1]
        x, hn = self.rnn(x)

        # Select the output of the final time step
        x = x[:, -1, :]

        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        # Output shape: [batch_size, 1]
        return x

# =========================================================
# 14. INITIALIZE MODEL
# =========================================================
model = WeatherForecastor(
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    dropout_prob=DROPOUT_PROB
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print('\nUsing device:', DEVICE)
sample_outputs = model(inputs.to(DEVICE))
print('Model output shape:', sample_outputs.shape)


# =========================================================
# 15. TRAINING FUNCTION
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

    epoch_loss = total_loss / total_samples

    return epoch_loss


# =========================================================
# 16. VALIDATION / EVALUATION FUNCTION
# =========================================================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

    epoch_loss = total_loss / total_samples

    return epoch_loss

# =========================================================
# 17. TRAINING LOOP AND BEST MODEL CHECKPOINT
# =========================================================
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    val_loss = evaluate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    # Save the model when validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'train_mean': train_mean,
                'train_std': train_std,
                'input_size': INPUT_SIZE,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_size': HIDDEN_SIZE,
                'n_layers': N_LAYERS,
                'dropout_prob': DROPOUT_PROB
            },
            MODEL_PATH
        )

    print(
        f'Epoch {epoch:02d}/{EPOCHS} | '
        f'Train Loss: {train_loss:.4f} | '
        f'Val Loss: {val_loss:.4f}'
    )


# =========================================================
# 18. PLOT TRAINING HISTORY
# =========================================================
def plot_history(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        train_losses,
        label='Training'
    )

    plt.plot(
        epochs,
        val_losses,
        label='Validation'
    )

    plt.title('Training and Validation Loss')

    plt.xlabel('Epoch')

    plt.ylabel('MSE Loss')

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()

plot_history(
    train_losses,
    val_losses
)


# =========================================================
# 19. LOAD THE BEST MODEL
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
# 20. FINAL TEST EVALUATION
# =========================================================
test_loss = evaluate(
    model,
    test_loader,
    criterion,
    DEVICE
)

print('\nFinal evaluation on test dataset')
print(f'Test MSE loss: {test_loss:.4f}')


# =========================================================
# 21. PLOT TRUE VALUES VS PREDICTIONS
# =========================================================
def plot_difference(y, pred):
    plt.figure(figsize=(20, 6))

    times = range(len(y))

    y_to_plot = y.flatten()

    pred_to_plot = pred.flatten()

    plt.plot(
        times,
        y_to_plot,
        color='steelblue',
        marker='o',
        label='True value'
    )

    plt.plot(
        times,
        pred_to_plot,
        color='orangered',
        marker='X',
        label='Prediction'
    )

    plt.title('Temperature Forecasting Results')

    plt.xlabel('Hour')

    plt.ylabel('Temperature (C)')

    plt.legend()

    plt.grid(alpha=0.3)

    plt.show()

inputs = torch.tensor(
    X_test[:100],
    dtype=torch.float32
).to(DEVICE)

model.eval()

with torch.no_grad():
    outputs = model(inputs).detach().cpu().numpy()

# Convert normalized values back to Celsius
y_true = y_test[:100] * train_std + train_mean

y_pred = outputs * train_std + train_mean

plot_difference(y_true, y_pred)


# =========================================================
# 22. INFERENCE:
#     Forecast the next temperature from 6 recent values
# =========================================================
def predict_next_temperature(recent_temperatures, model):
    # Check that exactly 6 temperature values are provided
    if len(recent_temperatures) != INPUT_SIZE:
        raise ValueError(
            f'Expected {INPUT_SIZE} temperature values.'
        )

    recent_temperatures = np.array(
        recent_temperatures,
        dtype=np.float32
    )

    # Apply the same normalization used during training
    normalized_temperatures = (
        recent_temperatures - train_mean
    ) / train_std

    # Shape: [6] -> [1, 6, 1]
    inputs = torch.tensor(
        normalized_temperatures,
        dtype=torch.float32
    )

    inputs = inputs.unsqueeze(0).unsqueeze(-1).to(DEVICE)

    with torch.no_grad():
        normalized_prediction = model(inputs)

    prediction = normalized_prediction.item()

    # Convert the prediction back to Celsius
    prediction = prediction * train_std + train_mean

    return prediction