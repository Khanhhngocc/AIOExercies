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
# 10. CREATE SLIDING WINDOWS
# =========================================================
def slicing_window(df, df_start_idx, df_end_idx, input_size, label_size, offset):
    features = []
    labels = []

    window_size = input_size + offset

    if df_end_idx == None:
     df_end_idx = len(df)

    for idx in range(df_start_idx, df_end_idx):
        features_end_idx = idx + input_size
        label_start_idx = idx + window_size - label_size

        feature = df[idx:features_end_idx]
        label = df[label_start_idx:(idx+window_size)]

        features.append(feature)
        labels.append(label)

    feature = np.expand_dims(np.array(features), -1)
    labels = np.array(labels)

    return features, labels



class WeatherForecastor(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers, dropout_prob):
        super(WeatherForecastor, self)
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, hn = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x