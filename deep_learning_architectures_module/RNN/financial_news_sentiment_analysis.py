# =========================================================
# 1. PROBLEM DEFINITION
#    Classify financial news into positive, negative,
#    or neutral sentiment classes.
# =========================================================



# =========================================================
# 2. IMPORT LIBRARIES AND SET RANDOM SEED
# =========================================================
from collections import Counter
from pathlib import Path
import random
import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import unidecode

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 3. CONFIGURATION
# =========================================================
SEED = 1

DATASET_PATH = 'dataset/all-data.csv'
MODEL_DIR = Path('./model')
MODEL_PATH = MODEL_DIR / 'financial_news_rnn_best.pt'

MAX_SEQ_LEN = 32
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128

EMBEDDING_DIM = 64
HIDDEN_SIZE = 64
N_LAYERS = 2
DROPOUT_PROB = 0.2

LEARNING_RATE = 1e-4
EPOCHS = 50

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# SET RANDOM SEED
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# 4. DATA LOADING AND UNDERSTANDING / EDA
# =========================================================
headers = ['sentiment', 'content']

df = pd.read_csv(
    DATASET_PATH,
    names=headers,
    encoding='ISO-8859-1'
)

print('Dataset shape:', df.shape)

print('\nFirst five rows:')
print(df.head())

print('\nMissing values:')
print(df.isnull().sum())

print('\nDuplicate rows:', df.duplicated().sum())

print('\nSentiment distribution:')
print(df['sentiment'].value_counts())

plt.figure(figsize=(7, 4))

df['sentiment'].value_counts().plot(
    kind='bar',
    color=['#6baed6', '#74c476', '#fd8d3c']
)

plt.title('Sentiment Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# =========================================================
# 5. DATA CLEANING
# =========================================================
df = df.dropna()

df = df.drop_duplicates()

df = df.reset_index(drop=True)


# =========================================================
# 6. LABEL ENCODING
# =========================================================
classes = {
    class_name: idx
    for idx, class_name in enumerate(
        df['sentiment'].unique().toList()
    )
}

idx_to_class = {
    idx: class_name
    for class_name, idx in classes.items()
}

df['sentiment'] = df['sentiment'].apply(
    lambda x: classes[x]
)

n_classes = len(classes)

print('\nClass mapping:')
print(classes)

# =========================================================
# 7. TEXT PREPROCESSING
# =========================================================
nltk.download('stopwords')
english_stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def text_normalize(text):
    text = text.lower()

    # Remove accents from characters
    text = unidecode.unidecode(text)

    # Remove leading and trailing whitespace
    text = text.strip()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove stop words
    text = ' '.join([
        word
        for word in text.split(' ')
        if word not in english_stop_words
    ])

    # Apply stemming
    text = ' '.join([
        stemmer.stem(word)
        for word in text.split(' ')
    ])

    return text

# Apply text normalization to every news content
df['content'] = df['content'].apply(
    lambda x: text_normalize(x)
)

df = df[df['content'].str.len() > 0]

df = df.reset_index(drop=True)

print('\nNormalized example:')
print(df['content'].iloc[0])

# =========================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# =========================================================
val_size = 0.2
test_size = 0.125
is_shuffle = True

texts = df['content'].toList()
labels = df['sentiment'].toList()

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels,
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
# 9. FEATURE REPRESENTATION:
#    Build vocabulary and convert text into token IDs
# =========================================================

# BUILD VOCABULARY FROM TRAINING DATA ONLY
word_counter = Counter()

vocab = []

for sentence in df['content'].tolist():
    tokens = sentence.split()

    for token in tokens:
        if token not in vocab:
            vocab.append(token)

vocab.append('UNK')
vocab.append('PAD')

word_to_idx = {
    word: idx
    for idx, word in enumerate(vocab)
}

vocab_size = len(vocab)

# TRANSFORM TEXT INTO TOKEN IDS WITH PADDING/TRUNCATION
def transform(text, word_to_idx, max_seq_len):
    tokens = []

    for w in text.split():
        try:
            w_ids = word_to_idx[w]
        except:
            w_ids = word_to_idx['UNK']

        tokens.append(w_ids)

    if len(tokens) < max_seq_len:
        tokens += [word_to_idx['PAD']] * (max_seq_len - len(tokens))

    elif len(tokens) > max_seq_len:
        tokens = tokens[: max_seq_len]

    return tokens

# =========================================================
# 7. CREATE CUSTOM PYTORCH DATASET
# =========================================================
class FinancialNews(Dataset):
    def __init__(self, X, y, word_to_idx, max_seq_len, transform=None):
        self.texts = X
        self.labels = y
        self.word_to_idx = word_to_idx
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.transform:
            text = self.transform(
                text,
                self.word_to_idx,
                self.max_seq_len
            )

        text = torch.tensor(text)

        return text, label

# =========================================================
# 8. CREATE DATASETS AND DATALOADERS
# =========================================================
max_seq_len = 32

train_dataset = FinancialNews(
    X_train, y_train,
    word_to_idx=word_to_idx,
    max_seq_len=max_seq_len,
    transform=transform
)

val_dataset = FinancialNews(
    X_val, y_val,
    word_to_idx=word_to_idx,
    max_seq_len=max_seq_len,
    transform=transform
)

test_dataset = FinancialNews(
    X_test, y_test,
    word_to_idx=word_to_idx,
    max_seq_len=max_seq_len,
    transform=transform
)

train_batch_size  = 128
test_batch_size = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=test_batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False
)


# =========================================================
# 9. BUILD THE RNN SENTIMENT CLASSIFIER
# =========================================================
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, n_classes, dropout_prob):
        super(SentimentClassifier, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            n_layers,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.embedding(x)

        x, hn = self.rnn(x)

        x = x[:, :-1, :]

        x = self.norm(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x

# =========================================================
# 10. INITIALIZE MODEL
# =========================================================
n_classes = len(list(classes.keys()))

embedding_dim = 64
hidden_size = 64
n_layers = 2
dropout_prob = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentimentClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    n_layers=n_layers,
    n_classes=n_classes,
    dropout_prob=dropout_prob
).to(device)








