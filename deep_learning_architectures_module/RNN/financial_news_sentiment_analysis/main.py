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
def build_vocabulary(texts):
    word_counter = Counter()

    for sentence in texts:
        tokens = sentence.split()

        word_counter.update(tokens)

    vocab = []

    for word, _ in word_counter.items():
        vocab.append(word)

    vocab.append('UNK')
    vocab.append('PAD')

    word_to_idx = {
        word: idx
        for idx, word in enumerate(vocab)
    }

    vocab_size = len(vocab)

    unk_idx = word_to_idx['UNK']
    pad_idx = word_to_idx['PAD']

    return vocab, word_to_idx, vocab_size, unk_idx, pad_idx

vocab, word_to_idx, vocab_size, UNK_IDX, PAD_IDX = build_vocabulary(
    X_train
)

print('\nVocabulary size:', vocab_size)
print('UNK index:', UNK_IDX)
print('PAD index:', PAD_IDX)


# TRANSFORM TEXT INTO TOKEN IDS WITH PADDING/TRUNCATION
def transform(text, word_to_idx, max_seq_len):
    tokens = []

    for word in text.split():
        if word in word_to_idx:
            word_id = word_to_idx[word]
        else:
            word_id = word_to_idx['UNK']

        tokens.append(word_id)

    if len(tokens) < max_seq_len:
        tokens += [word_to_idx['PAD']] * (max_seq_len - len(tokens))

    elif len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]

    return tokens

# =========================================================
# 10. CREATE CUSTOM PYTORCH DATASET
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

        text = torch.tensor(text, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return text, label

# =========================================================
# 11. CREATE DATASETS AND DATALOADERS
# =========================================================
train_dataset = FinancialNews(
    X_train, y_train,
    word_to_idx=word_to_idx,
    max_seq_len=MAX_SEQ_LEN,
    transform=transform
)

val_dataset = FinancialNews(
    X_val, y_val,
    word_to_idx=word_to_idx,
    max_seq_len=MAX_SEQ_LEN,
    transform=transform
)

test_dataset = FinancialNews(
    X_test, y_test,
    word_to_idx=word_to_idx,
    max_seq_len=MAX_SEQ_LEN,
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

# Expected input shape: [batch_size, max_seq_len]
# Expected label shape: [batch_size]


# =========================================================
# 12. BUILD THE RNN SENTIMENT CLASSIFIER
# =========================================================
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, n_classes, dropout_prob):
        super(SentimentClassifier, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_IDX
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
        # [batch_size, max_seq_len]
        x = self.embedding(x)

        # [batch_size, max_seq_len, embedding_dim]
        x, hn = self.rnn(x)

        # Select the output at the final time step
        x = x[:, :-1, :]

        # [batch_size, hidden_size]
        x = self.norm(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        # [batch_size, n_classes]
        return x

# =========================================================
# 10. INITIALIZE MODEL
# =========================================================
model = SentimentClassifier(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    n_classes=n_classes,
    dropout_prob=DROPOUT_PROB
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print('\nUsing device:', DEVICE)
sample_outputs = model(inputs.to(DEVICE))
print('Model output shape:', sample_outputs.shape)


# =========================================================
# 11. TRAINING FUNCTION
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
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

        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc


# =========================================================
# 12. VALIDATION / EVALUATION FUNCTION
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

        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc


# =========================================================
# 13. TRAINING LOOP AND BEST MODEL CHECKPOINT
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
                'word_to_idx': word_to_idx,
                'idx_to_class': idx_to_class,
                'vocab_size': vocab_size,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_size': HIDDEN_SIZE,
                'n_layers': N_LAYERS,
                'n_classes': n_classes,
                'dropout_prob': DROPOUT_PROB,
                'max_seq_len': MAX_SEQ_LEN
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
# 14. PLOT TRAINING HISTORY
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

plot_history(train_losses, val_losses, train_accs, val_accs)

# =========================================================
# 15. LOAD THE BEST MODEL
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
# 16. FINAL TEST EVALUATION
# =========================================================
test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

print('\nFinal evaluation on test dataset')
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.2%}')

# =========================================================
# 17. INFERENCE ON A NEW FINANCIAL NEWS SENTENCE
# =========================================================
def predict_sentiment(text, model):
    # Apply the same preprocessing used during training
    text = text_normalize(text)

    # Convert text into token IDs
    text = transform(text, word_to_idx, MAX_SEQ_LEN)

    # Add a batch dimension
    text = torch.tensor(
        text,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    # Predict without gradient calculation
    with torch.no_grad():
        outputs = model(text)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = idx_to_class[
        predicted_index.item()
    ]

    return predicted_class, confidence.item()


