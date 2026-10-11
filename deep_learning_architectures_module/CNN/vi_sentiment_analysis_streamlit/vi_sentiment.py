# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
from collections import Counter
from pathlib import Path
import os
import re
import string
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from langid.langid import LanguageIdentifier, model as langid_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# =========================================================
# 2. CONFIGURATION
# =========================================================
FOLDER_PATHS = {
    "train": "./data/data_train/train",
    "valid": "./data/data_train/test",
    "test": "./data/data_test/test",
}

VOCAB_SIZE = 10_000
BATCH_SIZE = 128
NUM_EPOCHS = 10

EMBEDDING_DIM = 100
KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 100
NUM_CLASSES = 2

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "text_cnn_model.pt"

# Select the available accelerator
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# =========================================================
# 3. LOAD RAW TEXT FILES INTO A DATAFRAME
# =========================================================
def load_data_from_path(folder_path):
    examples = []

    # Each subfolder represents one sentiment class: neg or pos
    for label in os.listdir(folder_path):
        full_path = os.path.join(folder_path, label)

        # Skip invalid entries that are not folders
        if not os.path.isdir(full_path):
            continue

        # Convert folder names into numeric labels
        if label == "neg":
            label = 0
        elif label == "pos":
            label = 1
        else:
            continue

        # Read every text file inside the current class folder
        for file_name in os.listdir(full_path):
            file_path = os.path.join(full_path, file_name)

            # Read the full review content
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Merge multiple lines into one sentence string
            sentence = " ".join(lines)

            examples.append({
                'sentence': sentence,
                'label': label
            })

    return pd.DataFrame(examples)

train_df = load_data_from_path(FOLDER_PATHS["train"])
valid_df = load_data_from_path(FOLDER_PATHS["valid"])
test_df = load_data_from_path(FOLDER_PATHS["test"])

print("Raw train size:", len(train_df))
print("Raw validation size:", len(valid_df))
print("Raw test size:", len(test_df))


# =========================================================
# 4. LANGUAGE IDENTIFICATION
# =========================================================
def identifiy_vn(df):
    # Create a language identification model
    identifier = LanguageIdentifier.from_modelstring(
        langid_model,
        norm_probs=True,
    )

    not_vi_idx = set()
    THRESHOLD = 0.9

    for idx, row in df.iterrows():
        score = identifier.classify(row['sentence'])

        if score[0] != "vi" or (score[0] == "vi" and score[1] <= THRESHOLD):
            not_vi_idx.add(idx)

    vi_df = df[~df.index.isin(not_vi_idx)].copy()
    not_vi_df = df[df.index.isin(not_vi_idx)].copy()

    return vi_df, not_vi_df

# Filter all splits consistently
train_df_vi, train_df_not_vi = identifiy_vn(train_df)
valid_df_vi, valid_df_not_vi = identifiy_vn(valid_df)
test_df_vi, test_df_not_vi = identifiy_vn(test_df)

print("\nVietnamese train size:", len(train_df_vi))
print("Vietnamese validation size:", len(valid_df_vi))
print("Vietnamese test size:", len(test_df_vi))


# =========================================================
# 5. TEXT PREPROCESSING
# =========================================================
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    # Remove punctuation and digits
    replace_chars = list(string.punctuation + string.digits)

    for char in replace_chars:
        text = text.replace(char, " ")

    # Remove emojis and special symbols
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r" ", text)

    # Normalize repeated whitespace and lowercase the text
    text = " ".join(text.split())

    return text.lower()

# Create a cleaned-text column for every split
for dataframe in [train_df, valid_df, test_df]:
    dataframe["clean_sentence"] = dataframe["sentence"].apply(
        preprocess_text
    )

print("\nOriginal text:")
print(train_df_vi.iloc[0]["sentence"])

print("\nCleaned text:")
print(train_df_vi.iloc[0]["clean_sentence"])


# =========================================================
# 6. TOKENIZATION
# =========================================================
def tokenize(text):
    # A simple word-level tokenizer for the TextCNN baseline
    return text.split()


# =========================================================
# 7. BUILD VOCABULARY FROM TRAINING DATA ONLY
# =========================================================
word_counter = Counter()

for sentence in train_df["clean_sentence"]:
    word_counter.update(tokenize(sentence))

vocab = {
    "<pad>": 0,
    "<unk>": 1,
}

for word, _ in word_counter.most_common(VOCAB_SIZE - len(vocab)):
    vocab[word] = len(vocab)

PAD_IDX = vocab["<pad>"]
UNK_IDX = vocab["<unk>"]

print("Vocabulary size:", len(vocab))

# =========================================================
# 8. NUMERICALIZATION
# =========================================================
def encode_sentence(sentence):
    tokens = tokenize(sentence)

    token_ids = [
        vocab.get(token, UNK_IDX)
        for token in tokens
    ]

    # Avoid empty sequences after preprocessing
    if len(token_ids) == 0:
        token_ids = [UNK_IDX]

    return token_ids

# =========================================================
# 9. CREATE A CUSTOM PYTORCH DATASET
# =========================================================
class SentimentDataset(Dataset):
    def __init__(self, dataframe):
        self.sentences = dataframe["clean_sentence"].tolist()
        self.labels = dataframe["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded_sentence = encode_sentence(
            self.sentences[index]
        )

        label = self.labels[index]

        return (
            torch.tensor(encoded_sentence, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

train_dataset = SentimentDataset(train_df)
valid_dataset = SentimentDataset(valid_df)
test_dataset = SentimentDataset(test_df)

# =========================================================
# 10. COLLATE FUNCTION: Pad variable-length sentences inside each batch
# =========================================================
def collate_batch(batch):
    sentences, labels = zip(*batch)

    # Output shape: [batch_size, max_sequence_length]
    padded_sentences = pad_sequence(
        sentences,
        batch_first=True,
        padding_value=PAD_IDX,
    )

    # Ensure that all convolution kernels can run
    minimum_length = max(KERNEL_SIZES)

    if padded_sentences.size(1) < minimum_length:
        padding_length = minimum_length - padded_sentences.size(1)

        padded_sentences = F.pad(
            padded_sentences,
            pad=(0, padding_length),
            value=PAD_IDX,
        )

    labels = torch.stack(labels)

    return padded_sentences, labels

# =========================================================
# 11. CREATE DATALOADERS
# =========================================================
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
)

# =========================================================
# 12. DATALOADER SANITY CHECK
# =========================================================
inputs, labels = next(iter(train_dataloader))

print("Input shape:", inputs.shape)
print("Label shape:", labels.shape)

# Input shape: [batch_size, max_sequence_length]
# Label shape: [batch_size]


# =========================================================
# 13. BUILD THE TEXTCNN MODEL
# =========================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes,
                 num_filters, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1
            )
            for kernel_size in kernel_sizes
        ])
        
        self.fc = nn.Linear(
            len(kernel_sizes) * num_filters,
            num_classes
        )

    def forward(self, x):
        # x: [batch_size, sequence_length]

        # Shape: [batch_size, sequence_length, embedding_dim]
        outputs = self.embedding(x)

        # Shape: [batch_size, embedding_dim, sequence_length]
        outputs = outputs.transpose(1, 2)

        # Apply Conv1D and ReLU for each kernel size
        outputs = [
            F.relu(conv(outputs))
            for conv in self.convs
        ]

        # Apply global max pooling to every feature map
        outputs = [
            F.max_pool1d(
                feature_map,
                kernel_size=feature_map.size(-1),
            ).squeeze(-1)
            for feature_map in outputs
        ]

        # Concatenate features from kernel sizes 3, 4, and 5
        outputs = torch.cat(outputs, dim=1)

        # Output shape: [batch_size, num_classes]
        outputs = self.fc(outputs)

        return outputs


# =========================================================
# 14. TRAINING FUNCTION
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

        predictions = model(inputs)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_acc, epoch_loss


# =========================================================
# 15. EVALUATION FUNCTION
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

        predictions = model(inputs)
        loss = criterion(predictions, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    return epoch_acc, epoch_loss

# =========================================================
# 16. INITIALIZE MODEL AND TRAINING COMPONENTS
# =========================================================
model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    kernel_sizes=KERNEL_SIZES,
    num_filters=NUM_FILTERS,
    num_classes=NUM_CLASSES,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
)

print("Using device:", DEVICE)
print("Model output shape:", model(inputs.to(DEVICE)).shape)


# =========================================================
# 17. TRAINING LOOP
# =========================================================
MODEL_DIR.mkdir(parents=True, exist_ok=True)

history = {
    "train_acc": [],
    "train_loss": [],
    "valid_acc": [],
    "valid_loss": [],
}

best_valid_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()

    train_acc, train_loss = train_one_epoch(
        model,
        train_dataloader,
        criterion,
        optimizer,
        DEVICE,
    )

    valid_acc, valid_loss = evaluate(
        model,
        valid_dataloader,
        criterion,
        DEVICE,
    )

    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["valid_acc"].append(valid_acc)
    history["valid_loss"].append(valid_loss)

    # Save the best model based on validation loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Time: {elapsed:.1f}s | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2%} | "
        f"Valid Loss: {valid_loss:.4f} | "
        f"Valid Acc: {valid_acc:.2%}"
    )


# =========================================================
# 18. LOAD THE BEST MODEL AND TEST IT
# =========================================================
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

test_acc, test_loss = evaluate(
    model,
    test_dataloader,
    criterion,
    DEVICE,
)

print("\nFinal test result")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")

# =========================================================
# 19. PLOT TRAINING HISTORY
# =========================================================
def plot_history(history):
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        epochs,
        history["train_acc"],
        label="Training",
    )
    axes[0].plot(
        epochs,
        history["valid_acc"],
        label="Validation",
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        history["train_loss"],
        label="Training",
    )
    axes[1].plot(
        epochs,
        history["valid_loss"],
        label="Validation",
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_history(history)


# =========================================================
# 20. INFERENCE ON A NEW SENTENCE
# =========================================================
def predict_sentiment(sentence, model):
    model.eval()

    cleaned_sentence = preprocess_text(sentence)
    token_ids = encode_sentence(cleaned_sentence)

    # Shape: [1, sequence_length]
    inputs = torch.tensor(
        token_ids,
        dtype=torch.long,
    ).unsqueeze(0)

    minimum_length = max(KERNEL_SIZES)
    if inputs.size(1) < minimum_length:
        inputs = F.pad(
            inputs,
            pad=(0, minimum_length - inputs.size(1)),
            value=PAD_IDX,
        )

    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_label = torch.max(
        probabilities,
        dim=1,
    )

    sentiment = (
        "positive"
        if predicted_label.item() == 1
        else "negative"
    )

    return sentiment, confidence.item()