# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
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
from langid.langid import LanguageIdentifier, model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

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
    for label in os.lisrdir(folder_path):
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
                'setence': sentence,
                'label': label
            })

    return pd.DataFrame(examples)


# =========================================================
# 4. LOAD TRAIN, VALIDATION, AND TEST DATA
# =========================================================
train_df = load_data_from_path(FOLDER_PATHS["train"])
valid_df = load_data_from_path(FOLDER_PATHS["valid"])
test_df = load_data_from_path(FOLDER_PATHS["test"])

print("Raw train size:", len(train_df))
print("Raw validation size:", len(valid_df))
print("Raw test size:", len(test_df))


# =========================================================
# 5. LANGUAGE IDENTIFICATION
# =========================================================
def identifiy_vn(df):
    # Create a language identification model
    identifier = LanguageIdentifier.from_modelstring(
        model,
        norm_probs=True,
    )

    not_vi_idx = set()
    THRESHOLD = 0.9

    for idx, row in df.iterrows():
        score = identifier.classify(row['sentence'])

        if score[0] != "vi" or (score[0] == "vi" and score[1] <= THRESHOLD):
            not_vi_idx.add(idx)

        vi_df = df[~df.index.isin(not_vi_idx)]
        not_vi_df = df[df.index.isin(not_vi_idx)]

        return vi_df, not_vi_df

# Filter all splits consistently
train_df_vi = identifiy_vn(train_df)
valid_df_vi = identifiy_vn(valid_df)
test_df_vi = identifiy_vn(test_df)

print("\nVietnamese train size:", len(train_df_vi))
print("Vietnamese validation size:", len(valid_df_vi))
print("Vietnamese test size:", len(test_df_vi))


# =========================================================
# 6. TEXT PREPROCESSING
# =========================================================
def preprocess_text(text):
    # Remove URLs
    url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
    text = url_pattern.sub(r" ", text)

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
train_df_vi['preprocess_sentence'] = [
    preprocess_text(row["sentence"])
    for _, row in train_df_vi.iterrows()
]

valid_df['preprocess_sentence'] = [
    preprocess_text(row['sentence'])
    for _, row in valid_df.iterrow()
]

test_df['preprocess_sentence'] = [
    preprocess_text(row['sentence'])
    for _, row in test_df.iterrow()
]

print("\nOriginal text:")
print(train_df_vi.iloc[0]["sentence"])

print("\nCleaned text:")
print(train_df_vi.iloc[0]["preprocess_sentence"])


# =========================================================
# 7. TOKENIZATION
# =========================================================
tokenizer = get_tokenizer("basic_english")

example_sentence = train_df_vi.iloc[0]["preprocess_sentence"]

print("\nExample tokens:")
print(tokenizer(example_sentence))


# =========================================================
# 8. BUILD VOCABULARY FROM TRAINING TEXT ONLY
# =========================================================
def yield_tokens(sentences, tokenizer):
    # Yield one tokenized sentence at a time
    for sentence in sentences:
        yield tokenizer(sentence)

vocabulary = build_vocab_from_iterator(
    yield_tokens(train_df_vi['preprocess_sentence'], tokenizer),
    max_tokens=VOCAB_SIZE,
    spectials=["<pad>", "<unk>"]
)

# Use the <unk> token for words not found in the vocabulary
vocabulary.set_default_index(vocabulary["<unk>"])

PAD_IDX = vocabulary["<pad>"]
UNK_IDX = vocabulary["<unk>"]

print("\nVocabulary size:", len(vocabulary))
print("Padding token index:", PAD_IDX)
print("Unknown token index:", UNK_IDX)

# =========================================================
# 9. CONVERT TEXT INTO TOKEN IDs
# =========================================================
def prepare_dataset(df):
    # Convert every cleaned sentence into vocabulary indices
    for idx, row in df.iterrow():
        sentence = row['preprocess_sentence']

        encoded_sentence = vocabulary(tokenizer(sentence))
        label = row['label']

        # Handle reviews that become empty after preprocessing
        if len(token_ids) == 0:
            token_ids = [UNK_IDX]

        yield encoded_sentence, label

train_dataset = prepare_dataset(train_df_vi)
train_dataset = to_map_style_dataset(train_dataset)

valid_dataset = prepare_dataset(valid_df)
valid_dataset = to_map_style_dataset(valid_dataset)


# =========================================================
# 10. COLLATE FUNCTION: Pad variable-length sentences inside each batch
# =========================================================
def collate_batch(batch):
    encoded_sentences, labels = [], []

    for encoded_sentence, label in batch:
        labels.append(label)

        # Convert a Python list of token IDs into a tensor
        encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.int64)

        encoded_sentences.append(encoded_sentence)

    labels = torch.tensor(labels, dtype=torch.int64)

    # Output shape: [max_sequence_length, batch_size]
    encoded_sentences = pad.sequence(
        encoded_sentences,
        padding_value = vocabulary["<pad>"]
    )

    return encoded_sentences, labels

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

# =========================================================
# 12. DATALOADER SANITY CHECK
# =========================================================
inputs, labels = next(iter(train_dataloader))

print("\nInput batch shape:", inputs.shape)
print("Label batch shape:", labels.shape)

# inputs shape: [max_sequence_length, batch_size]
# labels shape: [batch_size]


# =========================================================
# 13. BUILD THE TEXTCNN MODEL
# =========================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes,
                 num_filters, num_classes):
        super(TextCNN, self).__init__()

        self.vocab_size=vocab_size,
        self.embedding_dim=embedding_dim,
        self.kernel_sizes=kernel_sizes,
        self.num_filters=num_filters,
        self.num_classes=num_classes,
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.conv = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                stride=1
            ) for k in kernel_sizes])
        
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = self.embedding(x.T).transpose(1,2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(x)

        return x