from attr.validators import max_len
from datasets import load_dataset
import re
import string
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from datasets import load_dataset

import time

# ============================================================
# 1. SETUP
# ============================================================
random_state = 59
np.random.seed(random_state)
torch.manual_seed(random_state)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

VOCAB_SIZE = 10000
MAX_LENGTH = 100

EMBED_DIM = 200
NUM_LAYERS = 2
NUM_HEADS = 4
FF_DIM = 128
DROPOUT = 0.1

NUM_CLASSES = 2

BATCH_SIZE = 128
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "transformer_text_cls_best.pt"

# ============================================================
# 2. LOAD DATASET
# ============================================================
dataset = load_dataset("thainq107/ntc-scv")

print(dataset)
print("\nTrain example:")
print(dataset["train"][0])


# ============================================================
# 3. PREPROCESSING
# ============================================================
def preprocess_text(text):
    url_pattern = re.compile(r'https?://\ s+\ wwww\.\ s +')
    text = url_pattern.sub(r" ", text)

    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, "")

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    text = emoji_pattern.sub(r" ", text)
    text = " ".join(text.split())
    text = text.lower()

    return text

if "preprocessed_sentence" not in dataset["train"].column_names:
    def preprocess_batch(batch):
        return {
            "preprocessed_sentence": [
                preprocess_text(text)
                for text in batch["sentence"]
            ]
        }

    dataset = dataset.map(
        preprocess_batch,
        batched=True,
        des="Preprocessing text"
    )

TEXT_COLUMN = "preprocessed_sentence"
print("\nProcessed example:")
print(dataset["train"][0][TEXT_COLUMN])

# ============================================================
# 4. TOKENIZATION + VOCABULARY
# ============================================================
tokenizer = Tokenizer( WordLevel(unk_token="<unk>") )

tokenizer.pre_tokenizer = Whitespace()

trainer = WordLevelTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[
        "<pad>",
        "<unk>"
    ]
)

def batch_iterator(hf_dataset, batch_size=1000):
    for start_idx in range(0, len(hf_dataset), batch_size):
        end_idx = start_idx + batch_size

        batch = hf_dataset[start_idx: end_idx]

        yield batch[TEXT_COLUMN]

tokenizer.train_from_iterator(
    batch_iterator(dataset["train"]),
    trainer=trainer
)

PAD_ID = tokenizer.token_to_id("<pad>")
UNK_ID = tokenizer.token_to_id("<unk>")

VOCAB_SIZE_ACTUAL = tokenizer.get_vocab_size()

print("\nVocabulary size:", VOCAB_SIZE_ACTUAL)
print("PAD ID:", PAD_ID)
print("UNK ID:", UNK_ID)


# Test tokenizer
example_text = dataset["train"][0][TEXT_COLUMN]

encoding = tokenizer.encode(example_text)

print("\nText:")
print(example_text)

print("\nTokens:")
print(encoding.tokens)

print("\nToken IDs:")
print(encoding.ids)


# ============================================================
# 5. TOKENIZE DATASET
# ============================================================
def tokenize_batch(batch):
    encodings = tokenizer.encode_batch(
        batch[TEXT_COLUMN]
    )

    return {
        "input_ids": [
            encoding.ids
            for encoding in encodings
        ]
    }

tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing dataset"
)

print("\nTokenized example:")
print(tokenized_dataset["train"][0]["input_ids"][:20])


# ============================================================
# 6. COLLATE FUNCTION
# ============================================================
def collate_batch(batch):

    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []

    for item in batch:
        input_ids = item["input_ids"][:MAX_LENGTH]

        attention_mask = [1] * len(input_ids)

        padding_length = MAX_LENGTH - len(input_ids)

        input_ids = input_ids + [PAD_ID] * padding_length

        attention_mask = attention_mask + [0] * padding_length

        batch_input_ids.append(input_ids)
        batch_attention_masks.append(attention_mask)
        batch_labels.append(item["label"])

    input_ids = torch.tensor(
        batch_input_ids,
        dtype=torch.long
    )

    attention_mask = torch.tensor(
        batch_attention_masks,
        dtype=torch.long
    )

    labels = torch.tensor(
        batch_labels,
        dtype=torch.long
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ============================================================
# 7. DATALOADER
# ============================================================
train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

valid_dataloader = DataLoader(
    tokenized_dataset["valid"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

test_dataloader = DataLoader(
    tokenized_dataset["test"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

batch = next(iter(train_dataloader))

print("\nInput IDs shape:")
print(batch["input_ids"].shape)

print("\nAttention mask shape:")
print(batch["attention_mask"].shape)

print("\nLabels shape:")
print(batch["labels"].shape)

# ============================================================
# 8. TOKEN + POSITION EMBEDDING
# ============================================================
class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, pad_id):
        super().__init__()

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim
        )

    def forward(self, x):
        batch_size, seq_length = x.shape

        positions = torch.arange(seq_length, device=x.device)
        positions = positions.unsqueeze(0)
        positions = positions.expand(batch_size, seq_length)

        word_embeddings = self.word_embedding(x)

        position_embeddings = self.position_embedding(positions)

        output = word_embeddings + position_embeddings

        return output


# ============================================================
# 9. TRANSFORMER ENCODER BLOCK
# ============================================================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):

        attention_output, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        attention_output = self.dropout_1(attention_output)

        output_1 = self.layernorm_1(x + attention_output)

        ffn_output = self.ffn(output_1)

        ffn_output = self.dropout_2(ffn_output)

        output_2 = self.layernorm_2(output_1 + ffn_output)

        return output_2

# ============================================================
# 10. TRANSFORMER ENCODER
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_length, num_layers, num_heads, ff_dim, pad_id, dropout=0.1):
        super().__init__()

        self.embedding = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            max_length=max_length,
            pad_id=pad_id
        )

        self.layer = nn.ModuleList([
            TransformerEncoderBlock(
                emb_dim=emb_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, input_ids, attention_mask):
        output = self.embedding(input_ids)

        key_padding_mask = attention_mask == 0

        for layer in self.layers:
            output = layer(
                output,
                key_padding_mask=key_padding_mask
            )

        return output

# ============================================================
# 11. TRANSFORMER TEXT CLASSIFIER
# ============================================================
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, pad_id, max_length, num_heads, num_layers, num_classes=2, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            emb_dim=embed_dim,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            pad_id=pad_id,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 20),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(20, num_classes)
        )

    def masked_average_pooling(self, encoder_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()

        masked_output = encoder_output * mask

        summed_output = masked_output.sum(dim=1)

        token_count = mask.sum(dim=1).clamp(min=1)

        pooled_output = summed_output / token_count

        return pooled_output

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids, attention_mask)

        output = self.masked_average_pooling(output, attention_mask)

        logits = self.classifier(output)

        return logits

# ============================================================
# 12. INITIALIZE MODEL
# ============================================================
model = TransformerEncoderClassifier(
    vocab_size=VOCAB_SIZE_ACTUAL,
    max_length=MAX_LENGTH,
    num_layers=NUM_LAYERS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    pad_id=PAD_ID,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)

model = model.to(device)

print("\nModel:")
print(model)


# ============================================================
# 13. LOSS + OPTIMIZER
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ============================================================
# 14. TRAIN ONE EPOCH
# ============================================================
def train_epoch(model, criterion, optimizer, dataloader, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        batch_size = batch.size(0)
        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        total_correct += predictions.eq(labels).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


# ============================================================
# 15. EVALUATE ONE EPOCH
# ============================================================
def evaluate_epoch(model, criterion, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)

            loss = criterion(logits, labels)

            batch_size = batch.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            total_correct += logits.eq(predictions).sum().item()

            total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


# ============================================================
# 16. TRAINING LOOP
# ============================================================
def train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, num_epochs, device, model_path, patience=7):

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "valid_loss": [],
        "valid_accuracy": []
    }

    best_valid_loss = float("inf")

    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_dataloader, device)

        valid_loss, valid_accuracy = evaluate_epoch(model, criterion, valid_dataloader, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_accuracy)

        elapsed_time = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"| Time: {elapsed_time:.2f}s "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Valid Loss: {valid_loss:.4f} "
            f"| Valid Acc: {valid_accuracy:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            epochs_without_improvement = 0

            torch.save(model.state_dict(), model_path)

            print(f"Best model saved: {model_path}")

        else:
            epochs_without_improvement += 1


        if epochs_without_improvement >= patience:
            print("Early Stopping triggered")
            break


    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    return model, history


# ============================================================
# 17. TRAIN
# ============================================================
model, history = train_model(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=NUM_EPOCHS,
    device=device,
    model_path=MODEL_PATH,
    patience=7
)

# ============================================================
# 18. TEST
# ============================================================
test_loss, test_accuracy = evaluate_epoch(
    model=model,
    dataloader=test_dataloader,
    criterion=criterion,
    device=device
)

print("\n" + "=" * 50)
print(f"Test Loss: {test_loss:.4f}" )
print(f"Test Accuracy: {test_accuracy:.4f}")
print("=" * 50)


# ============================================================
# 19. PLOT TRAINING HISTORY
# ============================================================
def plot_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.show()


    plt.figure(figsize=(8, 5))

    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["valid_accuracy"], label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Acuracy")
    plt.title("Training and Valid Accuracy")
    plt.legend()

    plt.show()

plot_history(history)

# ============================================================
# 20. INFERENCE
# ============================================================
def predict_text(text, model, tokenzier, device, max_length, pad_id):
    model.eval()

    text = preprocess_text(text)

    encoding = tokenzier.encode(text)

    input_ids = encoding.ids[:max_length]

    attention_mask = [1] * len(input_ids)

    padding_length = max_length - len(input_ids)

    input_ids += [pad_id] * padding_length

    attention_mask += [0] * padding_length

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.tensor(
        [attention_mask],
        dtype=torch.long,
        device=device
    )

    with torch.inference_mode():
        logits = model(input_ids, attention_mask)

        probabilities = torch.softmax(logits, dim=1)

        prediction = probabilities.argmax(dim=1).item()

        confidence = probabilities[0, prediction].item()

    return prediction, confidence

text = "sản phẩm rất tốt tôi rất hài lòng"

prediction, confidence = predict_text(
    text=text,
    model=model,
    tokenzier=tokenizer,
    device=device,
    max_length=MAX_LENGTH,
    pad_id=PAD_ID
)

print("\nPrediction:", prediction)
print(f"Confidence: {confidence:.4f}")