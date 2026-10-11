from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from src.config import DATASET_NAME, MODEL_NAME, MAX_SEQ_LENGTH

def load_data():
    dataset = load_dataset(DATASET_NAME)
    label_names = dataset["train"].features["label"].names

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    
    return dataset, id2label, label2id

def load_tokenizer(model_name=MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_dataset(dataset, tokenizer):
    max_length = min(MAX_SEQ_LENGTH, tokenizer.model_max_length)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )

def create_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)