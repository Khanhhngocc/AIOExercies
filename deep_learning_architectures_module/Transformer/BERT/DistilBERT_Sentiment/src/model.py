from transformers import AutoModelForSequenceClassification
from src.config import MODEL_NAME

def create_model(id2label, label2id):

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )