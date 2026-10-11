import argparse
import torch
from functools import lru_cache
from transformers import AutoModelForSequenceClassification
from src.config import MODEL_DIR, MAX_SEQ_LENGTH
from src.data import load_tokenizer

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@lru_cache(maxsize=1)
def load_predictor():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError("Model not found. Run train.py first.")

    tokenizer = load_tokenizer(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(get_device())
    model.eval()
    return tokenizer, model

def predict_sentiment(text):
    tokenizer, model = load_predictor()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=min(MAX_SEQ_LENGTH, tokenizer.model_max_length)
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    predicted_label = model.config.id2label[predicted_class]
    return predicted_label, confidence

def main():
    parser = argparse.ArgumentParser(description="Predict movie review sentiment")
    parser.add_argument("--text", type=str, help="English text to classify")
    args = parser.parse_args()

    examples = [args.text] if args.text else [
        "This movie was absolutely fantastic. I loved every minute of it.",
        "The movie was boring and the story was a complete waste of time."
    ]

    for text in examples:
        label, confidence = predict_sentiment(text)
        print(f"\nText: {text}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()