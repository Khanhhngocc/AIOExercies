import json
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.config import MODEL_DIR, CHECKPOINT_DIR, RESULTS_DIR
from src.data import load_data, load_tokenizer, tokenize_dataset, create_data_collator
from src.metrics import compute_metrics

RUN_NAME = "run_01"

def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError("Model not found. Run train.py first.")

    # 1. Load dataset and saved model
    dataset, id2label, _ = load_data()
    tokenizer = load_tokenizer(str(MODEL_DIR))
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # 2. Evaluation configuration
    eval_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR / "evaluation"),
        per_device_eval_batch_size=16,
        report_to="none"
    )

    # 3. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=create_data_collator(tokenizer),
        compute_metrics=compute_metrics
    )

    # 4. Evaluate
    results = trainer.evaluate()

    print("\nTest Results:")
    for metric_name, value in results.items():
        print(f"{metric_name}: {value}")

    # 5. Predict full test set
    prediction_output = trainer.predict(tokenized_dataset["test"])
    logits = torch.tensor(prediction_output.predictions)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(probabilities, dim=-1).numpy()
    true_ids = prediction_output.label_ids

    # 6. Prepare result directories
    run_dir = RESULTS_DIR / RUN_NAME

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 7. Save full test predictions
    prediction_rows = []

    for i in range(len(dataset["test"])):
        true_id = int(true_ids[i])
        predicted_id = int(predicted_ids[i])
        confidence = float(probabilities[i][predicted_id])

        prediction_rows.append({
            "text": dataset["test"][i]["text"],
            "true_label": id2label[true_id],
            "predicted_label": id2label[predicted_id],
            "confidence": confidence,
            "correct": true_id == predicted_id
        })

    pd.DataFrame(prediction_rows).to_csv(
        run_dir / "full_test_predictions.csv",
        index=False
    )

    # 8. Save classification report
    labels = [id2label[i] for i in range(len(id2label))]

    report = classification_report(
        true_ids,
        predicted_ids,
        target_names=labels,
        output_dict=True
    )

    with open(run_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # 9. Save confusion matrix
    cm = confusion_matrix(true_ids, predicted_ids)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(values_format="d")
    plt.title("DistilBERT Sentiment Classification")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # 10. Print classification report
    print("\nClassification Report:")
    print(classification_report(
        true_ids,
        predicted_ids,
        target_names=labels
    ))

    print(f"\nFull predictions saved at: {run_dir / 'full_test_predictions.csv'}")
    print(f"Classification report saved at: {run_dir / 'classification_report.json'}")
    print(f"Confusion matrix saved at: {figures_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    main()