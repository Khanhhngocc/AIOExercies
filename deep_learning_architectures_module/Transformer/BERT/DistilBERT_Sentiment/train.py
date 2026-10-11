import json
import pandas as pd
import torch

from transformers import Trainer, TrainingArguments
from src.config import CHECKPOINT_DIR, MODEL_DIR, RESULTS_DIR, SEED
from src.data import load_data, load_tokenizer, tokenize_dataset, create_data_collator
from src.model import create_model
from src.metrics import compute_metrics
from src.experiment import create_run_dir, save_json

def main():
    # 1. Load dataset and tokenizer
    dataset, id2label, label2id = load_data()
    tokenizer = load_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    data_collator = create_data_collator(tokenizer)

    print("Dataset:", dataset)
    print("Labels:", id2label)
    print("Sample:", tokenized_dataset["train"][0])

    # 2. Initialize model
    model = create_model(id2label, label2id)

    # 3. Training configuration
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=SEED
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 5. Create experiment run
    run_dir = create_run_dir(RESULTS_DIR)

    run_config = {
        "model_name": "distilbert-base-uncased",
        "dataset": "rotten_tomatoes",
        "learning_rate": 2e-5,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "max_length": 128,
        "seed": 59
    }

    save_json(run_config, run_dir / "config.json")

    # 6. Fine-tune model
    train_result = trainer.train()

    save_json(
        train_result.metrics,
        run_dir / "train_metrics.json"
    )

    # 7. Save training history
    training_history = pd.DataFrame(trainer.state.log_history)
    training_history.to_csv(run_dir / "training_history.csv", index=False)

    # 8. Evaluate on test set
    test_results = trainer.evaluate(tokenized_dataset["test"])

    print("\nTest Results:")
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value}")

    save_json(test_results, run_dir / "test_metrics.json")

    # 9. Save sample predictions
    sample_size = min(20, len(dataset["test"]))
    raw_samples = dataset["test"].select(range(sample_size))
    tokenized_samples = tokenized_dataset["test"].select(range(sample_size))

    prediction_output = trainer.predict(tokenized_samples)
    logits = torch.tensor(prediction_output.predictions)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(probabilities, dim=-1)

    prediction_rows = []

    for i in range(sample_size):
        true_id = int(raw_samples[i]["label"])
        predicted_id = int(predicted_ids[i])
        confidence = float(probabilities[i][predicted_id])

        prediction_rows.append({
            "text": raw_samples[i]["text"],
            "true_label": id2label[true_id],
            "predicted_label": id2label[predicted_id],
            "confidence": confidence,
            "correct": true_id == predicted_id
        })

    pd.DataFrame(prediction_rows).to_csv(
        run_dir / "sample_predictions.csv",
        index=False
    )

    # 10. Update experiment summary
    summary_row = {
        "run": run_dir.name,
        "model": run_config["model_name"],
        "learning_rate": run_config["learning_rate"],
        "epochs": run_config["num_train_epochs"],
        "train_batch_size": run_config["train_batch_size"],
        "train_loss": train_result.metrics.get("train_loss"),
        "best_validation_accuracy": trainer.state.best_metric,
        "test_loss": test_results.get("eval_loss"),
        "test_accuracy": test_results.get("eval_accuracy")
    }

    summary_file = RESULTS_DIR / "summary.csv"
    summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(
        summary_file,
        mode="a",
        header=not summary_file.exists(),
        index=False
    )

    # 11. Save best model and tokenizer
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # 12. Save test metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nExperiment results saved at: {run_dir}")
    print(f"Best model saved at: {MODEL_DIR}")

if __name__ == "__main__":
    main()