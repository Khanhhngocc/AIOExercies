from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR

RUN_NAME = "run_01"

def main():
    # 1. Load training history
    run_dir = RESULTS_DIR / RUN_NAME
    history_file = run_dir / "training_history.csv"

    if not history_file.exists():
        raise FileNotFoundError(f"Training history not found: {history_file}")

    history = pd.read_csv(history_file)

    print("Columns:", history.columns.tolist())
    print(history.head())

    # 2. Create figures directory
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 3. Plot training and validation loss
    train_history = history.dropna(subset=["loss"])
    eval_history = history.dropna(subset=["eval_loss"])

    plt.figure(figsize=(8, 5))

    if not train_history.empty:
        plt.plot(train_history["epoch"], train_history["loss"], label="Training Loss")

    if not eval_history.empty:
        plt.plot(eval_history["epoch"], eval_history["eval_loss"], marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_curve.png", dpi=200)
    plt.close()

    # 4. Plot validation accuracy
    if "eval_accuracy" in history.columns:
        accuracy_history = history.dropna(subset=["eval_accuracy"])

        plt.figure(figsize=(8, 5))
        plt.plot(
            accuracy_history["epoch"],
            accuracy_history["eval_accuracy"],
            marker="o"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "validation_accuracy.png", dpi=200)
        plt.close()

    print(f"\nFigures saved at: {figures_dir}")

if __name__ == "__main__":
    main()