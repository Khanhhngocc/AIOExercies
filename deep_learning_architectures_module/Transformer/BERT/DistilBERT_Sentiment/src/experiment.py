from pathlib import Path
import json
import pandas as pd


def create_run_dir(base_dir):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_numbers = []

    for path in base_dir.iterdir():
        if path.is_dir() and path.name.startswith("run_"):
            try:
                run_number = int(path.name.split("_")[1])
                run_numbers.append(run_number)
            except ValueError:
                pass

    next_run = max(run_numbers, default=0) + 1

    run_dir = base_dir / f"run_{next_run:02d}"
    run_dir.mkdir()

    return run_dir


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_training_history(log_history, path):
    history_df = pd.DataFrame(log_history)
    history_df.to_csv(path, index=False)


def save_predictions(predictions, path):
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(path, index=False)


def append_summary(summary_row, path):
    summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False
    )