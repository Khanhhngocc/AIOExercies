import csv
import random

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    """
    Cố định random seed để kết quả giữa các lần chạy
    có khả năng giống nhau hơn.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Chọn device theo thứ tự:
    CUDA GPU -> Apple MPS -> CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def create_directories(*directories):
    """
    Tạo các folder nếu chưa tồn tại.
    """

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


def save_history_csv(history, save_path):
    """
    Lưu metrics từng epoch thành CSV.
    """

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_accuracy",
            "train_loss",
            "valid_accuracy",
            "valid_loss"
        ])

        for i in range(len(history["epoch"])):

            writer.writerow([
                history["epoch"][i],
                history["train_acc"][i],
                history["train_loss"][i],
                history["valid_acc"][i],
                history["valid_loss"][i]
            ])


def plot_history(history, save_path):
    """
    Vẽ Accuracy và Loss qua từng epoch.
    """

    epochs = history["epoch"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # Accuracy
    axes[0].plot(
        epochs,
        history["train_acc"],
        label="Training"
    )

    axes[0].plot(
        epochs,
        history["valid_acc"],
        label="Validation"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(
        epochs,
        history["train_loss"],
        label="Training"
    )

    axes[1].plot(
        epochs,
        history["valid_loss"],
        label="Validation"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()