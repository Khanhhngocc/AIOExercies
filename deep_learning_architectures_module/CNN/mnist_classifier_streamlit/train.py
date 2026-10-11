import torch

import torch.nn as nn
import torch.optim as optim


from src.config import (
    DATA_DIR,
    LOG_DIR,
    MODEL_DIR,
    MODEL_PATH,
    HISTORY_PATH,
    CURVE_PATH,
    SEED,
    TRAIN_RATIO,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES
)

from src.dataset import (
    create_dataloaders
)

from src.model import (
    LeNetClassifier
)

from src.trainer import (
    fit,
    evaluate
)

from src.utils import (
    set_seed,
    get_device,
    create_directories,
    save_history_csv,
    plot_history
)


def main():

    # ========================================================
    # 1. PREPARE PROJECT
    # ========================================================

    create_directories(
        DATA_DIR,
        LOG_DIR,
        MODEL_DIR
    )


    set_seed(SEED)


    device = get_device()

    print(
        f"Using device: {device}"
    )


    # ========================================================
    # 2. DATA
    # ========================================================

    (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        mean,
        std
    ) = create_dataloaders(

        data_dir=DATA_DIR,

        batch_size=BATCH_SIZE,

        train_ratio=TRAIN_RATIO,

        seed=SEED
    )


    # ========================================================
    # INSPECT ONE BATCH
    # ========================================================

    inputs, labels = next(
        iter(train_dataloader)
    )


    print(
        "Input batch shape:",
        inputs.shape
    )

    print(
        "Label batch shape:",
        labels.shape
    )


    # Expected:
    #
    # inputs:
    # [256, 1, 28, 28]
    #
    # labels:
    # [256]


    # ========================================================
    # 3. MODEL
    # ========================================================

    model = LeNetClassifier(
        num_classes=NUM_CLASSES
    )


    model = model.to(
        device
    )


    print(model)


    # ========================================================
    # 4. LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # 5. OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # 6. TRAIN
    # ========================================================

    history = fit(

        model=model,

        train_dataloader=
            train_dataloader,

        valid_dataloader=
            valid_dataloader,

        optimizer=optimizer,

        criterion=criterion,

        device=device,

        num_epochs=NUM_EPOCHS,

        model_path=MODEL_PATH,

        num_classes=NUM_CLASSES,

        mean=mean,

        std=std
    )


    # ========================================================
    # 7. SAVE LOG
    # ========================================================

    save_history_csv(
        history,
        HISTORY_PATH
    )


    plot_history(
        history,
        CURVE_PATH
    )


    # ========================================================
    # 8. LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()


    print(
        "Best epoch:",
        checkpoint["epoch"]
    )


    # ========================================================
    # 9. TEST
    # ========================================================

    test_acc, test_loss = evaluate(

        model=model,

        dataloader=test_dataloader,

        criterion=criterion,

        device=device
    )


    print("\nTEST RESULT")

    print(
        f"Test Accuracy: "
        f"{test_acc:.4f}"
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )


if __name__ == "__main__":
    main()