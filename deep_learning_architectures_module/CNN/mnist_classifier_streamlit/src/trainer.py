import time

import torch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):
    """
    Train model đúng 1 epoch.
    """

    model.train()

    total_loss = 0.0

    total_correct = 0
    total_samples = 0


    for inputs, labels in dataloader:

        # ====================================================
        # MOVE DATA TO DEVICE
        # ====================================================

        inputs = inputs.to(device)
        labels = labels.to(device)


        # ====================================================
        # 1. CLEAR OLD GRADIENTS
        # ====================================================

        optimizer.zero_grad()


        # ====================================================
        # 2. FORWARD
        # ====================================================

        predictions = model(inputs)


        # ====================================================
        # 3. COMPUTE LOSS
        # ====================================================

        loss = criterion(
            predictions,
            labels
        )


        # ====================================================
        # 4. BACKPROPAGATION
        # ====================================================

        loss.backward()


        # ====================================================
        # 5. UPDATE PARAMETERS
        # ====================================================

        optimizer.step()


        # ====================================================
        # METRICS
        # ====================================================

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predicted_classes = predictions.argmax(
            dim=1
        )

        total_correct += (
            predicted_classes == labels
        ).sum().item()

        total_samples += batch_size


    epoch_loss = (
        total_loss / total_samples
    )

    epoch_accuracy = (
        total_correct / total_samples
    )


    return (
        epoch_accuracy,
        epoch_loss
    )


def evaluate(
    model,
    dataloader,
    criterion,
    device
):
    """
    Validation hoặc Test.
    """

    model.eval()

    total_loss = 0.0

    total_correct = 0
    total_samples = 0


    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)


            # Forward only
            predictions = model(inputs)


            loss = criterion(
                predictions,
                labels
            )


            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )


            predicted_classes = predictions.argmax(
                dim=1
            )


            total_correct += (
                predicted_classes == labels
            ).sum().item()


            total_samples += batch_size


    epoch_loss = (
        total_loss / total_samples
    )

    epoch_accuracy = (
        total_correct / total_samples
    )


    return (
        epoch_accuracy,
        epoch_loss
    )


def fit(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    criterion,
    device,
    num_epochs,
    model_path,
    num_classes,
    mean,
    std
):
    """
    Training loop chính.

    Một epoch:
        Train
          ↓
        Validation
          ↓
        nếu tốt hơn -> Save model
    """

    history = {
        "epoch": [],
        "train_acc": [],
        "train_loss": [],
        "valid_acc": [],
        "valid_loss": []
    }


    best_valid_loss = float("inf")


    for epoch in range(
        1,
        num_epochs + 1
    ):

        start_time = time.time()


        # ====================================================
        # TRAIN
        # ====================================================

        train_acc, train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        valid_acc, valid_loss = evaluate(
            model=model,
            dataloader=valid_dataloader,
            criterion=criterion,
            device=device
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history["epoch"].append(epoch)

        history["train_acc"].append(
            train_acc
        )

        history["train_loss"].append(
            train_loss
        )

        history["valid_acc"].append(
            valid_acc
        )

        history["valid_loss"].append(
            valid_loss
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "num_classes":
                    num_classes,

                "mean":
                    mean,

                "std":
                    std,

                "epoch":
                    epoch,

                "valid_loss":
                    valid_loss
            }

            torch.save(
                checkpoint,
                model_path
            )

            print(
                f"Saved best model "
                f"at epoch {epoch}"
            )


        # ====================================================
        # LOG
        # ====================================================

        elapsed = (
            time.time() - start_time
        )

        print("-" * 70)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Time: {elapsed:.2f}s"
        )

        print(
            f"Train | "
            f"Accuracy: {train_acc:.4f} | "
            f"Loss: {train_loss:.4f}"
        )

        print(
            f"Valid | "
            f"Accuracy: {valid_acc:.4f} | "
            f"Loss: {valid_loss:.4f}"
        )

        print("-" * 70)


    return history