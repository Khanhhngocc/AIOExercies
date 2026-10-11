import argparse

from src.config import (
    MODEL_PATH
)

from src.utils import (
    get_device
)

from src.inference import (
    load_trained_model,
    predict_image
)


def main():

    # ========================================================
    # READ COMMAND-LINE ARGUMENT
    # ========================================================

    parser = argparse.ArgumentParser(
        description=
        "Predict MNIST digit from image"
    )


    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file"
    )


    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, mean, std = (
        load_trained_model(
            model_path=MODEL_PATH,
            device=device
        )
    )


    # ========================================================
    # PREDICT
    # ========================================================

    predicted_class, confidence = (
        predict_image(

            image_path=args.image_path,

            model=model,

            mean=mean,

            std=std,

            device=device
        )
    )


    print(
        f"Prediction: {predicted_class}"
    )

    print(
        f"Confidence: "
        f"{confidence:.2%}"
    )


if __name__ == "__main__":
    main()