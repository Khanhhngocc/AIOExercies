import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms

from .model import LeNetClassifier


def load_trained_model(
    model_path,
    device
):
    """
    Load checkpoint đã save lúc training.
    """

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )


    num_classes = checkpoint[
        "num_classes"
    ]


    model = LeNetClassifier(
        num_classes=num_classes
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(device)

    model.eval()


    mean = checkpoint["mean"]
    std = checkpoint["std"]


    return (
        model,
        mean,
        std
    )


def preprocess_image(
    image_path,
    mean,
    std
):
    """
    Biến ảnh ngoài thành đúng dạng MNIST model cần.
    """

    image = Image.open(
        image_path
    )


    # ========================================================
    # CENTER CROP TO SQUARE
    # ========================================================

    width, height = image.size

    if width != height:

        crop_size = min(
            width,
            height
        )

        crop = transforms.CenterCrop(
            crop_size
        )

        image = crop(image)


    # ========================================================
    # TRANSFORM
    # ========================================================

    transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.Resize(
            (28, 28)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[mean],
            std=[std]
        )
    ])


    image_tensor = transform(
        image
    )

    # Trước:
    # [1, 28, 28]
    #
    # Model cần:
    # [B, C, H, W]
    #
    # nên thêm batch dimension

    image_tensor = image_tensor.unsqueeze(
        dim=0
    )

    # [1, 1, 28, 28]

    return image_tensor


def predict_image(
    image_path,
    model,
    mean,
    std,
    device
):
    """
    Predict 1 ảnh.
    """

    image_tensor = preprocess_image(
        image_path=image_path,
        mean=mean,
        std=std
    )


    image_tensor = image_tensor.to(
        device
    )


    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )


        confidence, predicted_class = (
            probabilities.max(dim=1)
        )


    return (
        predicted_class.item(),
        confidence.item()
    )