import torch.nn as nn
import torch.nn.functional as F


class LeNetClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()


        # ====================================================
        # CONVOLUTION BLOCK 1
        # ====================================================

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding="same"
        )

        self.avgpool1 = nn.AvgPool2d(
            kernel_size=2
        )


        # ====================================================
        # CONVOLUTION BLOCK 2
        # ====================================================

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )

        self.avgpool2 = nn.AvgPool2d(
            kernel_size=2
        )


        # ====================================================
        # FLATTEN
        # ====================================================

        self.flatten = nn.Flatten()


        # ====================================================
        # FULLY CONNECTED LAYERS
        # ====================================================

        self.fc_1 = nn.Linear(
            16 * 5 * 5,
            120
        )

        self.fc_2 = nn.Linear(
            120,
            84
        )

        self.fc_3 = nn.Linear(
            84,
            num_classes
        )


    def forward(self, inputs):

        # Input:
        # [B, 1, 28, 28]

        outputs = self.conv1(inputs)

        # [B, 6, 28, 28]

        outputs = self.avgpool1(outputs)

        # [B, 6, 14, 14]

        outputs = F.relu(outputs)


        outputs = self.conv2(outputs)

        # [B, 16, 10, 10]

        outputs = self.avgpool2(outputs)

        # [B, 16, 5, 5]

        outputs = F.relu(outputs)


        outputs = self.flatten(outputs)

        # [B, 400]


        # Giữ đúng structure của bài 34.2
        outputs = self.fc_1(outputs)

        # [B, 120]

        outputs = self.fc_2(outputs)

        # [B, 84]

        outputs = self.fc_3(outputs)

        # [B, num_classes]
        # MNIST -> [B, 10]

        return outputs