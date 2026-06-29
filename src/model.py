import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, in_channel: int, num_classes: int):
        """
        Initialize a U-Net model with an encoder–decoder architecture.

        Args:
            in_channel (int): Number of input image channels.
            num_classes (int): Number of output segmentation classes.

        Returns:
            None: Initializes all convolution, pooling, and upsampling layers.
        """
        super().__init__()

        # We go down
        self.conv1_1 = nn.Conv2d(in_channel, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=1)

        # Now we go up
        self.upconvo1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, 3, padding=1)

        self.upconvo2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv7_2 = nn.Conv2d(256, 256, 3, padding=1)

        self.upconvo3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.upconvo4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor):
        """
        Perform a forward pass through the U-Net encoder and decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channel, H, W).

        Returns:
            torch.Tensor: Output segmentation logits of shape (B, num_classes, H, W).
        """
        # we go down
        x1 = self.relu(self.conv1_1(x))
        x1 = self.relu(self.conv1_2(x1))

        x2 = self.maxpool1(x1)
        x2 = self.relu(self.conv2_1(x2))
        x2 = self.relu(self.conv2_2(x2))

        x3 = self.maxpool2(x2)
        x3 = self.relu(self.conv3_1(x3))
        x3 = self.relu(self.conv3_2(x3))

        x4 = self.maxpool3(x3)
        x4 = self.relu(self.conv4_1(x4))
        x4 = self.relu(self.conv4_2(x4))

        x5 = self.maxpool4(x4)
        x5 = self.relu(self.conv5_1(x5))
        x5 = self.relu(self.conv5_2(x5))

        # Decoder
        up1 = self.upconvo1(x5)
        up1 = torch.cat([up1, x4], dim=1)
        up1 = self.relu(self.conv6_1(up1))
        up1 = self.relu(self.conv6_2(up1))

        up2 = self.upconvo2(up1)
        up2 = torch.cat([up2, x3], dim=1)
        up2 = self.relu(self.conv7_1(up2))
        up2 = self.relu(self.conv7_2(up2))

        up3 = self.upconvo3(up2)
        up3 = torch.cat([up3, x2], dim=1)
        up3 = self.relu(self.conv8_1(up3))
        up3 = self.relu(self.conv8_2(up3))

        up4 = self.upconvo4(up3)
        up4 = torch.cat([up4, x1], dim=1)
        up4 = self.relu(self.conv9_1(up4))
        up4 = self.relu(self.conv9_2(up4))

        output = self.conv10(up4)

        return output


