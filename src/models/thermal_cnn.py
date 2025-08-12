import torch
import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self, feature_dim=1280):
        super(CNN, self).__init__()
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')

        # Modify first convolution for grayscale
        original_first_conv = self.base_model.features[0][0]
        self.base_model.features[0][0] = nn.Conv2d(
            1,  # thermal images are single-channel
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False
        )

        # Initialize weights as the mean of RGB weights
        with torch.no_grad():
            new_weight = original_first_conv.weight.mean(dim=1, keepdim=True)
            self.base_model.features[0][0].weight.copy_(new_weight)

        # Feature extraction layers
        self.features = self.base_model.features
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Optional feature refinement
        self.feature_refiner = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x, raw=False):
        """
        Args:
            x (Tensor): Input tensor of shape [B, 1, H, W]
            raw (bool): If True, return raw pooled features without refinement.

        Returns:
            features (Tensor): Feature vector
        """
        x = self.features(x)
        x = self.feature_processor(x)
        if raw:
            return x
        return self.feature_refiner(x)