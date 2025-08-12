import torch
from torch.utils.data import Dataset
from PIL import Image

class DualInputDataset(Dataset):
    def __init__(self, vit_image_paths, cnn_image_paths, labels, vit_transform, cnn_transform):
        """
        Dataset for dual-input models (RGB + thermal)

        Args:
            vit_image_paths: List of paths to RGB images (ViT input)
            cnn_image_paths: List of paths to thermal images (CNN input)
            labels: List of corresponding labels
            vit_transform: Transform pipeline for ViT images
            cnn_transform: Transform pipeline for CNN images
        """
        self.vit_image_paths = vit_image_paths
        self.cnn_image_paths = cnn_image_paths
        self.labels = labels
        self.vit_transform = vit_transform
        self.cnn_transform = cnn_transform

        # Validate data alignment
        if len(vit_image_paths) != len(cnn_image_paths) or len(vit_image_paths) != len(labels):
            raise ValueError("Input lengths mismatch: ViT paths, CNN paths and labels must be equal")

    def __len__(self):
        """Returns total number of samples"""
        return len(self.labels)

    def __getitem__(self, idx):
        """Loads and transforms image pair, returns (vit_input, cnn_input, label)"""
        # Process RGB image for ViT
        vit_img = Image.open(self.vit_image_paths[idx]).convert('RGB')
        vit_tensor = self.vit_transform(vit_img)

        # Process thermal image for CNN
        cnn_img = Image.open(self.cnn_image_paths[idx])
        cnn_tensor = self.cnn_transform(cnn_img)

        # Convert label to tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return vit_tensor, cnn_tensor, label