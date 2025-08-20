import torch
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
from typing import Tuple

class FusionModel:
    """
    Deployment class for real-time multimodal fire prediction.
    Uses RGB (ViT) and thermal (CNN) image inputs.
    """

    def __init__(self, FusionNN, vit_extractor, cnn_extractor):
        """
        Initialize the predictor with a trained fusion model and feature extractors.

        Args:
            FusionNN: Neural Network combining ViT + CNN
            vit_extractor: Vision Transformer feature extractor for RGB images
            cnn_extractor: CNN feature extractor for thermal images
        """
        self.FusionNN = FusionNN
        self.vit_extractor = vit_extractor
        self.cnn_extractor = cnn_extractor

        # Preprocessing for RGB (ViT input)
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to standard size
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(           # ImageNet normalization
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Preprocessing for thermal (CNN input)
        self.cnn_transform = v2.Compose([
            v2.Resize(size=(224, 224)),             # Resize to standard size
            v2.ToImage(),                           # Convert to tensor
            v2.ToDtype(torch.float32, scale=True),  # Normalize to [0,1]
            v2.Grayscale(num_output_channels=1),    # Ensure single channel
            v2.Normalize(mean=[0.5], std=[0.5])     # Scale to [-1,1]
        ])

    def predict(self, vit_image_path, cnn_image_path) -> Tuple[str, float]:
        """
        Predict fire probability from paired RGB and thermal images.

        Args:
            vit_image_path: Path to RGB image file
            cnn_image_path: Path to thermal image file

        Returns:
            prediction: "Fire" or "Not Fire"
            probability: Confidence score (0.0–1.0)
        """
        try:

            self.FusionNN.eval() # Set to evaluation mode

            # Process RGB image
            vit_img = Image.open(vit_image_path).convert("RGB")
            vit_tensor = self.vit_transform(vit_img).unsqueeze(0)

            # Process thermal image
            cnn_img = Image.open(cnn_image_path)
            cnn_tensor = self.cnn_transform(cnn_img).unsqueeze(0)

            device = next(self.FusionNN.parameters()).device
            vit_tensor = vit_tensor.to(device)
            cnn_tensor = cnn_tensor.to(device)

            # Make predictions
            with torch.no_grad():
                logits = self.FusionNN(vit_tensor, cnn_tensor)
                prob = torch.sigmoid(logits).cpu().item()

                prediction = "Fire" if prob > 0.5 else "Not Fire"

            return prediction, prob

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}") from e