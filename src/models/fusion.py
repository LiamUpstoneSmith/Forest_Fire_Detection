import torch
from torchvision import transforms
from torchvision.transforms import v2

class FusionModelPredictor:
    def __init__(self, FusionNN, vit_extractor, cnn_extractor):
        self.FusionNN = FusionNN
        self.vit_extractor = vit_extractor
        self.cnn_extractor = cnn_extractor
        self.device = next(self.FusionNN.parameters()).device

        # EXACTLY the same eval transforms you used for val/test in training
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.cnn_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(num_output_channels=1),
            v2.Normalize([0.5], [0.5]),
        ])

    # Use the transforms above (no manual numpy / scaling)
    def preprocess_rgb(self, pil_image):
        return self.vit_transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

    def preprocess_thermal(self, pil_image):
        return self.cnn_transform(pil_image.convert("L")).unsqueeze(0).to(self.device)

    def predict(self, rgb_image, thermal_image):
        self.FusionNN.eval()
        with torch.no_grad():
            rgb_tensor = self.preprocess_rgb(rgb_image)
            thermal_tensor = self.preprocess_thermal(thermal_image)
            logit = self.FusionNN(rgb_tensor, thermal_tensor)  # shape [1]
            prob = torch.sigmoid(logit).item()                 # scalar in [0,1]
            prediction = "Fire" if prob > 0.5 else "Not Fire"
            return prediction, round(prob * 100, 2)
