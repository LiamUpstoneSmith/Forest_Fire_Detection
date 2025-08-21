import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from src.models import ViT, CNN, FusionModel, FusionNN
from src.config import configs
import os

# Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disables CUDA entirely
device = torch.device("cpu")

# Load Models for Inference
print("Loading pretrained feature extractors...")
vit_extractor = ViT.load_from_checkpoint("src/saved/features/ViT_fire_feature_extractor.pth")
cnn_extractor = CNN(configs("CNN"))
cnn_extractor.load_state_dict(
    torch.load("src/saved/features/CNN_fire_feature_extractor.pth", map_location=device),
    strict=False
)
cnn_extractor.classifier = nn.Identity()

vit_extractor.eval().to(device)
cnn_extractor.eval().to(device)
print("Feature extractors ready.")

# Fusion model
# THIS NEEDS TO BE TRAINED FUSION_MODEL
fusion_model =FusionNN(
        vit_extractor=vit_extractor,
        cnn_extractor=cnn_extractor,
        config=configs("fusion"),
    )

fusion_model.eval().to(device)
print("Fusion model loaded.")

# Predictor wrapper
class FirePredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def preprocess_rgb(self, pil_image, size=(224, 224)):
        pil_image = pil_image.resize(size).convert("RGB")
        arr = np.array(pil_image, dtype=np.float32) / 255.0
        tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def preprocess_thermal(self, pil_image, size=(224, 224)):
        pil_image = pil_image.resize(size).convert("L")  # grayscale
        arr = np.array(pil_image, dtype=np.float32) / 255.0
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return tensor.to(self.device)

    def predict(self, rgb_image, thermal_image):
        rgb_tensor = self.preprocess_rgb(rgb_image)
        thermal_tensor = self.preprocess_thermal(thermal_image)

        with torch.no_grad():
            outputs = self.model(rgb_tensor, thermal_tensor)
            probs = torch.sigmoid(outputs)  # convert to [0,1] probability
            pred = probs.item()
        return pred


predictor = FirePredictor(fusion_model, device)

# FastAPI App
app = FastAPI()

# Allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (optional frontend)
app.mount("/src/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("src/static/index.html")

@app.post("/predict-image/")
async def predict_image(
    rgb_file: UploadFile = File(...),
    thermal_file: UploadFile = File(...)
):
    try:
        print(f"Received files: {rgb_file.filename}, {thermal_file.filename}")  # Debug

        # Read RGB image
        rgb_bytes = await rgb_file.read()
        rgb_image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")

        # Read Thermal image
        thermal_bytes = await thermal_file.read()
        thermal_image = Image.open(io.BytesIO(thermal_bytes)).convert("L")  # single-channel

        # Run prediction
        prediction = predictor.predict(rgb_image, thermal_image)

        # Example mapping
        label = "Fire" if prediction >= 0.5 else "No Fire"

        print(f"Prediction: {label}")  # Debug
        return {"prediction": label}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# -------------------------
# To run:
# uvicorn src.app:app --reload
# -------------------------
