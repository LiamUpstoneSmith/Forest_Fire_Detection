import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from src.models import ViT, CNN, FusionModelPredictor, FusionNN
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
fusion_model = FusionNN(
        vit_extractor=vit_extractor,
        cnn_extractor=cnn_extractor,
        config=configs("fusion"),
    )
fusion_model.load_state_dict(torch.load("src/saved/features/fusion_model_weights.pth", map_location=device), strict=False)

vit_extractor.eval().to(device)
cnn_extractor.eval().to(device)
fusion_model.eval().to(device)
print("Feature extractors ready.")

predictor = FusionModelPredictor(fusion_model, vit_extractor, cnn_extractor)

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
        prediction, probability  = predictor.predict(rgb_image, thermal_image)

        return JSONResponse(
            content={
                "prediction": prediction,
                "probability": probability
            }
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# -------------------------
# To run:
# uvicorn src.app:app --reload
# -------------------------
