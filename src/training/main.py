import torch.nn as nn
from src.data.dataset import *
from src.training import *
from src.models import *


def main():
    # Clear GPU memory cache
    torch.cuda.empty_cache()

    print("Creating Dataloaders...\n")
    # Define Dataloaders
    dataloaders = prepareCombinedDataset()

    # === CNN Training ===
    print("\n\nCNN Training Starting...\n")
    # Clear GPU memory cache
    torch.cuda.empty_cache()

    # Access thermal dataloaders
    thermal_train, thermal_val, thermal_test = dataloaders["thermal"]

    # Define Config
    cnn_config = configs("cnn")

    # Train model
    trained_cnn = train_cnn(cnn_config, thermal_train, thermal_val, thermal_test)

    # === ViT Training ===
    print("\n\nViT Training Starting...\n")
    # Clear GPU memory cache
    torch.cuda.empty_cache()

    # Access thermal dataloaders
    rgb_train, rgb_val, rgb_test= dataloaders["rgb"]

    # Define Config
    vit_config = configs("vit")

    # train Model
    trained_model, trainer = train_vit(vit_config, rgb_train, rgb_val, rgb_test)

    # === Fusion Training ===
    print("\n\nFusion Training Starting...\n")
    # Load pretrained feature extractors
    print("Loading pretrained feature extractors...")
    vit_extractor = ViT.load_from_checkpoint(
        "../saved/features/ViT_fire_feature_extractor.pth"
    )
    cnn_extractor = CNN(cnn_config)
    cnn_extractor.load_state_dict(
        torch.load("../saved/features/CNN_fire_feature_extractor.pth"),
        strict=False
    )
    cnn_extractor.classifier = nn.Identity()

    vit_extractor.eval()
    cnn_extractor.eval()
    print("Feature extractors ready.")

    # Fusion model config
    fusion_config = configs("fusion")

    # Access fusion dataloaders
    fusion_train, fusion_val, fusion_test = dataloaders["fusion"]

    # Train fusion model
    print("\nTraining fusion model...")
    fusion_model, results = train_fusion(
        vit_extractor=vit_extractor,
        cnn_extractor=cnn_extractor,
        config=fusion_config,
        train_dataloader=fusion_train,
        val_dataloader=fusion_val,
        test_dataloader=fusion_test
    )

    # Save trained weights
    fusion_model.eval()
    torch.save(fusion_model.state_dict(), "../saved/features/fusion_model_weights.pth")
    print("Fusion model weights saved.")

    # Initialize predictor for deployment
    predictor = FusionModel(fusion_model, vit_extractor, cnn_extractor)
    print("Fire predictor initialized.")

    return predictor

if __name__ == "__main__":
    predictor = main()