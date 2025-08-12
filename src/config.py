
def configs(model):
    vit_config = {
        # Model
        "num_epochs": 40,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "freeze_backbone": True,
        "pretrained": True,
        "dropout_rate": 0.5,
        "pos_weight": 1.0,

        # Data
        "data_dir": "../../dataset/raw/RGB",
        "augmented_dir":"../../dataset/vit",
        "regenerate": True,
        "train_augmented_multiplicity":5,
        "save_FE_path": "../saved/features/ViT_fire_feature_extractor.pth", # Save feature extractor path
    }

    cnn_config = {
        # Model
        "num_epochs":15,
        "lr": 1e-4,
        "max_grad_norm": 0.1,
        "dropout_rate": 0.3,

        # Data
        "data_dir": "../../dataset/raw/thermal",
        "processed_dir": "../../dataset/thermal_processed",
        "train_augmented_multiplicity":5,
        "batch_size": 16,
        "regenerate": True, # Set True to recreate dataset
        "save_FE_path": "../saved/features/CNN_fire_feature_extractor.pth",
    }

    fusion_config = {

    }

    # Decide which config to return
    config = {}
    model = model.lower()
    if model == "vit":
        config = vit_config
    elif model == "cnn":
        config = cnn_config
    elif model == "fusion":
        config = fusion_config

    return config

c = configs("vit")
print(c["data_dir"])
