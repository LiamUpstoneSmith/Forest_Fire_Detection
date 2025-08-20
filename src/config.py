
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
        "batch_size": 16,
        "num_workers": 4,
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
        # Model
        "lr": 5e-4,
        "hidden_dim": 768,
        "dropout_rate": 0.2,
        "pos_weight": 1.0,
        "max_epochs": 20,
        "visuals": True, # If you want visualisations
        "weight_decay": 1e-4, # L2 regularization

        # Data
        "vit_fire_dir": "../../dataset/raw/RGB/fire",
        "vit_non_fire_dir": "../../dataset/raw/RGB/no_fire",
        "cnn_fire_dir": "../../dataset/raw/thermal/fire",
        "cnn_non_fire_dir": "../../dataset/raw/thermal/no_fire",
        "processed_rgb_dir": "../../dataset/processed/RGB",
        "processed_thermal_dir": "../../dataset/processed/thermal",
        "num_aug_copies": 10,
        "batch_size": 16
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
