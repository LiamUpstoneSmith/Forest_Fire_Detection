def configs(model):
    vit_config = {
        "num_epochs": 40,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "freeze_backbone": True,
        "pretrained": True,
        "dropout_rate": 0.5,
        "pos_weight": None,
        "save_FE_path": "../saved/features/ViT_fire_feature_extractor.pth", # Save feature extractor path
    }

    cnn_config = {
        "num_epochs":15,
        "lr": 1e-4,
        "max_grad_norm": 0.1,
        "dropout_rate": 0.3,
        "save_FE_path": "../saved/features/CNN_fire_feature_extractor.pth",
    }

    fusion_config = {
        "lr": 5e-4,
        "hidden_dim": 768,
        "dropout_rate": 0.2,
        "pos_weight": 1.0,
        "max_epochs": 20,
        "visuals": True, # If you want visualisations
        "weight_decay": 1e-4, # L2 regularization
    }

    data = {
        "vit_fire_dir": "../../dataset/raw/RGB/fire",
        "vit_non_fire_dir": "../../dataset/raw/RGB/no_fire",
        "cnn_fire_dir": "../../dataset/raw/thermal/fire",
        "cnn_non_fire_dir": "../../dataset/raw/thermal/no_fire",
        "batch_size": 16,
        "img_size": (224, 224),
        "rotation": 15,
        "hflip_p": 0.5, # Horizontal flip probability
        "vflip_p": 0.5, # Vertical flip probability
        "multiplier": 3
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
    elif model == "data":
        config = data

    return config