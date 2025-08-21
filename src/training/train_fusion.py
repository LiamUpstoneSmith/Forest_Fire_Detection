import torch.nn as nn
import pytorch_lightning as pl
from src.models import FusionModel, ViT, CNN, FusionNN
from src.data.dataset import *
import numpy as np

def train_fusion(vit_extractor, cnn_extractor, config, train_dataloader, val_dataloader, test_dataloader ):
    """
    End-to-end training pipeline for multimodal fusion model with validation and checkpointing.

    Args:
        vit_extractor: Pretrained ViT feature extractor
        cnn_extractor: Pretrained CNN feature extractor
        config: Hyperparameter configuration dictionary
    Returns:
        model: Trained fusion model
        test_results: Dictionary of test performance metrics
    """

    # Check validation set size
    if val_dataloader is not None:
        val_size = len(val_dataloader.dataset)
        print(f"Validation samples: {val_size}")

        # Check class distribution
        all_val_labels = []
        for _, _, labels in val_dataloader:
            labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
            all_val_labels.extend(labels)

        non_fire_count = sum(all_val_labels)
        fire_count = len(all_val_labels) - non_fire_count
        print(f"Non-Fire samples: {non_fire_count} ({non_fire_count/len(all_val_labels):.1%})")
        print(f"Fire samples: {fire_count} ({fire_count/len(all_val_labels):.1%})")
    else:
        print("No validation loader provided. Training will run without validation.")

    # Initialize fusion model
    print("Initializing fusion model...")
    model = FusionNN(
        vit_extractor=vit_extractor,
        cnn_extractor=cnn_extractor,
        config=config,
    )

    # Validate dataloaders
    print("\nDataloader summary:")
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader) if val_dataloader else 0}")
    print(f"Test batches: {len(test_dataloader)}")

    # Configure callbacks
    callbacks = []
    if val_dataloader is None or len(val_dataloader) == 0:
        print("\nValidation loader is empty. Switching to train-only mode.")
    else:
        monitor_metric = "val_acc"
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=5,
                mode="max",
                verbose=True
            ),
            pl.callbacks.ModelCheckpoint(
                monitor=monitor_metric,
                mode="max",
                filename="best-fusion-model",
                save_top_k=1,
                save_last=True
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs"),
        accelerator="CPU",     # auto Selects GPU/CPU automatically
        devices="auto",
        log_every_n_steps=5,
        callbacks=callbacks,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        enable_model_summary=True
    )

    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Restore best checkpoint if available
    if val_dataloader and hasattr(trainer, "checkpoint_callback"):
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"\nLoading best checkpoint: {best_model_path}")
            model = FusionNN.load_from_checkpoint(
                best_model_path,
                vit_extractor=vit_extractor,
                cnn_extractor=cnn_extractor,
                config=config
            )

    # Test evaluation
    print("\nEvaluating on test set...")
    test_results = trainer.test(model, dataloaders=test_dataloader)

    return model, test_results


if __name__ == "__main__":
    # Load pretrained feature extractors
    print("Loading pretrained feature extractors...")
    vit_extractor = ViT.load_from_checkpoint(
        "../saved/features/ViT_fire_feature_extractor.pth"
    )
    cnn_extractor = CNN(configs("CNN"))
    cnn_extractor.load_state_dict(
        torch.load("../saved/features/CNN_fire_feature_extractor.pth"),
        strict=False
    )
    cnn_extractor.classifier = nn.Identity()

    vit_extractor.eval()
    cnn_extractor.eval()
    print("Feature extractors ready.")

    # Fusion model config
    config = configs("fusion")

    # Train fusion model
    print("\nTraining fusion model...")
    fusion_model, results = train_fusion(
        vit_extractor=vit_extractor,
        cnn_extractor=cnn_extractor,
        config=config,
    )

    # Save trained weights
    fusion_model.eval()
    torch.save(fusion_model.state_dict(), "../saved/features/fusion_model_weights.pth")
    print("Fusion model weights saved.")

    # Initialize predictor for deployment
    predictor = FusionModel(fusion_model, vit_extractor, cnn_extractor)
    print("Fire predictor initialized.")