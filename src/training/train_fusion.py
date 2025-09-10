import os
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from src.models import FusionModelPredictor, ViT, CNN, FusionNN
from src.data.dataset import *
from src.config import configs


def evaluate_classification_report_with_probs(model, dataloader, device, split_name="Dataset"):
    """Run a classification report and return true labels + probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for rgb_imgs, thermal_imgs, labels in dataloader:
            rgb_imgs, thermal_imgs, labels = (
                rgb_imgs.to(device),
                thermal_imgs.to(device),
                labels.to(device).long(),
            )

            outputs = model(rgb_imgs, thermal_imgs)
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print(f"\n--- {split_name} Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=["No Fire", "Fire"]))
    return np.array(all_labels), np.array(all_probs)


def train_fusion(vit_extractor, cnn_extractor, config, train_dataloader, val_dataloader, test_dataloader, notebook=False):
    """
    End-to-end training pipeline for multimodal fusion model with validation and checkpointing.
    """

    # Initialize fusion model
    print("Initializing fusion model...")
    model = FusionNN(
        vit_extractor=vit_extractor,
        cnn_extractor=cnn_extractor,
        config=config,
        notebook=notebook,
    )

    # Configure callbacks
    callbacks = []
    if val_dataloader:
        monitor_metric = "val_acc"
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor=monitor_metric, patience=5, mode="max", verbose=True
            ),
            pl.callbacks.ModelCheckpoint(
                monitor=monitor_metric,
                mode="max",
                filename="best-fusion-model",
                save_top_k=1,
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs"),
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        callbacks=callbacks,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        enable_model_summary=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Ensure convergence plots are always generated
    try:
        if model.visuals:
            if model.train_acc_history and model.val_acc_history:
                model.plot_accuracy_history()
            if model.train_loss_history and model.val_loss_history:
                model.plot_convergence_history()
    except Exception as e:
        print(f"Warning: Could not generate convergence plots: {e}")

    # Restore best checkpoint if available
    if val_dataloader and hasattr(trainer, "checkpoint_callback"):
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"\nLoading best checkpoint: {best_model_path}")
            model = FusionNN.load_from_checkpoint(
                best_model_path,
                vit_extractor=vit_extractor,
                cnn_extractor=cnn_extractor,
                config=config,
            )

    # --- Classification Reports ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_train, p_train = evaluate_classification_report_with_probs(
        model, train_dataloader, device, split_name="Train"
    )

    evaluate_classification_report_with_probs(
        model, val_dataloader, device, split_name="Validation"
    )
    y_test, p_test = evaluate_classification_report_with_probs(
        model, test_dataloader, device, split_name="Test"
    )

    # Lightning test results (optional)
    print("\nEvaluating with Lightning test loop...")
    test_results = trainer.test(model, dataloaders=test_dataloader)

    if model.visuals:
        model.plot_precision_recall_comparison(y_train, p_train, y_test, p_test)

    return model, test_results


if __name__ == "__main__":
    # Load pretrained extractors
    print("Loading pretrained feature extractors...")
    vit_extractor = ViT.load_from_checkpoint("../saved/features/ViT_fire_feature_extractor.pth")
    cnn_extractor = CNN(configs("CNN"))
    cnn_extractor.load_state_dict(
        torch.load("../saved/features/CNN_fire_feature_extractor.pth"),
        strict=False,
    )
    cnn_extractor.classifier = nn.Identity()

    vit_extractor.eval()
    cnn_extractor.eval()
    print("Feature extractors ready.")

    # Fusion model config
    config = configs("fusion")

    # Dataloaders
    dataloaders = prepareCombinedDataset()
    train_loader, val_loader, test_loader = dataloaders["fusion"]

    # Train fusion model
    fusion_model, results = train_fusion(
        vit_extractor=vit_extractor,
        cnn_extractor=cnn_extractor,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
    )

    # Save trained weights
    fusion_model.eval()
    torch.save(fusion_model.state_dict(), "../saved/features/fusion_model_weights.pth")
    print("Fusion model weights saved.")

    # Predictor for deployment
    predictor = FusionModelPredictor(fusion_model, vit_extractor, cnn_extractor)
    print("Fire predictor initialized.")
