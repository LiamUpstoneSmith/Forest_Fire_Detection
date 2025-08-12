from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchmetrics import Accuracy

class ViT(pl.LightningModule):
    def __init__(self, config=None):
        """
        Custom Vision Transformer model for fire detection with feature extraction capability.

        Args:
            config: Configuration dictionary with hyperparameters. Defaults to empty dict.
        """
        super().__init__()
        self.config = config or {}
        self.save_hyperparameters(config)  # Save config for checkpointing

        # Initialize with pretrained weights if specified, otherwise random initialization
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if self.config.get("pretrained", True) else None
        self.vit = vit_b_16(weights=weights)

        # Freeze backbone parameters if requested (transfer learning)
        if self.config.get("freeze_backbone", True):
            for param in self.vit.parameters():
                param.requires_grad = False

        # Replace classification head with identity to output raw features (768-dim)
        self.vit.heads = nn.Identity()

        # Custom classification head for binary fire detection
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # Feature compression layer
            nn.ReLU(),            # Activation for non-linearity
            nn.BatchNorm1d(256),  # Normalization for stability
            nn.Dropout(self.config.get("dropout_rate")),  # Regularization
            nn.Linear(256, 1)     # Binary classification output (fire/no-fire)
        )

        # Loss function with class weighting to handle imbalanced datasets
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.config.get("pos_weight"))
        )

        # Accuracy metrics for different phases
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

    def forward(self, x):
        """Feature extraction pass - returns 768-dimensional embeddings"""
        return self.vit(x)

    def classify(self, x):
        """Full classification pass (features + classifier head)"""
        features = self.vit(x)  # Extract visual features
        return self.classifier(features).squeeze(1)  # Remove extra dimension

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.get("lr"),                # Default learning rate
            weight_decay=self.config.get("weight_decay")  # L2 regularization
        )

        # Learning rate scheduler that reduces when validation accuracy plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',     # Monitor validation accuracy (maximize)
            factor=0.5,     # Reduce LR by half when triggered
            patience=3,     # Wait 3 epochs without improvement
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",  # Watch validation accuracy
                "frequency": 1         # Check every epoch
            }
        }

    def training_step(self, batch, batch_idx):
        """Single training step with loss calculation and metrics logging"""
        images, labels = batch
        labels = labels.float()  # Convert to float for BCE loss

        # Forward pass through classifier
        outputs = self.classify(images)
        loss = self.criterion(outputs, labels)

        # Convert logits to binary predictions (threshold=0.5)
        preds = torch.sigmoid(outputs) > 0.5
        self.train_acc(preds, labels)  # Update accuracy metric

        # Log training metrics (step-level loss, epoch-level accuracy)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step with metrics calculation"""
        images, labels = batch
        labels = labels.float()

        # Forward pass
        outputs = self.classify(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = torch.sigmoid(outputs) > 0.5
        self.val_acc(preds, labels)

        # Log validation metrics (epoch-level only)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Single test step for final evaluation"""
        images, labels = batch
        labels = labels.float()

        # Forward pass
        outputs = self.classify(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        preds = torch.sigmoid(outputs) > 0.5
        self.test_acc(preds, labels)

        # Log test metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)

        return loss
