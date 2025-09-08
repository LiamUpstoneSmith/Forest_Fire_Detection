from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchmetrics import Accuracy

class ViT(pl.LightningModule):
    def __init__(self, config: Optional[Dict] = None):
        """
        Vision Transformer with feature extraction. Forward returns 768-d embeddings.
        """
        super().__init__()
        self.config = config or {}
        self.save_hyperparameters(self.config)

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if self.config.get("pretrained") else None
        self.vit = vit_b_16(weights=weights)

        # Determine feature dim BEFORE replacing heads
        feat_dim = 768
        try:
            # torchvision vit_b_16 exposes .heads.head.in_features
            if hasattr(self.vit, "heads") and hasattr(self.vit.heads, "head"):
                feat_dim = int(self.vit.heads.head.in_features)
        except Exception:
            pass

        # Replace classification head with identity to output raw features
        self.vit.heads = nn.Identity()

        # Optionally freeze backbone
        if self.config.get("freeze_backbone", True):
            for p in self.vit.parameters():
                p.requires_grad = False

        # Small classification head (if needed during training/metrics)
        dropout_rate = float(self.config.get("dropout_rate"))
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

        # Loss handling with optional pos_weight
        pos_w = self.config.get("pos_weight")
        if pos_w is not None:
            self.register_buffer("_pos_weight_buf", torch.as_tensor(pos_w, dtype=torch.float32))
            self.criterion = None
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

    def forward(self, x):
        """Return ViT features (B, feat_dim)."""
        return self.vit(x)

    def classify(self, x):
        """Return logits for binary classification."""
        features = self.vit(x)
        return self.classifier(features).squeeze(1)  # (B,)

    def _get_loss(self):
        if self.criterion is not None:
            return self.criterion
        if hasattr(self, "_pos_weight_buf"):
            pw = self._pos_weight_buf.to(self.device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)
        return nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        lr = float(self.config.get("lr"))
        wd = float(self.config.get("weight_decay"))
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "frequency": 1
            }
        }

    # ==== (1) Always compute loss via _get_loss(); remove double-loss ====
    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()
        outputs = self.classify(images)
        loss = self._get_loss()(outputs, labels)

        preds = (torch.sigmoid(outputs) > 0.5).int()
        self.train_acc(preds, labels.int())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()
        outputs = self.classify(images)
        loss = self._get_loss()(outputs, labels)

        preds = (torch.sigmoid(outputs) > 0.5).int()
        self.val_acc(preds, labels.int())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()
        outputs = self.classify(images)
        loss = self._get_loss()(outputs, labels)

        preds = (torch.sigmoid(outputs) > 0.5).int()
        self.test_acc(preds, labels.int())

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        return loss