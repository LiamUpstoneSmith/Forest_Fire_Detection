import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
import pandas as pd
from torchmetrics import Accuracy, F1Score, Precision, Recall

# Fusion Model combining ViT and CNN features
class FusionNN(pl.LightningModule):
    def __init__(self, vit_extractor, cnn_extractor, config=None, visuals=True, notebook=False):
        """
        Multimodal fusion model combining ViT (RGB) and CNN (thermal) features.

        Args:
            vit_extractor: Pretrained ViT feature extractor
            cnn_extractor: Pretrained CNN feature extractor
            config: Configuration dictionary for hyperparameters
            visuals: Enable advanced visualization metrics
        """
        self.notebook = notebook # Save figures path dependant if its being ran in a notebook or python files

        super().__init__()
        # Configuration setup
        default_config = {
            "lr": 1e-3,                # Learning rate
            "weight_decay": 1e-4,      # L2 regularization
            "dropout_rate": 0.4,       # Dropout probability
            "hidden_dim": 512,         # Fusion layer dimension
            "pos_weight": 1.0          # Class imbalance weight
        }
        if config:
            default_config.update(config)  # Merge with user config
        self.config = default_config

        # Visualization settings
        self.visuals = visuals
        self.save_hyperparameters(ignore=["vit_extractor", "cnn_extractor"])

        # Feature extractors (frozen)
        self.vit_extractor = vit_extractor
        self.cnn_extractor = cnn_extractor

        # Freeze extractors to preserve learned features
        for param in self.vit_extractor.parameters():
            param.requires_grad = False
        for param in self.cnn_extractor.parameters():
            param.requires_grad = False

        # Fusion classifier network
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, self.config["hidden_dim"]),  # Combine features
            nn.ReLU(),                                        # Non-linearity
            nn.BatchNorm1d(self.config["hidden_dim"]),        # Stabilization
            nn.Dropout(self.config["dropout_rate"]),          # Regularization
            nn.Linear(self.config["hidden_dim"], 1)           # Binary output
        )

        # Loss function with class weighting
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.config["pos_weight"])
        )

        # Core metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

        self.train_acc_history = []  # Store training accuracy per epoch
        self.val_acc_history = []    # Store validation accuracy per epoch

        # Advanced metrics for visualization
        if self.visuals:
            # F1 Scores
            self.train_f1 = F1Score(task='binary')
            self.val_f1 = F1Score(task='binary')
            self.test_f1 = F1Score(task='binary')

            # Precision metrics
            self.train_precision = Precision(task='binary')
            self.val_precision = Precision(task='binary')
            self.test_precision = Precision(task='binary')

            # Recall metrics
            self.train_recall = Recall(task='binary')
            self.val_recall = Recall(task='binary')
            self.test_recall = Recall(task='binary')

        # Storage for test visualizations
        self.test_preds = []    # Predicted labels
        self.test_targets = []  # Ground truth labels
        self.test_probs = []    # Prediction probabilities

    def forward(self, vit_input, cnn_input):
        """Forward pass through both feature extractors and fusion classifier"""
        # Extract features from both modalities
        vit_features = self.vit_extractor(vit_input)  # RGB features (768-dim)
        cnn_features = self.cnn_extractor(cnn_input)  # Thermal features (512-dim)

        # Concatenate features along channel dimension
        combined = torch.cat((vit_features, cnn_features), dim=1)
        return self.classifier(combined).squeeze(1)  # Remove extra dimension

    def training_step(self, batch, batch_idx):
        """Single training step with metrics calculation"""
        vit_imgs, cnn_imgs, labels = batch
        logits = self(vit_imgs, cnn_imgs)  # Forward pass
        loss = self.criterion(logits, labels)  # Compute loss

        # Convert to probabilities and predictions
        probs = torch.sigmoid(logits)
        preds = probs > 0.5  # Threshold at 0.5

        # Update metrics
        self.train_acc(preds, labels)
        if self.visuals:
            self.train_f1(preds, labels)
            self.train_precision(preds, labels)
            self.train_recall(preds, labels)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        if self.visuals:
            self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
            self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_start(self):
        """Reset training metrics at the beginning of each epoch"""
        self.train_acc.reset()
        if self.visuals:
            self.train_f1.reset()
            self.train_precision.reset()
            self.train_recall.reset()

    def on_validation_epoch_start(self):
        """Reset validation metrics at the beginning of each epoch"""
        self.val_acc.reset()
        if self.visuals:
            self.val_f1.reset()
            self.val_precision.reset()
            self.val_recall.reset()

    def validation_step(self, batch, batch_idx):
        """
        Validation step with proper model state handling

        Ensures:
        - Model is in evaluation mode
        - Gradients are disabled for efficiency
        - Metrics are properly synchronized across devices
        """
        # Set model to evaluation mode and disable gradients
        self.eval()
        with torch.no_grad():
            vit_imgs, cnn_imgs, labels = batch
            logits = self(vit_imgs, cnn_imgs)
            loss = self.criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = probs > 0.5

        # Update metrics
        self.val_acc(preds, labels)
        if self.visuals:
            self.val_f1(preds, labels)
            self.val_precision(preds, labels)
            self.val_recall(preds, labels)

        # Log metrics with synchronization for distributed training
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.visuals:
            self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_precision", self.val_precision, on_epoch=True, sync_dist=True)
            self.log("val_recall", self.val_recall, on_epoch=True, sync_dist=True)

        return loss


    def test_step(self, batch, batch_idx):
        """Test step with metrics and data collection for visualization"""
        vit_imgs, cnn_imgs, labels = batch
        logits = self(vit_imgs, cnn_imgs)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = probs > 0.5

        # Update metrics
        self.test_acc(preds, labels)
        if self.visuals:
            self.test_f1(preds, labels)
            self.test_precision(preds, labels)
            self.test_recall(preds, labels)

            # Store for visualization
            self.test_preds.append(preds.detach().cpu())
            self.test_targets.append(labels.detach().cpu())
            self.test_probs.append(probs.detach().cpu())

        # Log metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        if self.visuals:
            self.log("test_f1", self.test_f1, on_epoch=True)
            self.log("test_precision", self.test_precision, on_epoch=True)
            self.log("test_recall", self.test_recall, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        """Record training and validation accuracy at epoch end"""
        # Only record if metrics are available (handles incomplete epochs)
        if self.trainer.callback_metrics.get("train_acc") is not None:
            self.train_acc_history.append(self.trainer.callback_metrics["train_acc"].item())
        if self.trainer.callback_metrics.get("val_acc") is not None:
            self.val_acc_history.append(self.trainer.callback_metrics["val_acc"].item())

    def on_train_end(self):
        """Generate accuracy history plot after training completes"""
        if self.visuals and self.train_acc_history and self.val_acc_history:
            self.plot_accuracy_history()

    def plot_accuracy_history(self):
        """Plot training and validation accuracy over epochs"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_acc_history) + 1)

        # Plot training and validation accuracy
        plt.plot(epochs, self.train_acc_history, 'bo-', label='Training Accuracy')
        plt.plot(epochs, self.val_acc_history, 'go-', label='Validation Accuracy')

        # Format plot
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)  # Ensure accuracy range is visible
        plt.legend()
        plt.grid(True)
        if self.notebook:
            plt.savefig("figures/accuracy_history.png")
        else:
            plt.savefig("../saved/figures/accuracy_history.png")
        plt.show()

    def on_test_end(self):
        """Generate evaluation visualizations after testing completes"""
        if self.visuals and self.test_targets:
            # NEW: Gather results from all devices and flatten
            test_preds_tensor = torch.cat(self.test_preds, dim=0)
            test_targets_tensor = torch.cat(self.test_targets, dim=0)
            test_probs_tensor = torch.cat(self.test_probs, dim=0)

            # Gather across devices
            gathered_preds = self.all_gather(test_preds_tensor).cpu()
            gathered_targets = self.all_gather(test_targets_tensor).cpu()
            gathered_probs = self.all_gather(test_probs_tensor).cpu()

            # Flatten results
            gathered_preds = gathered_preds.view(-1)
            gathered_targets = gathered_targets.view(-1)
            gathered_probs = gathered_probs.view(-1)

            # Only generate visuals on main process
            if self.trainer.is_global_zero:
                self.test_preds = gathered_preds.numpy()
                self.test_targets = gathered_targets.numpy()
                self.test_probs = gathered_probs.numpy()
                self.generate_visualizations()

            # Clear stored data
            self.test_preds = []
            self.test_targets = []
            self.test_probs = []

    def generate_visualizations(self):
        """Create comprehensive evaluation plots and reports"""
        y_true = np.array(self.test_targets)
        y_pred = np.array(self.test_preds)
        y_probs = np.array(self.test_probs)

        # 1. Classification Report
        print("\n" + "="*50)
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Non-Fire", "Fire"]))

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Fire', 'Fire'],
                    yticklabels=['Non-Fire', 'Fire'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        if self.notebook:
            plt.savefig("figures/confusion_matrix.png")
        else:
            plt.savefig("../saved/figures/confusion_matrix.png")
        plt.show()

        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        if self.notebook:
            plt.savefig("figures/roc_curve.png")
        else:
            plt.savefig("../saved/figures/roc_curve.png")
        plt.show()

        # 4. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        avg_precision = np.mean(precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'Precision-Recall (Avg Precision = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        if self.notebook:
            plt.savefig("figures/precision_curve.png")
        else:
            plt.savefig("../saved/figures/precision_recall_curve.png")
        plt.show()

        # 5. Probability Distribution
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame({
            'Probability': y_probs,
            'Class': ['Fire' if t == 1 else 'Non-Fire' for t in y_true]
        })
        sns.histplot(
            data=df,
            x='Probability', hue='Class', element='step', stat='density',
            common_norm=False, bins=20, palette=['red', 'green']
        )
        plt.title('Predicted Probability Distribution')
        plt.axvline(0.5, color='black', linestyle='--')
        if self.notebook:
            plt.savefig("figures/probability_distribution.png")
        else:
            plt.savefig("../saved/figures/probability_distribution.png")
        plt.show()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),  # Only optimize fusion classifier
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )

        # Learning rate scheduler (monitors validation accuracy)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',       # Maximize validation accuracy
            factor=0.5,       # Reduce LR by half
            patience=3,       # Wait 3 epochs without improvement
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",  # Track validation accuracy
                "frequency": 1          # Check every epoch
            }
        }