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
    def __init__(self, vit_extractor, cnn_extractor, config=None, notebook=False):
        """
        Multimodal fusion model combining ViT (RGB) and CNN (thermal) features.
        """
        self.notebook = notebook
        super().__init__()

        # Configuration setup
        self.config = config

        # Visualization settings
        self.visuals = config.get("visuals")
        self.save_hyperparameters(ignore=["vit_extractor", "cnn_extractor"])

        # Feature extractors (frozen)
        self.vit_extractor = vit_extractor
        self.cnn_extractor = cnn_extractor
        for p in self.vit_extractor.parameters():
            p.requires_grad = False
        for p in self.cnn_extractor.parameters():
            p.requires_grad = False

        # Fusion classifier network
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, self.config["hidden_dim"]),
            nn.ReLU(),
            nn.BatchNorm1d(self.config["hidden_dim"]),
            nn.Dropout(self.config["dropout_rate"]),
            nn.Linear(self.config["hidden_dim"], 1)
        )

        # Register a buffer for pos_weight so device/dtype always match logits
        self.register_buffer("pos_weight", torch.tensor(self.config["pos_weight"], dtype=torch.float))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # Core metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

        # Advanced metrics for visualization
        if self.visuals:
            self.train_f1 = F1Score(task='binary')
            self.val_f1 = F1Score(task='binary')
            self.test_f1 = F1Score(task='binary')

            self.train_precision = Precision(task='binary')
            self.val_precision = Precision(task='binary')
            self.test_precision = Precision(task='binary')

            self.train_recall = Recall(task='binary')
            self.val_recall = Recall(task='binary')
            self.test_recall = Recall(task='binary')

        # Storage for visualizations and histories
        self.test_preds, self.test_targets, self.test_probs = [], [], []
        self.train_acc_history, self.val_acc_history, self.test_acc_history = [], [], []
        self.train_loss_history, self.val_loss_history, self.test_loss_history = [], [], []


    def forward(self, vit_input, cnn_input):
        """Forward pass through both feature extractors and fusion classifier"""
        vit_features = self.vit_extractor(vit_input)   # shape: (B, 768)
        cnn_features = self.cnn_extractor(cnn_input)   # shape: (B, 512)
        combined = torch.cat((vit_features, cnn_features), dim=1)
        logits = self.classifier(combined).squeeze(1)  # shape: (B,)
        return logits

    def training_step(self, batch, batch_idx):
        vit_imgs, cnn_imgs, labels = batch
        logits = self(vit_imgs, cnn_imgs)

        # BCEWithLogitsLoss expects float targets
        labels_float = labels.float()
        loss = self.criterion(logits, labels_float)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5)

        # Metrics prefer int/bool targets
        labels_int = labels.long()
        self.train_acc(preds, labels_int)
        if self.visuals:
            self.train_f1(preds, labels_int)
            self.train_precision(preds, labels_int)
            self.train_recall(preds, labels_int)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        if self.visuals:
            self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
            self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        self.train_acc.reset()
        if self.visuals:
            self.train_f1.reset()
            self.train_precision.reset()
            self.train_recall.reset()

    def on_validation_epoch_start(self):
        self.val_acc.reset()
        if self.visuals:
            self.val_f1.reset()
            self.val_precision.reset()
            self.val_recall.reset()

    def validation_step(self, batch, batch_idx):
        vit_imgs, cnn_imgs, labels = batch
        logits = self(vit_imgs, cnn_imgs)

        labels_float = labels.float()
        loss = self.criterion(logits, labels_float)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5)
        labels_int = labels.long()

        self.val_acc(preds, labels_int)
        if self.visuals:
            self.val_f1(preds, labels_int)
            self.val_precision(preds, labels_int)
            self.val_recall(preds, labels_int)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.visuals:
            self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_precision", self.val_precision, on_epoch=True, sync_dist=True)
            self.log("val_recall", self.val_recall, on_epoch=True, sync_dist=True)
        return loss



    def test_step(self, batch, batch_idx):
        vit_imgs, cnn_imgs, labels = batch
        logits = self(vit_imgs, cnn_imgs)

        labels_float = labels.float()
        loss = self.criterion(logits, labels_float)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5)
        labels_int = labels.long()

        self.test_acc(preds, labels_int)
        if self.visuals:
            self.test_f1(preds, labels_int)
            self.test_precision(preds, labels_int)
            self.test_recall(preds, labels_int)

            self.test_preds.append(preds.detach().cpu())
            self.test_targets.append(labels_int.detach().cpu())
            self.test_probs.append(probs.detach().cpu())

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        if self.visuals:
            self.log("test_f1", self.test_f1, on_epoch=True)
            self.log("test_precision", self.test_precision, on_epoch=True)
            self.log("test_recall", self.test_recall, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """Record training/validation accuracy and loss at epoch end"""
        metrics = self.trainer.callback_metrics

        # Record accuracy
        if metrics.get("train_acc") is not None:
            self.train_acc_history.append(metrics["train_acc"].item())
        if metrics.get("val_acc") is not None:
            self.val_acc_history.append(metrics["val_acc"].item())

        # Record loss
        if metrics.get("train_loss_epoch") is not None:
            self.train_loss_history.append(metrics["train_loss_epoch"].item())
        elif metrics.get("train_loss") is not None:  # fallback
            self.train_loss_history.append(metrics["train_loss"].item())

        if metrics.get("val_loss") is not None:
            self.val_loss_history.append(metrics["val_loss"].item())

    def on_train_end(self):
        """Generate accuracy & loss history plots after training completes"""
        if self.visuals:
            if self.train_acc_history and self.val_acc_history:
                self.plot_accuracy_history()
            if self.train_loss_history and self.val_loss_history:
                self.plot_convergence_history()

    def plot_convergence_history(self):
        """Plot training vs validation loss & accuracy for convergence check"""
        epochs = range(1, len(self.train_acc_history) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Loss plot ---
        axes[0].plot(epochs, self.train_loss_history, 'bo-', label='Training Loss')
        axes[0].plot(epochs, self.val_loss_history, 'ro-', label='Validation Loss')
        axes[0].set_title('Training vs Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # --- Accuracy plot ---
        axes[1].plot(epochs, self.train_acc_history, 'bo-', label='Training Accuracy')
        axes[1].plot(epochs, self.val_acc_history, 'go-', label='Validation Accuracy')
        axes[1].set_title('Training vs Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle("Model Convergence")
        plt.tight_layout()

        if self.notebook:
            plt.savefig("figures/convergence_history.png")
        else:
            plt.savefig("../saved/figures/convergence_history.png")

        plt.show()

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
                # Save test accuracy/loss history for convergence comparison
                if hasattr(self, "trainer"):
                    metrics = self.trainer.callback_metrics
                    if metrics.get("test_acc") is not None:
                        self.test_acc_history.append(metrics["test_acc"].item())
                    if metrics.get("test_loss") is not None:
                        self.test_loss_history.append(metrics["test_loss"].item())

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

        # 6. loss curves
        epochs = range(1, len(self.train_loss_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_loss_history, 'bo-', label="Training Loss")
        if self.val_loss_history:
            plt.plot(epochs, self.val_loss_history, 'ro-', label="Validation Loss")
        if self.test_loss_history:
            # Plot as flat line since test runs once
            plt.axhline(y=self.test_loss_history[-1], color='g', linestyle='--', label="Test Loss")
        plt.title("Loss Curves (Train/Val/Test)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        if self.notebook:
            plt.savefig("figures/loss_curves.png")
        else:
            plt.savefig("../saved/figures/loss_curves.png")
        plt.show()

        # 7. accuracy curves
        epochs = range(1, len(self.train_acc_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_acc_history, 'bo-', label="Training Accuracy")
        if self.val_acc_history:
            plt.plot(epochs, self.val_acc_history, 'go-', label="Validation Accuracy")
        if self.test_acc_history:
            plt.axhline(y=self.test_acc_history[-1], color='r', linestyle='--', label="Test Accuracy")
        plt.title("Accuracy Curves (Train/Val/Test)")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        if self.notebook:
            plt.savefig("figures/accuracy_curves.png")
        else:
            plt.savefig("../saved/figures/accuracy_curves.png")
        plt.show()

    # 8. Precision Recall comparison (train v test)
    def plot_precision_recall_comparison(self, y_true_train, y_probs_train, y_true_test, y_probs_test):
        """Compare Precision-Recall curves for train vs test sets"""
        from sklearn.metrics import precision_recall_curve, auc

        precision_train, recall_train, _ = precision_recall_curve(y_true_train, y_probs_train)
        precision_test, recall_test, _ = precision_recall_curve(y_true_test, y_probs_test)

        auc_train = auc(recall_train, precision_train)
        auc_test = auc(recall_test, precision_test)

        plt.figure(figsize=(10, 6))
        plt.plot(recall_train, precision_train, label=f"Train (AUC={auc_train:.2f})", color="blue")
        plt.plot(recall_test, precision_test, label=f"Test (AUC={auc_test:.2f})", color="red")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves (Train vs Test)")
        plt.legend()
        plt.grid(True)
        if self.notebook:
            plt.savefig("figures/precision_recall_train_vs_test.png")
        else:
            plt.savefig("../saved/figures/precision_recall_train_vs_test.png")
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