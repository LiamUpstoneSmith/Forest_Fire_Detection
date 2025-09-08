import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.models import CNN
from src.data.dataset import *
from collections import Counter

def train_cnn(config, train_dataloader, val_dataloader, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True

    # Define model
    model = CNN(config)
    model = model.to(device)

    num_epochs = config.get("num_epochs")
    lr = config.get("lr")
    max_grad_norm = config.get("max_grad_norm")

    # Temporary classifier for training
    in_features = model.feature_refiner[-3].out_features  # 512 in your case
    classifier_head = nn.Linear(in_features, 1).to(device)

    # Class imbalance handling
    ds = train_dataloader.dataset  # ThermalOnlyDataset wrapping AugmentedDataset -> FusionDataset

    # Unwrap wrappers to get to the FusionDataset that stores labels
    fusion_ds = None
    # ThermalOnlyDataset(base=AugmentedDataset(base=FusionDataset))
    if hasattr(ds, "base") and hasattr(ds.base, "base") and hasattr(ds.base.base, "labels"):
        fusion_ds = ds.base.base
    # AugmentedDataset(base=FusionDataset)
    elif hasattr(ds, "base") and hasattr(ds.base, "labels"):
        fusion_ds = ds.base
    # Direct FusionDataset
    elif hasattr(ds, "labels"):
        fusion_ds = ds
    else:
        fusion_ds = None

    if fusion_ds is not None:
        # labels stored as ints in fusion_ds.labels
        train_labels = [int(x) for x in fusion_ds.labels]
        # if dataset is augmented, account for multiplier by scaling counts (optional)
        mult = 1
        if hasattr(ds, "base") and isinstance(ds.base, type(ds.base)) and hasattr(ds.base, "multiplier"):
            # ds.base is AugmentedDataset when ThermalOnlyDataset.base is AugmentedDataset
            try:
                mult = int(ds.base.multiplier)
            except Exception:
                mult = 1
        class_counts = [train_labels.count(0) * mult, train_labels.count(1) * mult]
    else:
        # Fallback (will trigger image I/O); still better than comprehension with tuple unpack.
        # This fallback is kept defensive; in normal runs fusion_ds should be found.
        train_labels = []
        for item in ds:
            # ThermalOnlyDataset returns (cnn_t, lbl)
            lbl = item[-1]
            try:
                train_labels.append(int(lbl))
            except Exception:
                train_labels.append(int(lbl.item() if hasattr(lbl, "item") else lbl))
        class_counts = [train_labels.count(0), train_labels.count(1)]

    pos_weight = torch.tensor([float(class_counts[0]) / max(1.0, float(class_counts[1]))], device=device)
    print(f"Class counts: {class_counts}, pos_weight={pos_weight.item():.3f}")

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(list(model.parameters()) + list(classifier_head.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    best_model_state = None

    # Track last-epoch train/val accuracy
    final_train_acc, final_val_acc = 0.0, 0.0

    for epoch in range(num_epochs):
        model.train()
        classifier_head.train()

        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                features = model(images)  # [B, feature_dim]
                outputs = classifier_head(features).squeeze()
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier_head.parameters()), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), acc=correct / total)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        final_train_acc = correct / total

        val_loss, final_val_acc = validate_model(model, classifier_head, val_dataloader, criterion, device)
        scheduler.step(final_val_acc)

        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_model_state = {
                "feature_extractor": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "classifier_head": {k: v.cpu().clone() for k, v in classifier_head.state_dict().items()}
            }

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state["feature_extractor"])
        classifier_head.load_state_dict(best_model_state["classifier_head"])
        torch.save(best_model_state["feature_extractor"], config.get("save_FE_path"))

    # --- Final Statistics ---
    print("\n=== Final Model Statistics ===")
    print(f"Training Accuracy   : {final_train_acc:.4f}")
    print(f"Validation Accuracy : {final_val_acc:.4f}")

    test_loss, test_acc = validate_model(model, classifier_head, test_dataloader, criterion, device)
    print(f"Test Accuracy       : {test_acc:.4f}")
    print("================================")

    return model

def validate_model(model, classifier_head, dataloader, criterion, device):
    model.eval()
    classifier_head.eval()

    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            with torch.amp.autocast('cuda'):
                features = model(images)
                outputs = classifier_head(features).squeeze()
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    if total == 0:
        return float("inf"), 0.0

    return running_loss / total, correct / total

if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = configs("cnn")
    dataloaders = prepareCombinedDataset()
    train_dataloader, val_dataloader, test_dataloader = dataloaders["thermal"]

    trained_model = train_cnn(config, train_dataloader, val_dataloader, test_dataloader)
    print("Finished")
