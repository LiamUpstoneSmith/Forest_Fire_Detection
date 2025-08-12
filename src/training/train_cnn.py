from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data import ThermalDataset
from src.models import CNN
from src.config import configs

def train_cnn(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data_module = ThermalDataset(
        raw_data_dir=config.get("data_dir"),
        processed_dir=config.get("processed_dir"),
        train_augmented_multiplicity=config.get("train_augmented_multiplicity"),
        batch_size=config.get("batch_size"),
        regenerate=config.get("regenerate"),
    )

    train_dataloader = data_module.train_dataloaderCNN
    val_dataloader = data_module.val_dataloaderCNN

    num_epochs = config.get("num_epochs")
    lr = config.get("lr")
    max_grad_norm = config.get("max_grad_norm")

    # Calculate positive class weight once
    train_targets = train_dataloader.dataset.targets
    class_counts = [
        np.sum(np.array(train_targets) == 0),  # no_fire
        np.sum(np.array(train_targets) == 1)   # fire
    ]
    pos_weight = torch.tensor(
        [class_counts[0] / max(1, class_counts[1])]
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc, best_model_state = 0.0, None



    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(
            train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=True
        )

        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)
            train_acc = correct / total

            progress_bar.set_postfix(loss=loss.item(), acc=train_acc)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_loss)
        epoch_train_acc = correct / total

        # Validation each epoch
        val_loss, val_acc = validate_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Save best model
    if best_model_state:
        torch.save(best_model_state, '../saved/features/CNN_best_model_weights.pth')
        model.load_state_dict(best_model_state)

    return model

def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0, 0
    return running_loss / total, correct / total

def save_feature_extractor_cnn(model, path):
    state_dict = model.state_dict()
    torch.save(state_dict, path)

if __name__ == "__main__":
    # Clear GPU memory cache to prevent out-of-memory errors
    torch.cuda.empty_cache()

    # Define Config
    config = configs("cnn")

    # Define model
    model = CNN(config)

    # Train model
    trained_model = train_cnn(model)

    # Save model
    save_feature_extractor_cnn(trained_model, config.get("save_FE_path"))

    print("Finished")

