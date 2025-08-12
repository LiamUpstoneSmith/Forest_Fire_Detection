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
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True

    model = model.to(device)

    print("Generating Dataset...")
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

    # Temporary classifier for training
    classifier_head = nn.Linear(512, 1).to(device)

    # Class imbalance handling
    train_labels = [label for _, label in train_dataloader.dataset]
    class_counts = [
        train_labels.count(0),  # no_fire
        train_labels.count(1)   # fire
    ]
    pos_weight = torch.tensor(
        [class_counts[0] / max(1, class_counts[1])]
    ).to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(list(model.parameters()) + list(classifier_head.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        classifier_head.train()

        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
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
        train_acc = correct / total

        val_loss, val_acc = validate_model(model, classifier_head, val_dataloader, criterion, device)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Save only the feature extractor (not the classifier)
    if best_model_state:
        torch.save(best_model_state, config.get("save_FE_path"))
        model.load_state_dict(best_model_state)

    return model

def validate_model(model, classifier_head, dataloader, criterion, device):
    model.eval()
    classifier_head.eval()

    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            with torch.cuda.amp.autocast():
                features = model(images)
                outputs = classifier_head(features).squeeze()
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

if __name__ == "__main__":
    # Clear GPU memory cache to prevent out-of-memory errors
    torch.cuda.empty_cache()

    # Define Config
    config = configs("cnn")

    # Define model
    model = CNN(config)

    # Train model
    trained_model = train_cnn(model)

    print("Finished")

