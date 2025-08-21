from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.models import CNN
from src.data.dataset import *

def train_cnn(config, train_dataloader, val_dataloader, test_dataloader):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True

    # Define model
    model = CNN(config)
    model = model.to(device)

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

    print(f"Class counts: {class_counts}, pos_weight={pos_weight.item():.3f}")
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
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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

    if total == 0:
        return float("inf"), 0.0

    return running_loss / total, correct / total

if __name__ == "__main__":
    # Clear GPU memory cache to prevent out-of-memory errors
    torch.cuda.empty_cache()

    # Define Config
    config = configs("cnn")

    # Define model
    model = CNN(config)

    dataloaders = prepareCombinedDataset()

    train_dataloader, val_dataloader, test_dataloader = dataloaders["thermal"]

    # Train model
    trained_model = train_cnn(config, train_dataloader, val_dataloader, test_dataloader)

    print("Finished")