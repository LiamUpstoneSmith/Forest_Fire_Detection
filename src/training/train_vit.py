import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from src.data.dataset import *
from src.models import ViT
from src.config import configs

def train_vit(config, train_dataloader, val_dataloader, test_dataloader):
    """
    Train ViT model with explicit dataloaders instead of a LightningDataModule.
    """
    model = ViT(config)

    checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best_feature_extractor",
    )

    early_stop = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
    )

    trainer = pl.Trainer(
        max_epochs=config.get("num_epochs"),
        callbacks=[checkpoint, early_stop],
        accelerator="cpu",
        devices="auto",
        log_every_n_steps=10,
    )

    # Training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Load best model for evaluation
    best_path = checkpoint.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"Loading best model from checkpoint: {best_path}")
        best_model = ViT.load_from_checkpoint(best_path, config=config)
    else:
        print("WARNING: No best checkpoint found. Using last-epoch model for testing.")
        best_model = model

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    best_model.to(device)

    # Lightning test loop (optional, still works)
    test_result = trainer.test(best_model, dataloaders=test_dataloader)
    print(f"\nLightning Test Accuracy: {test_result[0]['test_acc']:.2f}")

    # Save checkpoint
    save_path = config.get("save_FE_path")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if best_path and os.path.exists(best_path):
            shutil.copy2(best_path, save_path)
        else:
            trainer.save_checkpoint(save_path)

    return best_model, trainer


if __name__ == "__main__":
    torch.cuda.empty_cache()
    dataloaders = prepareCombinedDataset()

    # Access RGB dataloaders
    rgb_train, rgb_val, rgb_test = dataloaders["rgb"]

    config = configs("vit")

    trained_model, trainer = train_vit(config, rgb_train, rgb_val, rgb_test)
    print("Finished")
