import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint , EarlyStopping
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
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    # ✅ Pass loaders directly
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_path = checkpoint.best_model_path
    if not best_path or not os.path.exists(best_path):
        print("WARNING: No best checkpoint found. Using last-epoch model for testing.")
        test_result = trainer.test(model, dataloaders=test_dataloader)
    else:
        best_model = ViT.load_from_checkpoint(best_path, config=config)
        test_result = trainer.test(best_model, dataloaders=test_dataloader)

    print(f"Test Accuracy: {test_result[0]['test_acc']:.2f}")

    save_path = config.get("save_FE_path")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if best_path and os.path.exists(best_path):
            shutil.copy2(best_path, save_path)  # save the checkpoint
        else:
            trainer.save_checkpoint(save_path)  # fallback

    return model, trainer


if __name__ == "__main__":
    # Clear GPU memory cache
    torch.cuda.empty_cache()

    # Define Dataloaders
    dataloaders = prepareCombinedDataset()

    # Access thermal dataloaders
    rgb_train, rgb_val, rgb_test = dataloaders["rgb"]

    # Define Config
    config = configs("vit")

    # train Model
    trained_model, trainer, data = train_vit(config, rgb_train, rgb_val, rgb_test)

    print("Finished")