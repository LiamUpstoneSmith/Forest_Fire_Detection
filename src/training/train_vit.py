import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import RGBDataset
from src.models import ViT
from src.config import configs

def train_vit(config):
    data = RGBDataset(
        data_dir=config.get("data_dir"),
        augmented_dir=config.get("augmented_dir"),
        batch_size=config.get("batch_size"),
        regenerate=config.get("regenerate"),
        train_augmented_multiplicity=config.get("train_augmented_multiplicity")
    )
    data.setup()

    model = ViT(config)
    checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best_feature_extractor"
    )

    trainer = pl.Trainer(
        max_epochs=config.get("num_epochs"),
        callbacks=[checkpoint],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, data)
    test_result = trainer.test(model, dataloaders=data.test_dataloader())
    print(f"Test Accuracy: {test_result[0]['test_acc']:.2f}")

    return model, trainer, data

if __name__ == "__main__":
    # Clear GPU memory cache
    torch.cuda.empty_cache()

    # Define Config
    config = configs("vit")

    # train Model
    trained_model, trainer, data = train_vit(config)

    # Save Model
    trainer.save_checkpoint(config.get("save_FE_path"))

    print("Finished")