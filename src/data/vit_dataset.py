import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split

class RGBDataset(pl.LightningDataModule):
    def __init__(self, data_dir, augmented_dir,
                 batch_size, num_workers, train_augmented_multiplicity,
                 regenerate=False):
        super().__init__()
        self.data_dir = data_dir
        self.augmented_dir = augmented_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_augmented_multiplicity = train_augmented_multiplicity
        self.regenerate = regenerate
        self.classes = None
        self.transform_load = None
        self.augment_save = None
        self.transform_save = None

    def setup(self, stage=None):
        # Define saving transforms
        self.augment_save = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

        self.transform_save = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        # Define loading transform
        self.transform_load = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load base dataset
        base_dataset = datasets.ImageFolder(root=self.data_dir, transform=None)
        self.classes = base_dataset.classes
        targets = [s[1] for s in base_dataset.samples]

        # Create splits
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(skf.split(np.zeros(len(targets)), targets))
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

        # Save augmented datasets
        self._save_augmented_dataset(base_dataset, train_idx, val_idx, test_idx)

        # Create datasets with loading transform
        self.train_dataset = datasets.ImageFolder(
            root=os.path.join(self.augmented_dir, "train"),
            transform=self.transform_load
        )
        self.val_dataset = datasets.ImageFolder(
            root=os.path.join(self.augmented_dir, "val"),
            transform=self.transform_load
        )
        self.test_dataset = datasets.ImageFolder(
            root=os.path.join(self.augmented_dir, "test"),
            transform=self.transform_load
        )

    def _save_augmented_dataset(self, base_dataset, train_idx, val_idx, test_idx):
        """Saves processed images to filesystem with robust directory checks"""
        splits = {
            "train": (train_idx, self.augment_save, self.train_augmented_multiplicity),
            "val": (val_idx, self.transform_save, 1),
            "test": (test_idx, self.transform_save, 1)
        }

        # Create root augmented directory if needed
        os.makedirs(self.augmented_dir, exist_ok=True)

        # Check if we should skip regeneration
        if not self.regenerate:
            all_dirs_exist = True
            for split_name in splits:
                split_dir = os.path.join(self.augmented_dir, split_name)
                if not os.path.exists(split_dir):
                    all_dirs_exist = False
                    break

                # Check class subdirectories
                for class_name in base_dataset.classes:
                    class_dir = os.path.join(split_dir, class_name)
                    if not os.path.exists(class_dir) or len(os.listdir(class_dir)) == 0:
                        all_dirs_exist = False
                        break

            if all_dirs_exist:
                return

        for split_name, (indices, transform, multiplicity) in splits.items():
            split_dir = os.path.join(self.augmented_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)  # Ensure split directory exists

            # Create class directories
            for class_name in base_dataset.classes:
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

            for idx in indices:
                img, label = base_dataset[idx]
                class_name = base_dataset.classes[label]

                for aug_idx in range(multiplicity):
                    # Apply transformations
                    transformed_img = transform(img)

                    # Ensure we have PIL image for saving
                    if not isinstance(transformed_img, Image.Image):
                        transformed_img = transforms.ToPILImage()(transformed_img)

                    # Save with unique filename
                    original_name = os.path.splitext(os.path.basename(base_dataset.samples[idx][0]))[0]
                    save_path = os.path.join(split_dir, class_name, f"{original_name}_aug{aug_idx}.jpg")
                    transformed_img.save(save_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )