from pathlib import Path
from joblib import Parallel, delayed
import json
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from torchvision.transforms import v2
from src.utils import grayscale_loader

class ProcessedThermalDataset(Dataset):
    """PyTorch Dataset for loading processed thermal images from directory structure."""
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform, loader=grayscale_loader)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def targets(self):
        return self.dataset.targets

class ThermalPreprocessor:
    """Handles processing & saving augmented images with GPU acceleration."""
    def __init__(self,
                 raw_data_dir,
                 processed_dir,
                 classes=('no_fire', 'fire'),
                 train_augmented_multiplicity=4,
                 split_ratios=(0.7, 0.15, 0.15),
                 random_state=42):

        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.classes = list(classes)
        self.train_augmented_multiplicity = train_augmented_multiplicity
        self.split_ratios = split_ratios
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Common CPU-to-GPU preprocessing steps
        common = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(num_output_channels=1),
        ]

        # GPU transforms
        self.augment_transform = torch.compile(v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0)),
        ] + common))

        self.test_transform = torch.compile(v2.Compose([
            v2.Resize(size=(224, 224), antialias=True),
        ] + common))

    def _load_image_paths(self):
        """Load image paths & labels."""
        image_paths, labels = [], []
        for label, class_name in enumerate(self.classes):
            class_dir = self.raw_data_dir / class_name
            if class_dir.is_dir():
                for f in class_dir.iterdir():
                    if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                        image_paths.append(f)
                        labels.append(label)
        return image_paths, labels

    def _process_and_save(self, img_path, label, split_name, multiplicity):
        """Process a single image and save augmented versions."""
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Move to GPU tensor
        img_tensor = v2.ToImage()(img).to(self.device, dtype=torch.float32) / 255.0

        class_name = self.classes[label]
        split_dir = self.processed_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for aug_idx in range(multiplicity):
            transform = self.augment_transform if split_name == 'train' else self.test_transform
            transformed = transform(img_tensor)  # GPU augmentations

            # Move back to CPU for saving
            img_cpu = (transformed.clamp(0, 1) * 255).byte().cpu()
            fname = img_path.stem + f"_aug{aug_idx}" + img_path.suffix
            v2.ToPILImage()(img_cpu).save(split_dir / fname)

    def process_dataset(self):
        """Process dataset & save images, caching stats."""
        image_paths, labels = self._load_image_paths()
        if not image_paths:
            raise ValueError("No images found in raw data directories.")

        if labels.count(1) == 0 or labels.count(0) == 0:
            raise ValueError("Dataset missing one class! Cannot perform stratified split.")

        # Split dataset
        train_img, temp_img, train_lbl, temp_lbl = train_test_split(
            image_paths, labels,
            test_size=1 - self.split_ratios[0],
            random_state=self.random_state,
            stratify=labels
        )
        val_size = self.split_ratios[1] / (self.split_ratios[1] + self.split_ratios[2])
        val_img, test_img, val_lbl, test_lbl = train_test_split(
            temp_img, temp_lbl,
            test_size=1 - val_size,
            random_state=self.random_state,
            stratify=temp_lbl
        )

        splits = {
            'train': (train_img, train_lbl, self.train_augmented_multiplicity),
            'val': (val_img, val_lbl, 1),
            'test': (test_img, test_lbl, 1)
        }

        # Parallel processing
        for split_name, (img_paths, lbls, multiplicity) in splits.items():
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(self._process_and_save)(img_path, label, split_name, multiplicity)
                for img_path, label in zip(img_paths, lbls)
            )

        # Cache stats
        stats = {
            'total_images': len(image_paths),
            'split_counts': {k: len(v[0]) for k, v in splits.items()},
            'class_distribution': {
                'train': {cls: train_lbl.count(i) for i, cls in enumerate(self.classes)},
                'val': {cls: val_lbl.count(i) for i, cls in enumerate(self.classes)},
                'test': {cls: test_lbl.count(i) for i, cls in enumerate(self.classes)}
            }
        }
        with open(self.processed_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

class ThermalDataset:
    def __init__(self,
                 raw_data_dir='../../dataset/raw/thermal',
                 processed_dir='../../dataset/thermal_processed',
                 train_augmented_multiplicity=4,
                 batch_size=16,
                 regenerate=False,
                 split_ratios=(0.7, 0.15, 0.15),
                 random_state=42):

        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size

        self.loading_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5])
        ])

        if regenerate or not self.processed_dir.exists():
            preprocessor = ThermalPreprocessor(
                raw_data_dir=raw_data_dir,
                processed_dir=processed_dir,
                train_augmented_multiplicity=train_augmented_multiplicity,
                split_ratios=split_ratios,
                random_state=random_state
            )
            preprocessor.process_dataset()

        self.setup()

    def setup(self):
        self.train_dataset = ProcessedThermalDataset(self.processed_dir / 'train', transform=self.loading_transform)
        self.val_dataset = ProcessedThermalDataset(self.processed_dir / 'val', transform=self.loading_transform)
        self.test_dataset = ProcessedThermalDataset(self.processed_dir / 'test', transform=self.loading_transform)

        self.train_dataloaderCNN = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloaderCNN = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloaderCNN = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
