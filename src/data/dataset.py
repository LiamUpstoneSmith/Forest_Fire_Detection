# 2.4
from typing import List
from src.config import configs
import os, re, random
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold
import numpy as np
import warnings

# --- Dataset Wrappers ---
class RGBOnlyDataset(Dataset):
    """Dataset wrapper for only RGB (ViT input)."""
    def __init__(self, fusion_dataset):
        self.base = fusion_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        vit_t, _, lbl = self.base[idx]
        return vit_t, lbl


class ThermalOnlyDataset(Dataset):
    """Dataset wrapper for only Thermal (CNN input)."""
    def __init__(self, fusion_dataset):
        self.base = fusion_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _, cnn_t, lbl = self.base[idx]
        return cnn_t, lbl


class FusionDataset(Dataset):
    """
    Pair-aligned dataset: each index returns (vit_tensor, cnn_tensor, label)
    Accepts transforms for train/eval mode (online augmentation).
    """
    def __init__(self,
                 vit_paths: List[str],
                 cnn_paths: List[str],
                 labels: List[int],
                 vit_train_transform,
                 vit_eval_transform,
                 cnn_train_transform,
                 cnn_eval_transform,
                 train_mode: bool = True):
        assert len(vit_paths) == len(cnn_paths) == len(labels), "Paths/labels lengths must match"
        self.vit_paths = list(vit_paths)
        self.cnn_paths = list(cnn_paths)
        self.labels = list(labels)
        self.vit_train_transform = vit_train_transform
        self.vit_eval_transform = vit_eval_transform
        self.cnn_train_transform = cnn_train_transform
        self.cnn_eval_transform = cnn_eval_transform
        self.train_mode = bool(train_mode)
        self.config = configs("fusion")

        # Validate data alignment
        if len(vit_paths) != len(cnn_paths) or len(vit_paths) != len(labels):
            raise ValueError("Input lengths mismatch: ViT paths, CNN paths and labels must be equal")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vit_p = self.vit_paths[idx]
        cnn_p = self.cnn_paths[idx]
        lbl = int(self.labels[idx])

        with Image.open(vit_p) as vit_img:
            vit_img = vit_img.convert("RGB")
            vit_t = self.vit_train_transform(vit_img) if self.train_mode else self.vit_eval_transform(vit_img)

        with Image.open(cnn_p) as cnn_img:
            # thermal might be RGB or grayscale on disk; transforms should handle conversions
            cnn_t = self.cnn_train_transform(cnn_img) if self.train_mode else self.cnn_eval_transform(cnn_img)

        return vit_t, cnn_t, torch.tensor(lbl, dtype=torch.float32)

def _parse_filename(filename):
    """
    Extract incident group and frame index from filename.
    Example: incident123_frame045_rgb.jpg -> group='incident123', frame='045'
    """
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    # Expect incident id + frame number
    m = re.match(r"(incident\d+)_frame(\d+)", stem)
    if m:
        return m.group(1), m.group(2)   # group key, frame index
    # fallback: whole stem as group
    return stem, None


def _scan_dir(dir_path, label):
    """Scan directory for images, return mapping of group -> {frame: (path,label)}"""
    mapping = {}
    if not dir_path or not os.path.isdir(dir_path):
        return mapping
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        group, frame = _parse_filename(fn)
        if group not in mapping:
            mapping[group] = {}
        mapping[group][frame] = {"path": os.path.join(dir_path, fn), "label": label}
    return mapping

def print_split_statistics(loader, split_name: str):
    """Utility to print dataset size and class distribution for a DataLoader."""
    if loader is None:
        print(f"{split_name} loader not provided.")
        return

    all_labels = []
    for batch in loader:
        # batch can be (vit, cnn, labels) or (img, labels) depending on dataset
        labels = batch[-1]
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        all_labels.extend(labels)

    all_labels = np.array(all_labels, dtype=int)
    total = len(all_labels)
    class_counts = np.bincount(all_labels, minlength=2)

    print(f"\n--- {split_name} Split ---")
    print(f"Total samples: {total}")
    for cls, count in enumerate(class_counts):
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"Class {cls}: {count} ({pct:.1f}%)")


def prepareCombinedDataset(config=None):

    if config is None:
        config = configs("fusion")

    vit_fire_dir = config.get("vit_fire_dir")
    vit_non_fire_dir = config.get("vit_non_fire_dir")
    cnn_fire_dir = config.get("cnn_fire_dir")
    cnn_non_fire_dir = config.get("cnn_non_fire_dir") or config.get("cnn_no_fire_dir")
    num_aug_copies = int(config.get("num_aug_copies", 0))
    batch_size = int(config.get("batch_size", 16))

    # --- Build maps by group ---
    vit_map = {}
    for src_dir, lbl in [(vit_fire_dir, 1), (vit_non_fire_dir, 0)]:
        vit_map.update(_scan_dir(src_dir, lbl))

    cnn_map = {}
    for src_dir, lbl in [(cnn_fire_dir, 1), (cnn_non_fire_dir, 0)]:
        cnn_map.update(_scan_dir(src_dir, lbl))

    # --- Match pairs by group & frame ---
    matched, groups = [], []
    for group in set(vit_map.keys()) & set(cnn_map.keys()):
        for frame in set(vit_map[group].keys()) & set(cnn_map[group].keys()):
            v = vit_map[group][frame]
            c = cnn_map[group][frame]
            if v["label"] == c["label"]:
                matched.append((v["path"], c["path"], v["label"], group))
                groups.append(group)

    if len(matched) == 0:
        raise ValueError("No matched pairs found. Check dataset directories & filename patterns.")

    combined = list(matched)
    random.shuffle(combined)
    vit_paths_all, cnn_paths_all, labels_all, groups_all = zip(*combined)
    labels_all, groups_all = list(labels_all), list(groups_all)

    # --- Handle small dataset instability ---
    dataset_size = len(labels_all)
    if dataset_size < 200:
        warnings.warn(
            f"Dataset has only {dataset_size} samples. Validation/test splits may be unstable. "
            "Consider using cross-validation."
        )
        # Optionally switch to 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Just pick the first fold for now (can loop in experiments)
        train_idx, temp_idx = next(skf.split(range(dataset_size), labels_all))
        # split temp into val/test (50/50)
        mid = len(temp_idx) // 2
        val_idx, test_idx = temp_idx[:mid], temp_idx[mid:]
    else:
        # --- Stratified Group Split (70/15/15) ---
        train_idx, temp_idx = train_test_split(
            range(len(labels_all)),
            test_size=0.3,
            stratify=labels_all,
            random_state=42
        )
        temp_labels = [labels_all[i] for i in temp_idx]
        temp_groups = [groups_all[i] for i in temp_idx]

        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
        val_idx, test_idx = next(sgkf.split(temp_idx, temp_labels, groups=temp_groups))
        val_idx = [temp_idx[i] for i in val_idx]
        test_idx = [temp_idx[i] for i in test_idx]

    # --- Helper ---
    def subset(paths, idxs):
        return [paths[i] for i in idxs]

    vit_train, cnn_train = subset(vit_paths_all, train_idx), subset(cnn_paths_all, train_idx)
    train_labels = [labels_all[i] for i in train_idx]

    vit_val, cnn_val = subset(vit_paths_all, val_idx), subset(cnn_paths_all, val_idx)
    val_labels = [labels_all[i] for i in val_idx]

    vit_test, cnn_test = subset(vit_paths_all, test_idx), subset(cnn_paths_all, test_idx)
    test_labels = [labels_all[i] for i in test_idx]

    # --- Class imbalance handling ---
    def check_class_balance(labels, split_name):
        counts = np.bincount(labels)
        total = len(labels)
        if len(counts) < 2:  # only one class present
            warnings.warn(f"{split_name} set contains only one class. Metrics may be invalid.")
        else:
            minority = min(counts) / total
            if minority < 0.1:
                warnings.warn(f"{split_name} set is imbalanced "
                              f"(class counts={counts.tolist()}). "
                              f"Consider oversampling or using stratified resampling.")

    check_class_balance(train_labels, "Train")
    check_class_balance(val_labels, "Validation")
    check_class_balance(test_labels, "Test")

    # --- Transforms ---
    vit_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    vit_eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cnn_train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(15),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=1),
        v2.Normalize([0.5], [0.5]),
    ])
    cnn_eval_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=1),
        v2.Normalize([0.5], [0.5]),
    ])

    # --- Dataset objects ---
    train_base = FusionDataset(
        vit_train, cnn_train, train_labels,
        vit_train_transform, vit_eval_transform,
        cnn_train_transform, cnn_eval_transform,
        train_mode=True
    )
    val_dataset = FusionDataset(
        vit_val, cnn_val, val_labels,
        vit_train_transform, vit_eval_transform,
        cnn_train_transform, cnn_eval_transform,
        train_mode=False
    )
    test_dataset = FusionDataset(
        vit_test, cnn_test, test_labels,
        vit_train_transform, vit_eval_transform,
        cnn_train_transform, cnn_eval_transform,
        train_mode=False
    )

    # --- Oversampling to multiply dataset by num_aug_copies ---
    num_aug_copies = int(config.get("num_aug_copies", 3))  # fallback to 3

    # Group indices by class
    class_to_indices = {cls: [i for i, lbl in enumerate(train_labels)] for cls in set(train_labels)}
    for cls in class_to_indices:
        class_to_indices[cls] = [i for i, lbl in enumerate(train_labels) if lbl == cls]

    # Compute target number of samples per class
    original_counts = {cls: len(idxs) for cls, idxs in class_to_indices.items()}
    target_count = max(original_counts.values()) * num_aug_copies

    balanced_indices = []

    for cls, idxs in class_to_indices.items():
        if len(idxs) == 0:
            continue
        repeat_factor = target_count // len(idxs)
        remainder = target_count % len(idxs)
        balanced_indices.extend(idxs * repeat_factor)
        balanced_indices.extend(random.choices(idxs, k=remainder))

    random.shuffle(balanced_indices)

    # These indices are already correct for train_base
    train_dataset = Subset(train_base, balanced_indices)
    train_labels_balanced = [train_labels[i] for i in balanced_indices]

    # --- Balanced sampler (optional, keeps per-class weighting) ---
    class_counts = torch.bincount(torch.tensor(train_labels_balanced))
    class_weights = 1. / class_counts.float()
    sample_weights = [class_weights[label] for label in train_labels_balanced]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # --- Dataloaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # --- Build RGB-only and Thermal-only datasets from the balanced fusion dataset ---
    rgb_train_dataset = RGBOnlyDataset(train_dataset)
    rgb_val_dataset = RGBOnlyDataset(val_dataset)
    rgb_test_dataset = RGBOnlyDataset(test_dataset)

    thermal_train_dataset = ThermalOnlyDataset(train_dataset)
    thermal_val_dataset = ThermalOnlyDataset(val_dataset)
    thermal_test_dataset = ThermalOnlyDataset(test_dataset)

    rgb_train_loader = DataLoader(rgb_train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True)
    rgb_val_loader = DataLoader(rgb_val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
    rgb_test_loader = DataLoader(rgb_test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)

    thermal_train_loader = DataLoader(thermal_train_dataset, batch_size=batch_size, sampler=sampler,
                                      num_workers=4, pin_memory=True)
    thermal_val_loader = DataLoader(thermal_val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)
    thermal_test_loader = DataLoader(thermal_test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)

    # dataset statistice
    print_split_statistics(train_loader, "Train")
    print_split_statistics(val_loader, "Validation")
    print_split_statistics(test_loader, "Test")

    return {
        "fusion":  (train_loader, val_loader, test_loader),
        "rgb":     (rgb_train_loader, rgb_val_loader, rgb_test_loader),
        "thermal": (thermal_train_loader, thermal_val_loader, thermal_test_loader)
    }


if __name__ == "__main__":
    config = configs("fusion")
    dataloaders = prepareCombinedDataset()

    train_dataloader, val_dataloader, test_dataloader = dataloaders["fusion"]


