from typing import List
from src.config import configs
import os, re, random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold
import numpy as np
import warnings

# --- Dataset Wrappers ---
class RGBOnlyDataset(Dataset):
    def __init__(self, fusion_dataset):
        self.base = fusion_dataset
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        vit_t, _, lbl = self.base[idx]
        return vit_t, lbl

class ThermalOnlyDataset(Dataset):
    def __init__(self, fusion_dataset):
        self.base = fusion_dataset
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        _, cnn_t, lbl = self.base[idx]
        return cnn_t, lbl

# --- Synchronized Augmentations ---
class PairedRandomTransform:
    def __init__(self, config, train_mode=True):
        self.img_size = tuple(config.get("img_size", (224,224)))
        self.rot = config.get("rotation", 15)
        self.hflip_p = config.get("hflip_p", 0.5)
        self.vflip_p = config.get("vflip_p", 0.5)
        self.color_jitter = transforms.ColorJitter(0.2,0.2,0.2) if train_mode else None
        self.train_mode = train_mode

        # base ops
        self.resize_rgb  = transforms.Resize(self.img_size)
        self.resize_thr  = v2.Resize(self.img_size)
        self.to_tensor_rgb = transforms.ToTensor()
        self.to_tensor_thr = v2.ToImage()

    def __call__(self, rgb_img, thr_img):
        if self.train_mode:
            do_h = random.random() < self.hflip_p
            do_v = random.random() < self.vflip_p
            angle = random.uniform(-self.rot, self.rot)
        else:
            do_h, do_v, angle = False, False, 0

        rgb = self.resize_rgb(rgb_img).rotate(angle)
        thr = self.resize_thr(thr_img).rotate(angle)
        if do_h:
            rgb = transforms.functional.hflip(rgb)
            thr = v2.functional.hflip(thr)
        if do_v:
            rgb = transforms.functional.vflip(rgb)
            thr = v2.functional.vflip(thr)

        if self.color_jitter:
            rgb = self.color_jitter(rgb)

        rgb = self.to_tensor_rgb(rgb)  # float in [0,1]

        thr = self.to_tensor_thr(thr)  # uint8
        thr = v2.functional.rgb_to_grayscale(thr, num_output_channels=1)
        thr = v2.ToDtype(torch.float32, scale=True)(thr)  # float [0,1]

        rgb = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(rgb)
        thr = v2.Normalize([0.5], [0.5])(thr)
        return rgb, thr

# --- Fusion Dataset ---
class FusionDataset(Dataset):
    def __init__(self, vit_paths, cnn_paths, labels, paired_transform, train_mode=True):
        assert len(vit_paths) == len(cnn_paths) == len(labels)
        self.vit_paths = list(vit_paths)
        self.cnn_paths = list(cnn_paths)
        self.labels = list(labels)
        self.paired_transform = paired_transform
        self.train_mode = bool(train_mode)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        vit_p = self.vit_paths[idx]
        cnn_p = self.cnn_paths[idx]
        lbl = int(self.labels[idx])

        with Image.open(vit_p) as vit_img, Image.open(cnn_p) as cnn_img:
            vit_img = vit_img.convert("RGB")
            cnn_img = cnn_img.convert("RGB")
            vit_t, cnn_t = self.paired_transform(vit_img, cnn_img)

        return vit_t, cnn_t, torch.tensor(lbl, dtype=torch.long)

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, multiplier=1):
        self.base = base_dataset
        self.multiplier = multiplier

    def __len__(self):
        return len(self.base) * self.multiplier

    def __getitem__(self, idx):
        # Cycle through the base dataset
        base_idx = idx % len(self.base)
        return self.base[base_idx]

# --- Helpers ---
def _parse_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    m = re.match(r"(incident\d+)_frame(\d+)", stem)
    return (m.group(1), m.group(2)) if m else (stem, None)

def _scan_dir(dir_path, label):
    mapping = {}
    if not dir_path or not os.path.isdir(dir_path): return mapping
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(('.jpg','.jpeg','.png')): continue
        group, frame = _parse_filename(fn)
        mapping.setdefault(group,{})[frame] = {"path": os.path.join(dir_path, fn), "label": label}
    return mapping

def print_split_statistics(loader, split_name: str):
    all_labels = []
    for batch in loader:
        labels = batch[-1]
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        all_labels.extend(labels)
    all_labels = np.array(all_labels, dtype=int)
    total = len(all_labels)
    class_counts = np.bincount(all_labels, minlength=2)
    print(f"\n--- {split_name} Split ---")
    print(f"Total samples: {total}")
    for cls, count in enumerate(class_counts):
        pct = 100.0*count/total if total>0 else 0.0
        print(f"Class {cls}: {count} ({pct:.1f}%)")

# --- Main ---
def prepareCombinedDataset(config=None):
    if config is None:
        config = configs("data")
    vit_fire_dir = config["vit_fire_dir"]
    vit_non_fire_dir = config["vit_non_fire_dir"]
    cnn_fire_dir = config["cnn_fire_dir"]
    cnn_non_fire_dir = config["cnn_non_fire_dir"]
    batch_size = int(config.get("batch_size", 16))


    vit_map, cnn_map = {}, {}
    for src_dir,lbl in [(vit_fire_dir,1),(vit_non_fire_dir,0)]: vit_map.update(_scan_dir(src_dir,lbl))
    for src_dir,lbl in [(cnn_fire_dir,1),(cnn_non_fire_dir,0)]: cnn_map.update(_scan_dir(src_dir,lbl))

    matched, groups = [], []
    for group in set(vit_map)&set(cnn_map):
        for frame in set(vit_map[group])&set(cnn_map[group]):
            v,c = vit_map[group][frame], cnn_map[group][frame]
            if v["label"]==c["label"]:
                matched.append((v["path"],c["path"],v["label"],group))
                groups.append(group)
    if not matched: raise ValueError("No matched pairs found.")

    combined = list(matched)
    random.shuffle(combined)
    vit_paths_all,cnn_paths_all,labels_all,groups_all = zip(*combined)
    labels_all,groups_all = list(labels_all),list(groups_all)

    dataset_size = len(labels_all)
    if dataset_size<200:
        warnings.warn(f"Dataset only {dataset_size} samples.")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx,temp_idx = next(skf.split(range(dataset_size),labels_all))
        mid=len(temp_idx)//2
        val_idx,test_idx=temp_idx[:mid],temp_idx[mid:]
    else:
        train_idx,temp_idx = train_test_split(range(len(labels_all)),
            test_size=0.3,stratify=labels_all,random_state=42)
        temp_labels=[labels_all[i] for i in temp_idx]
        temp_groups=[groups_all[i] for i in temp_idx]
        sgkf=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=42)
        val_idx,test_idx = next(sgkf.split(temp_idx,temp_labels,groups=temp_groups))
        val_idx=[temp_idx[i] for i in val_idx]; test_idx=[temp_idx[i] for i in test_idx]

    def subset(paths,idxs): return [paths[i] for i in idxs]

    vit_train,cnn_train=[subset(vit_paths_all,train_idx),subset(cnn_paths_all,train_idx)]
    train_labels=[labels_all[i] for i in train_idx]
    vit_val,cnn_val=[subset(vit_paths_all,val_idx),subset(cnn_paths_all,val_idx)]
    val_labels=[labels_all[i] for i in val_idx]
    vit_test,cnn_test=[subset(vit_paths_all,test_idx),subset(cnn_paths_all,test_idx)]
    test_labels=[labels_all[i] for i in test_idx]

    train_tf = PairedRandomTransform(config,train_mode=True)
    eval_tf  = PairedRandomTransform(config,train_mode=False)

    base_train_dataset = FusionDataset(vit_train,cnn_train,train_labels,train_tf,train_mode=True)
    train_dataset = AugmentedDataset(base_train_dataset, multiplier=int(config.get("multiplier")))
    val_dataset   = FusionDataset(vit_val,cnn_val,val_labels,eval_tf,train_mode=False)
    base_test_dataset = FusionDataset(vit_test,cnn_test, test_labels, train_tf, train_mode=True)
    test_dataset = AugmentedDataset(base_test_dataset, multiplier=10)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader   = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    test_loader  = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)

    # RGB-only
    rgb_train_loader=DataLoader(RGBOnlyDataset(train_dataset),batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    rgb_val_loader  =DataLoader(RGBOnlyDataset(val_dataset),batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    rgb_test_loader =DataLoader(RGBOnlyDataset(test_dataset),batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)

    # Thermal-only
    thr_train_loader=DataLoader(ThermalOnlyDataset(train_dataset),batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    thr_val_loader  =DataLoader(ThermalOnlyDataset(val_dataset),batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    thr_test_loader =DataLoader(ThermalOnlyDataset(test_dataset),batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)

    print_split_statistics(train_loader,"Train")
    print_split_statistics(val_loader,"Validation")
    print_split_statistics(test_loader,"Test")

    # compute class weights for loss
    train_counts=np.bincount(train_labels,minlength=2)
    total=sum(train_counts)
    weights=[total/(2*c) if c>0 else 0 for c in train_counts]
    class_weights=torch.tensor(weights,dtype=torch.float32)

    return {
        "fusion":  (train_loader,val_loader,test_loader),
        "rgb":     (rgb_train_loader,rgb_val_loader,rgb_test_loader),
        "thermal": (thr_train_loader,thr_val_loader,thr_test_loader),
        "class_weights": class_weights
    }

if __name__=="__main__":
    dataloaders=prepareCombinedDataset()
