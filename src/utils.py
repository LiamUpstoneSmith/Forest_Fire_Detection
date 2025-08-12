import os
import torch
from PIL import Image
from torchvision import transforms


def compute_class_weights_from_folder(folder: str):
    """
    Count examples per class in folder structure (class subfolders)
    and return pos_weight for BCEWithLogitsLoss (pos_weight = negative_count / positive_count)
    Expects two classes where class name containing 'fire' is positive; otherwise uses order.
    """
    classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    counts = []
    for c in classes:
        cnt = 0
        p = os.path.join(folder, c)
        for f in os.listdir(p):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                cnt += 1
        counts.append(cnt)
    # simple safe fallback
    if len(counts) < 2:
        return 1.0
    # try to detect which class is positive (contains "fire" in name)
    pos_idx = None
    for i, c in enumerate(classes):
        if "fire" in c.lower():
            pos_idx = i
            break
    if pos_idx is None:
        pos_idx = 0  # assume first is positive if unsure
    pos_count = counts[pos_idx]
    neg_count = sum(counts) - pos_count
    if pos_count == 0:
        return 1.0
    return float(neg_count / pos_count)


# image transform helpers for prediction
def get_vit_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def get_thermal_transform(image_size: int = 224):
    # thermal images loaded as RGB (converted)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

def load_image_as_tensor(path: str, transform):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return transform(img).unsqueeze(0)  # add batch dim

def save_checkpoint(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# Custom grayscale loader function
def grayscale_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
