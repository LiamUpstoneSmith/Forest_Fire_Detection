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


def denormalize_tensor(tensor, mean, std):
    """Denormalize tensor using mean and std"""
    if tensor.dim() == 4:  # batch dimension
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    else:
        return tensor * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)

def generate_augmented_images(image_paths, transform, output_dir, modality, num_copies):
    """
    Generate and save augmented images to disk
    Returns list of paths to augmented images
    """
    os.makedirs(output_dir, exist_ok=True)
    new_paths = []

    for path in image_paths:
        img = Image.open(path)
        base_name = os.path.splitext(os.path.basename(path))[0]

        for j in range(num_copies):
            # Apply transformation
            tensor = transform(img)

            # Denormalize for saving
            if modality == 'RGB':
                denorm_tensor = denormalize_tensor(
                    tensor,
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            elif modality == 'thermal':
                denorm_tensor = denormalize_tensor(
                    tensor,
                    [0.5],
                    [0.5]
                )
                denorm_tensor = denorm_tensor.squeeze(0)  # Remove channel dim
                pil_img = transforms.ToPILImage()(denorm_tensor.cpu())

            # Convert to PIL and save
            if modality == 'RGB':
                pil_img = transforms.ToPILImage()(denorm_tensor)
            else:
                # Handle grayscale separately
                pil_img = transforms.ToPILImage()(denorm_tensor.squeeze(0))

            new_filename = f"{base_name}_aug{j}.jpg"
            new_path = os.path.join(output_dir, new_filename)
            pil_img.save(new_path)
            new_paths.append(new_path)

    return new_paths