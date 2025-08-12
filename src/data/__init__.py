# from .fusion_dataset import CombinedDataset
from .cnn_dataset import ThermalDataset
from .vit_dataset import RGBDataset

__all__ = ["RGBDataset", "ThermalDataset"]