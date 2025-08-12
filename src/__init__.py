"""
FireFusion — A multimodal fire detection package combining RGB Vision Transformers
and Thermal CNNs with a late fusion classifier.
"""

from . import data
from . import models
from . import training
# from .predict import FirePredictor
from . import utils

__all__ = ["data", "models", "training", "utils"]
