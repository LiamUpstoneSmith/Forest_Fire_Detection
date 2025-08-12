# from typing import Tuple
# import torch
# from PIL import Image
# from src.utils import get_vit_transform, get_thermal_transform
# from src.models import ViT, CNN, FusionModel
# import os
#
# class FirePredictor:
#     def __init__(self, vit_ckpt: str, cnn_ckpt: str, fusion_ckpt: str, device: str = "cpu", image_size: int = 224,
#                  vit_feat_dim: int = 768, cnn_feat_dim: int = 128):
#         self.device = torch.device(device)
#         # load models
#         self.vit = ViT({"pretrained": False, "freeze_backbone": True})
#         self.cnn = CNN({"feature_dim": cnn_feat_dim})
#         # attempt to load state dicts
#         if vit_ckpt and os.path.exists(vit_ckpt):
#             self._load_checkpoint_into(self.vit, vit_ckpt)
#         if cnn_ckpt and os.path.exists(cnn_ckpt):
#             self._load_checkpoint_into(self.cnn, cnn_ckpt)
#         self.vit.eval().to(self.device)
#         self.cnn.eval().to(self.device)
#         # fusion model
#         self.fusion = FusionModel(self.vit, self.cnn, {"vit_feat_dim": vit_feat_dim, "cnn_feat_dim": cnn_feat_dim, "freeze_extractors": True})
#         if fusion_ckpt and os.path.exists(fusion_ckpt):
#             self._load_checkpoint_into(self.fusion, fusion_ckpt)
#         self.fusion.eval().to(self.device)
#
#         self.vit_transform = get_vit_transform(image_size)
#         self.thermal_transform = get_thermal_transform(image_size)
#
#     def _load_checkpoint_into(self, model, ckpt_path):
#         state = torch.load(ckpt_path, map_location="cpu")
#         # support both raw state_dict and Lightning checkpoint
#         if isinstance(state, dict) and 'state_dict' in state:
#             sd = state['state_dict']
#         else:
#             sd = state
#         # adapt keys if necessary
#         try:
#             model.load_state_dict(sd)
#         except RuntimeError:
#             # try stripping leading "model." or "module."
#             new_sd = {}
#             for k, v in sd.items():
#                 new_k = k
#                 if new_k.startswith("model."):
#                     new_k = new_k.replace("model.", "", 1)
#                 if new_k.startswith("module."):
#                     new_k = new_k.replace("module.", "", 1)
#                 new_sd[new_k] = v
#             model.load_state_dict(new_sd)
#
#     def predict(self, rgb_path: str, thermal_path: str) -> Tuple[str, float]:
#         x1 = self._load_and_preprocess(rgb_path, self.vit_transform).to(self.device)
#         x2 = self._load_and_preprocess(thermal_path, self.thermal_transform).to(self.device)
#
#         with torch.no_grad():
#             logits = self.fusion(x1, x2)
#             prob = torch.sigmoid(logits).cpu().item()
#             label = "Fire" if prob > 0.5 else "No fire"
#             return label, prob
#
#     def _load_and_preprocess(self, path, transform):
#         with Image.open(path) as img:
#             img = img.convert("RGB")
#             return transform(img).unsqueeze(0)
