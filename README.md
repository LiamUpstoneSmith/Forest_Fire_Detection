# Forest Fire Detection using Multimodal Fusion

This project implements a multimodal deep learning system for forest fire detection using both RGB and thermal images. The system combines Vision Transformers (ViT) for RGB feature extraction and CNNs for thermal feature extraction, fusing these features for highly accurate fire detection.
This project was made in collaberation for a research dissertation.

## Team Members
-  Ufuoma Oyibo
- Panithi Seehawong
- Liam Upstone-Smith

## Key Features

- **Multimodal Fusion Architecture**: Combines RGB and thermal imagery for superior detection accuracy
- **Vision Transformer (ViT)**: State-of-the-art feature extraction for RGB images
- **Custom CNN Architecture**: Specialized feature extraction for thermal images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forest-fire-detection.git
cd forest-fire-detection
```
2. Create and activate virtual environment:
```
python -m venv .venv
source .venv/bin/activate # Linux
.\.venv\Scripts\activate  # Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Download [Flame 3 dataset](https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh)

## Dataset Preparation
Organise your dataset in the following structure:
```
dataset/
├── RGB/
│   ├── fire/
│   │   ├── image1.jpg
│   │   └── ...
│   └── no_fire/
│       ├── image1.jpg
│       └── ...
└── thermal/
    ├── fire/
    │   ├── image1.jpg
    │   └── ...
    └── no_fire/
        ├── image1.jpg
        └── ...
```