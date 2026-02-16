# 🌿 Coffee Disease Detection System


Integrated detection and classification system for coffee plant diseases using Deep Learning. Combines YOLOv8 for region detection and PavicNet-MCv2 for precise disease classification.

> **Note:** This repository contains only the code and documentation. Datasets and trained models must be downloaded separately (see instructions below).

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Model Download](#-model-download)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Results](#-results)

## ✨ Features

- 🔍 Automatic detection of coffee leaves using YOLOv8
- 🎯 Precise classification of disease types with PavicNet-MCv2
- 🚀 3 approaches: YOLO only, PavicNet only, or integrated pipeline
- 📊 Detailed metrics with confusion matrices
- 🖼️ Visualizations with bounding boxes and labels
- ⚡ Batch processing of multiple images

## 🦠 Detected Diseases

| Disease | Description |
|---------|-------------|
| 🍂 Coffee Rust (Roya) | Fungus causing yellow-orange spots |
| 🕷️ Red Spider Mite | Mite that damages leaves |
| 🔴 Cercospora | Circular spots with gray center |
| 🐛 Leaf Miner (Bicho Mineiro) | Larva that perforates leaves |
| ✅ Healthy | No disease signs |

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/coffee-disease-detection.git
cd coffee-disease-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (see Model Download section)

# 5. Setup your dataset (see Dataset Setup section)

# 6. Run a quick test
python prueba.py
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.x (optional, for GPU acceleration)
- 8GB RAM minimum
- 10GB disk space

### Step-by-step Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/coffee-disease-detection.git
cd coffee-disease-detection
```

2. **Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset Setup

## 📊 Dataset Download

Download the dataset from [Releases](https://github.com/your-username/coffee-disease-detection/releases):

1. Go to Releases
2. Download `dataset.zip` (1.2 GB)
3. Extract in project root:
````bash
   unzip dataset.zip
````

#### For YOLO Training (Object Detection)

```
DATA/
├── train/
│   ├── images/          # Training images (.jpg)
│   └── labels/          # YOLO annotations (.txt)
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # YOLO annotations
├── test/
│   ├── images/          # Test images
│   └── labels/          # YOLO annotations
└── data.yaml            # Dataset configuration
```

**data.yaml example:**
```yaml
train: DATA/train/images
val: DATA/valid/images
test: DATA/test/images

nc: 4  # number of classes
names: ['rust', 'cercospora', 'phoma', 'leaf_miner']
```

#### For Classification Training

```
classification_dataset/
├── train/
│   ├── rust/           # Images of rust disease
│   ├── cercospora/     # Images of cercospora
│   ├── leaf_miner/     # Images of leaf miner
│   └── healthy/        # Images of healthy leaves
├── valid/
│   ├── rust/
│   ├── cercospora/
│   ├── leaf_miner/
│   └── healthy/
└── test/
    ├── rust/
    ├── cercospora/
    ├── leaf_miner/
    └── healthy/
```

### Option 2: Download from Roboflow

```bash
# Edit dataset.py with your Roboflow API key
python dataset.py
```

### Option 3: Request Dataset

The dataset used in this project is a custom collection. For access:
- **Email:** your.email@example.com
- **Note:** Dataset is ~2-5GB and contains 1,500+ annotated images

### Convert YOLO Dataset to Classification Format

If you have YOLO format data and want to train the classifier:

```bash
python to_classification.py
```

This script will:
1. Read YOLO annotations
2. Crop detected regions
3. Organize into class folders

## 🤖 Model Download

Pre-trained models are available in **[GitHub Releases](https://github.com/your-username/coffee-disease-detection/releases/latest)**

### Download Instructions

1. Go to [Releases](https://github.com/your-username/coffee-disease-detection/releases/latest)
2. Download:
   - `yolov8n.pt` (~6.3 MB) - Object detector
   - `pavicnet_mcv2.h5` (~6.3 MB) - Disease classifier
3. Place them in the project root directory:

```
coffee-disease-detection/
├── yolov8n.pt          ← Place here
├── pavicnet_mcv2.h5    ← Place here
├── prueba.py
└── ...
```

### Alternative: Train Your Own Models

See [Training](#-training) section below.

## 💻 Usage

## Quick Test

Test the classifier on a single image:

```bash
python prueba.py
```

To customize:
1. Edit line with `img_path` variable
2. Adjust `class_names` to match your classes

### Approach 1: YOLO Detection Only

Train YOLO detector:
```bash
python trainingyolo.py
```

Detect and crop leaves:
```bash
python clasi.py
```

### Approach 2: PavicNet Classification Only

Train classifier:
```bash
python train_classification.py
```

Evaluate:
```bash
python evaluate.py
python confusion_matrix_test.py
```

Inference:
```bash
python inference.py
```

### Approach 3: Integrated Pipeline (⭐ Recommended)

Process images with the full pipeline (YOLO + PavicNet):

```bash
python integrate_yolo_pavicnet.py \
    --images_dir "path/to/images" \
    --classifier_model "pavicnet_mcv2.h5" \
    --output_dir "integration_results"
```

**Batch processing:**
```bash
python pipeline_batch.py
```

Results will be saved in `results_batch/` with:
- Bounding boxes around leaves
- Disease classification labels
- Confidence scores

## 📁 Project Structure

```
coffee-disease-detection/
│
├── 📄 Python Scripts (13 files)
│   ├── trainingyolo.py              # Train YOLO detector
│   ├── clasi.py                     # Crop detections
│   ├── training.py                  # Train PavicNet (simple)
│   ├── train_classification.py      # Train PavicNet (with validation)
│   ├── evaluate.py                  # Evaluate classifier
│   ├── inference.py                 # Single image inference
│   ├── confusion_matrix_test.py     # Generate confusion matrix
│   ├── prueba.py                    # Quick test script
│   ├── integrate_yolo_pavicnet.py   # Integrated pipeline
│   ├── pipeline_batch.py            # Batch processing
│   ├── dataset.py                   # Download from Roboflow
│   └── to_classification.py         # Convert YOLO to classification
│
├── 📄 Models (download separately)
│   ├── yolov8n.pt                   # YOLO detector (6.3 MB)
│   └── pavicnet_mcv2.h5             # PavicNet classifier (6.3 MB)
│
├── 📂 DATA/                          # YOLO dataset (not in repo)
├── 📂 classification_dataset/        # Classification dataset (not in repo)
├── 📂 results_batch/                 # Output results (not in repo)
├── 📂 integration_results/           # Pipeline outputs (not in repo)
│
├── 📄 .gitignore                     # Git ignore rules
├── 📄 README.md                      # This file
├── 📄 requirements.txt               # Python dependencies
└── 📄 LICENSE                        # MIT License
```

## 🎓 Training

### Train YOLO Detector

Requirements:
- YOLO format dataset in `DATA/` folder
- `data.yaml` configuration file

Command:
```bash
python trainingyolo.py
```

Configuration:
- Epochs: 300
- Image size: 640x640
- Batch size: 16
- Optimizer: Adam
- Early stopping: patience=20

Results saved in: `runs/detect/train/`

### Train PavicNet Classifier

Requirements:
- Classification dataset in `classification_dataset/` folder
- Organized by class folders

Command:
```bash
python train_classification.py
```

Configuration:
- Epochs: 300
- Image size: 224x224
- Batch size: 16
- Learning rate: 0.001
- Early stopping: patience=20

Model saved as: `pavicnet_mcv2.h5`

### PavicNet-MCv2 Architecture

```
Input (224x224x3)
    ↓
Conv2D(16) → BatchNorm → MaxPool(4x4)
Conv2D(32) → BatchNorm → MaxPool(2x2)
Conv2D(64) → BatchNorm → MaxPool(2x2)
Conv2D(128) → BatchNorm → MaxPool(2x2)
    ↓
Flatten
    ↓
Dense(128) → Dropout(0.2)
Dense(64) → Dropout(0.2)
Dense(32) → Dropout(0.2)
Dense(16) → Dropout(0.2)
    ↓
Dense(4) → Softmax
```

## 📊 Results

### Approach Comparison

| Approach | Accuracy | Speed | Best For |
|----------|----------|-------|----------|
| YOLO Only | ~85% | ⚡⚡⚡ Fast | Quick field screening |
| PavicNet Only | ~92% | ⚡⚡ Medium | Lab analysis (pre-cropped) |
| **Integrated** | **~94%** | ⚡⚡ Medium | **Production use** |

### Outputs

**Integrated Pipeline creates:**
- `integration_results/crops/` - Cropped leaf regions
- `integration_results/overlays/` - Annotated images
- `integration_results/confusion_matrix.png` - Performance metrics
- `integration_results/predictions_summary.csv` - Detailed results

**Batch Processing creates:**
- `results_batch/*.jpg` - Images with bounding boxes and labels

## 🛠️ Customization

### Change Disease Classes

Edit class names in these files:
- `prueba.py` - Line ~8: `class_names = [...]`
- `inference.py` - Line ~15: `class_names = [...]`
- `pipeline_batch.py` - Line ~16: `class_names = [...]`
- `integrate_yolo_pavicnet.py` - Lines 15-20: `label_mapping` and `class_names`

### Adjust Confidence Thresholds

In `pipeline_batch.py` and `integrate_yolo_pavicnet.py`:
```python
CONF = 0.25  # Detection confidence
IOU = 0.45   # IoU threshold
```

## 🐛 Troubleshooting

### "Model file not found"
Solution: Download models from Releases and place in project root

### "Dataset directory not found"
Solution: Setup dataset as described in [Dataset Setup](#-dataset-setup)

### "CUDA out of memory"
**Solution:** Reduce batch size:
```python
batch_size = 8  # Reduce from 16
```

### Import errors
Solution: Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the project
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📧 Contact

Anderson Del Castillo - delcast2210@gmail.com
linkedin - www.linkedin.com/in/anderson-sneider-del-castillo-criollo-12b987297


## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [TensorFlow](https://www.tensorflow.org/)
- Coffee farming community for domain expertise

---

Made with ❤️ for precision agriculture
