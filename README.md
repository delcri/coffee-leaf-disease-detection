# 🌿 Coffee Disease Detection System


Integrated detection and classification system for coffee plant diseases using Deep Learning. Combines YOLOv8 for region detection and PavicNet-MCv2 for precise disease classification.

> Note: This repository contains only the code and documentation. Datasets and trained models must be downloaded separately (see instructions below).

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Model Download](#-model-download)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)

## ✨ Features

- 🔍 **Automatic detection** of coffee leaves using YOLOv8
- 🎯 **Precise classification** of disease types with PavicNet-MCv2
- 🚀 **3 approaches**: YOLO only, PavicNet only, or integrated pipeline
- 📊 **Detailed metrics** with confusion matrices
- 🖼️ **Visualizations** with bounding boxes and labels
- ⚡ **Batch processing** of multiple images

## 🦠 Detected Diseases

| Disease | Description |
|---------|-------------|
| 🍂 **Coffee Rust** (Roya) | Fungus causing yellow-orange spots |
| 🕷️ **Red Spider Mite** | Mite that damages leaves |
| 🔴 **Cercospora** | Circular spots with gray center |
| 🐛 **Leaf Miner** (Bicho Mineiro) | Larva that perforates leaves |
| ✅ **Healthy** | No disease signs |

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone ...
cd coffee-disease-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (see Model Download section)

# 5. Setup your dataset (see Dataset Setup section)

# 6. Run a quick test
python classification/prueba.py
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.x (optional, for GPU acceleration)
- 8GB RAM minimum
- 10GB disk space

### Step-by-step Installation

1. Clone the repository
```bash
git clone ...
cd coffee-disease-detection
```

2. Create and activate virtual environment
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

IMPORTANT: The dataset is NOT included in this repository, download here: https://drive.google.com/file/d/14vRtxCvQFc-qt2suqlvAwTx1f1J0xwvD/view?usp=sharing

### Required Directory Structure

After downloading/preparing your dataset, you should have:

```
coffee-disease-detection/
├── DATA/                          # YOLO format dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
└── classification_dataset/        # Classification format
    ├── train/
    │   ├── rust/
    │   ├── cercospora/
    │   ├── leaf_miner/
    │   └── healthy/
    ├── valid/
    └── test/
```

### How to Get Dataset

**Option 1: Download from Google Drive**

Dataset available at: [Download Link] (Contact author)

```bash
# Download and extract
# Place in project root directory
```

**Option 2: Use Your Own Images**

Organize your images following the structure above. See [DATASET_INFO.md](DATASET_INFO.md) for details.

**Option 3: Download from Roboflow**

```bash
python dataset.py
```

(Requires Roboflow API key)

### Convert YOLO to Classification Format

```bash
python to_classification.py
```

## 🤖 Model Download

Pre-trained models are available in **[GitHub Releases](https://github.com/your-username/coffee-disease-detection/releases/latest)**

### Download Instructions

1. Go to [Releases](https://github.com/your-username/coffee-disease-detection/releases/latest)
2. Download:
   - `yolov8n.pt` (~6.3 MB) - Object detector
   - `pavicnet_mcv2.h5` (~6.3 MB) - Disease classifier
3. Place them in the `models/` directory:

```
coffee-disease-detection/
├── models/
│   ├── yolov8n.pt          ← Place here
│   └── pavicnet_mcv2.h5    ← Place here
```

## 💻 Usage

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

Quick test:
```bash
python classification/prueba.py
```

Train classifier:
```bash
# Simple training
python classification/training.py

# Or full training with validation
python classification/train_classification.py
```

Evaluate model:
```bash
python classification/evaluate.py
python classification/confusion_matrix_test.py
```

Single image inference:
```bash
python classification/inference.py
```

### Approach 3: Integrated Pipeline (⭐ Recommended)

Process images with the full pipeline (YOLO + PavicNet):

```bash
python integration/integrate_yolo_pavicnet.py \
    --images_dir "DATA/test/images" \
    --classifier_model "models/pavicnet_mcv2.h5" \
    --output_dir "integration_results"
```

Batch processing:
```bash
python integration/pipeline_batch.py
```

Results saved in `results_batch/` with:
- Bounding boxes around leaves
- Disease classification labels
- Confidence scores

## 📁 Project Structure

```
coffee-disease-detection/
│
├── 📂 classification/              # Classification scripts
│   ├── confusion_matrix_test.py   # Generate confusion matrix
│   ├── evaluate.py                # Model evaluation
│   ├── inference.py               # Single image inference
│   ├── prueba.py                  # Quick test script
│   ├── train_classification.py    # Full training pipeline
│   └── training.py                # Simple training
│
├── 📂 integration/                 # Integrated pipeline
│   ├── integrate_yolo_pavicnet.py # Full pipeline (YOLO + PavicNet)
│   └── pipeline_batch.py          # Batch processing
│
├── 📂 models/                      # Trained models (download separately)
│   ├── yolov8n.pt                 # YOLO detector (6.3 MB)
│   └── pavicnet_mcv2.h5           # PavicNet classifier (6.3 MB)
│
├── 📂 integration_results/         # Pipeline outputs (not in repo)
│   ├── crops/                     # Cropped leaf regions
│   ├── overlays/                  # Annotated images
│   ├── confusion_matrix.png
│   └── predictions_summary.csv
│
├── 📂 DATA/                        # YOLO dataset (not in repo)
├── 📂 classification_dataset/      # Classification dataset (not in repo)
├── 📂 results_batch/               # Batch results (not in repo)
├── 📂 runs/                        # Training logs (not in repo)
├── 📂 roya_test/                   # Rust test data (not in repo)
├── 📂 roya_train/                  # Rust train data (not in repo)
├── 📂 roya_valid/                  # Rust validation data (not in repo)
│
├── clasi.py                        # Crop YOLO detections
├── dataset.py                      # Download from Roboflow
├── to_classification.py            # Convert YOLO to classification format
├── trainingyolo.py                 # Train YOLO detector
│
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── LICENSE                         # MIT License
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

Commands:

Simple training:
```bash
python classification/training.py
```

Full training with validation:
```bash
python classification/train_classification.py
```

Configuration:
- Epochs: 300
- Image size: 224x224
- Batch size: 16
- Learning rate: 0.001
- Early stopping: patience=20

Model saved as: `models/pavicnet_mcv2.h5`

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
| Integrated | ~94% | ⚡⚡ Medium | **Production use** |

### Integrated Pipeline Outputs

Results are saved in `integration_results/`:
- **crops/** - Individual leaf detections
- **overlays/** - Original images with annotations
- **confusion_matrix.png** - Classification performance
- **predictions_summary.csv** - Detailed results

Batch results in `results_batch/` with:
- Bounding boxes around detected leaves
- Disease classification labels
- Confidence scores

## 🛠️ Customization

### Change Disease Classes

Edit class names in these files:
- `classification/prueba.py` - Line ~8
- `classification/inference.py` - Line ~15
- `integration/pipeline_batch.py` - Line ~16
- `integration/integrate_yolo_pavicnet.py` - Lines 15-20

### Adjust Model Paths

Update paths in scripts:
```python
# YOLO model
model = YOLO("models/yolov8n.pt")

# PavicNet model
classifier = load_model("models/pavicnet_mcv2.h5")
```

### Configure Confidence Thresholds

In integration scripts:
```python
CONF = 0.25  # Detection confidence
IOU = 0.45   # IoU threshold
```

## 🐛 Troubleshooting

### "Model file not found"
**Solution:** Download models from Releases and place in `models/` directory

### "Dataset directory not found"
**Solution:** Setup dataset as described in [Dataset Setup](#-dataset-setup)

### "CUDA out of memory"
**Solution:** Reduce batch size in training scripts:
```python
batch_size = 8  # Reduce from 16
```

### Import errors
**Solution:** Install all dependencies:
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

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📧 Contact

Anderson Del Castillo
mail - delcast2210@gmail.com
linkedin - www.linkedin.com/in/anderson-sneider-del-castillo-criollo-12b987297

Diego Hernandez 
mail - hernandezdiegoalejandro35@gmail.com
linkedin - www.linkedin.com/in/diego-hernandez-1827ab256

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [TensorFlow](https://www.tensorflow.org/)
- Coffee farming community

---

Made with ❤️ for precision agriculture
