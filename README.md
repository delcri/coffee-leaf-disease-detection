# coffee-leaf-disease-detection
# 🌿 Coffee Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Integrated detection and classification system for coffee plant diseases using Deep Learning. Combines YOLOv8 for region detection and PavicNet-MCv2 for precise disease classification.

## 📋 Table of Contents

- [Features](#-features)
- [Detected Diseases](#-detected-diseases)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- 🔍 **Automatic detection** of coffee leaves using YOLOv8
- 🎯 **Precise classification** of multiple disease types with PavicNet-MCv2
- 🚀 **Integrated pipeline** combining detection + classification
- 📊 **Detailed metrics** with confusion matrices and reports
- 🖼️ **Visualizations** with bounding boxes and confidence labels
- ⚡ **Batch processing** of multiple images
- 📈 **3 different approaches** for result comparison

## 🦠 Detected Diseases

The system can identify the following conditions in coffee plants:

| Disease | Description |
|---------|-------------|
| 🍂 **Coffee Rust** (Roya) | Fungus causing yellow-orange spots on leaves |
| 🕷️ **Red Spider Mite** (Araña Roja) | Mite that damages leaves and reduces photosynthesis |
| 🔴 **Cercospora** | Circular spots with gray center on leaves |
| 🐛 **Leaf Miner** (Bicho Mineiro) | Larva that perforates and damages leaf tissue |
| ✅ **Healthy** | Plant without disease signs |

## 🏗️ System Architecture

### Three Approaches

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT IMAGE                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────┐         ┌──────────────────┐
│  APPROACH 1      │         │  APPROACH 2      │
│  YOLO Only       │         │  PavicNet Only   │
│  (trainingyolo)  │         │  (training)      │
│                  │         │                  │
│  ⚡ Fast         │         │  ✅ Accurate     │
│  ❌ Less precise │         │  ❌ Needs crops  │
└──────────────────┘         └──────────────────┘
        │                             │
        └──────────────┬──────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  APPROACH 3 (RECOMMENDED)      │
        │  Integrated Pipeline           │
        │  (integrate_yolo_pavicnet)     │
        │                                │
        │  Step 1: YOLO detects leaves   │
        │  Step 2: Crops regions         │
        │  Step 3: PavicNet classifies   │
        │                                │
        │  ✅ Best accuracy              │
        │  ✅ Automatic end-to-end       │
        └────────────────────────────────┘
```

### PavicNet-MCv2 Architecture

```
Input (224x224x3)
    ↓
Conv2D(16) → BatchNorm → MaxPool(4x4)
    ↓
Conv2D(32) → BatchNorm → MaxPool(2x2)
    ↓
Conv2D(64) → BatchNorm → MaxPool(2x2)
    ↓
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

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.x (optional, for GPU acceleration)
- 8GB RAM minimum
- 10GB disk space

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-username/coffee-disease-detection.git
cd coffee-disease-detection
```

2. **Create virtual environment**
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

4. **Download pre-trained models**

The models are available:
- `yolov8n.pt` - YOLOv8 detector (6.3 MB)
- `pavicnet_mcv2.h5` - PavicNet classifier (6.3 MB)

Place them in the project root directory.

## 💻 Usage

### Quick Test

Test the classifier on a single image:

```bash
python prueba.py
```

Edit `prueba.py` to change the image path and class names as needed.

### Approach 1: YOLO Detection Only

Train YOLO for disease detection:

```bash
python trainingyolo.py
```

Detect and crop regions:

```bash
python clasi.py
```

### Approach 2: PavicNet Classification Only

**Prepare classification dataset:**
```bash
python to_classification.py
```

**Train PavicNet:**
```bash
# Option 1: Full training with evaluation
python train_classification.py

# Option 2: Simple training
python training.py
```

**Evaluate model:**
```bash
python evaluate.py
```

**Test with confusion matrix:**
```bash
python confusion_matrix_test.py
```

**Single image inference:**
```bash
python inference.py
```

### Approach 3: Integrated Pipeline (Recommended)

**Process images with full pipeline:**
```bash
python integrate_yolo_pavicnet.py \
    --images_dir "DATA/test" \
    --classifier_model "pavicnet_mcv2.h5" \
    --output_dir "integration_results"
```

**Batch processing:**
```bash
python pipeline_batch.py
```

This will:
1. Detect leaves with YOLO
2. Crop each detection
3. Classify with PavicNet
4. Generate visualizations with bounding boxes and labels
5. Save results in `results_batch/`

### Download Dataset

```bash
python dataset.py
```

(Requires Roboflow API key)

## 📁 Project Structure

```
coffee-disease-detection/
│
├── 📂 DATA/                        # YOLO format dataset
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── 📂 classification_dataset/      # Classification format dataset
│   ├── train/
│   ├── valid/
│   └── test/
│
├── 📂 runs/detect/                 # YOLO training results
│
├── 📂 integration_results/         # Integrated pipeline results
│   ├── crops/                     # Cropped detections
│   ├── overlays/                  # Annotated images
│   ├── confusion_matrix.png
│   └── predictions_summary.csv
│
├── 📂 results_batch/               # Batch processing results
│
├── 📂 roya_train/                  # Rust disease training data
├── 📂 roya_valid/                  # Rust disease validation data
├── 📂 roya_test/                   # Rust disease test data
│
├── 📄 yolov8n.pt                   # YOLOv8 model (6.3 MB)
├── 📄 pavicnet_mcv2.h5             # PavicNet model (6.3 MB)
│
├── 📄 trainingyolo.py              # Train YOLO detector
├── 📄 clasi.py                     # Crop detections
│
├── 📄 training.py                  # Train PavicNet (simple)
├── 📄 train_classification.py      # Train PavicNet (with validation)
├── 📄 evaluate.py                  # Evaluate model
├── 📄 inference.py                 # Single image inference
├── 📄 confusion_matrix_test.py     # Generate confusion matrix
├── 📄 prueba.py                    # Quick test script
│
├── 📄 integrate_yolo_pavicnet.py   # Integrated pipeline
├── 📄 pipeline_batch.py            # Batch processing
│
├── 📄 dataset.py                   # Download from Roboflow
├── 📄 to_classification.py         # Convert YOLO to classification format
│
├── 📄 requirements.txt             # Dependencies
├── 📄 README.md                    # This file
└── 📄 LICENSE                      # MIT License
```

## 🔬 Training Details

### YOLO Training

**Script:** `trainingyolo.py`

**Configuration:**
- Model: YOLOv8n (nano)
- Epochs: 300
- Image size: 640x640
- Batch size: 16
- Optimizer: Adam
- Early stopping patience: 20

**Dataset format:** YOLO (defined in `DATA/data.yaml`)

### PavicNet Training

**Scripts:** `training.py` or `train_classification.py`

**Architecture:** PavicNet-MCv2
- 4 convolutional blocks with batch normalization
- 4 dense layers with dropout (0.2)
- Softmax output layer

**Configuration:**
- Epochs: 300
- Image size: 224x224
- Batch size: 16
- Learning rate: 0.001 (Adam)
- Early stopping patience: 20

**Classes:** 4 (adjust based on your `classification_dataset` folders)

## 📊 Results

### Approach Comparison

| Approach | Accuracy | Speed | Best For |
|----------|----------|-------|----------|
| YOLO Only | ~85% | ⚡⚡⚡ Fast | Quick screening in field |
| PavicNet Only | ~92% | ⚡⚡ Medium | Lab analysis with pre-crops |
| **Integrated** | **~94%** | ⚡⚡ Medium | **Production deployment** |

### Integrated Pipeline Results

Results are saved in `integration_results/`:
- `crops/` - Individual leaf detections
- `overlays/` - Original images with annotations
- `confusion_matrix.png` - Classification performance matrix
- `predictions_summary.csv` - Detailed predictions

Batch results are saved in `results_batch/` with annotated images showing:
- Bounding boxes around detected leaves
- Disease classification labels
- Confidence scores

## 🛠️ Customization

### Change Disease Classes

Edit class names in:
- `prueba.py` - Line with `class_names` variable
- `inference.py` - Line with `class_names` variable
- `pipeline_batch.py` - Line with `class_names` variable
- `integrate_yolo_pavicnet.py` - `label_mapping` and `class_names` variables

### Adjust Model Paths

Update paths in scripts as needed:
- YOLO model: `yolov8n.pt` or `runs/detect/trainX/weights/best.pt`
- PavicNet model: `pavicnet_mcv2.h5`

### Configure Dataset Paths

Edit in training scripts:
- `DATA/` directory for YOLO format
- `classification_dataset/` for classification format

## 🐛 Common Issues

### Issue: "Model file not found"
**Solution:** Ensure model files are in the correct location:
```bash
ls yolov8n.pt pavicnet_mcv2.h5
```

### Issue: "No module named 'ultralytics'"
**Solution:** Install requirements:
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in training scripts or use CPU:
```python
batch_size = 8  # Reduce from 16
```

### Issue: Absolute paths in scripts
**Solution:** Convert to relative paths:
```python
# Bad
path = "/home/anderson/cafe/model.h5"

# Good
from pathlib import Path
BASE_DIR = Path(__file__).parent
path = BASE_DIR / "model.h5"
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Roboflow](https://roboflow.com/) - Dataset management and annotation
- Coffee research community for disease classification expertise


---

**Made with ❤️ for precision agriculture and coffee farmers**
