# 📊 Dataset Information

## Overview

This project requires two types of datasets:
1. **YOLO Format Dataset** - For training object detection
2. **Classification Dataset** - For training disease classifier

## Dataset Not Included

⚠️ **Important:** The dataset download here:
https://drive.google.com/file/d/14vRtxCvQFc-qt2suqlvAwTx1f1J0xwvD/view?usp=sharing 

## Required Dataset Structure

### 1. YOLO Format Dataset (for Object Detection)

Place in `DATA/` folder:

```
DATA/
├── train/
│   ├── images/              # .jpg training images
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── labels/              # .txt YOLO annotations
│       ├── image001.txt
│       ├── image002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml                # Dataset configuration
```

**data.yaml example:**
```yaml
path: ../DATA
train: train/images
val: valid/images
test: test/images

nc: 4
names: ['rust', 'cercospora', 'phoma', 'leaf_miner']
```

**YOLO annotation format (.txt files):**
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to 0-1.

### 2. Classification Dataset

Place in `classification_dataset/` folder:

```
classification_dataset/
├── train/
│   ├── rust/                # Images of rust disease
│   │   ├── rust001.jpg
│   │   ├── rust002.jpg
│   │   └── ...
│   ├── cercospora/          # Images of cercospora
│   ├── leaf_miner/          # Images of leaf miner
│   └── healthy/             # Images of healthy leaves
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

## Dataset Specifications

### Recommended Sizes

| Split | Min Images | Recommended |
|-------|------------|-------------|
| Train | 800 | 1,200+ |
| Valid | 200 | 300+ |
| Test | 100 | 200+ |

### Image Requirements

- **Format:** JPG or PNG
- **Size:** Minimum 640x640 pixels
- **Quality:** Clear, well-lit images
- **Content:** One or more coffee leaves per image
- **Diversity:** Various lighting conditions, angles, disease stages

### Class Distribution

Try to maintain balanced classes:
- Rust: 25-30%
- Cercospora: 20-25%
- Leaf Miner: 20-25%
- Healthy: 25-30%

## How to Get Dataset

### Option 1: Use Your Own Images

1. Collect coffee leaf images
2. Annotate using [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/heartexlabs/labelImg)
3. Export in YOLO format
4. Organize as described above

### Option 2: Download from Roboflow

```bash
# Edit dataset.py with your API key
python dataset.py
```

Get API key from: https://roboflow.com/

### Option 3: Public Datasets

Search for coffee disease datasets on:
- [Roboflow Universe](https://universe.roboflow.com/)
- [Kaggle](https://www.kaggle.com/datasets)
- [PlantVillage](https://plantvillage.psu.edu/)

### Option 4: Request from Author

Contact project author for dataset access:
- **Email:** your.email@example.com
- **Note:** Dataset may be subject to usage restrictions

## Converting Between Formats

### YOLO → Classification

Use the provided script:

```bash
python to_classification.py
```

This will:
1. Read YOLO annotations
2. Crop bounding boxes from images
3. Save crops in class folders
4. Create train/valid/test splits

## Dataset Preparation Tips

### Quality Control

✅ **Good images:**
- Clear focus
- Good lighting
- Visible disease symptoms
- Minimal background clutter

❌ **Avoid:**
- Blurry images
- Poor lighting
- Heavily occluded leaves
- Multiple diseases on same leaf

### Data Augmentation

Consider these augmentations during training:
- Horizontal/vertical flips
- Rotation (±15°)
- Brightness/contrast adjustment
- Zoom (0.8x - 1.2x)

⚠️ Avoid augmentations that change disease appearance!

### Annotation Guidelines

For YOLO annotations:
1. Draw tight bounding boxes around leaves
2. Include entire leaf (including petiole if visible)
3. Annotate all visible leaves
4. Use correct class labels
5. Double-check annotations

## Dataset Statistics (Example)

For reference, a good dataset might have:

```
Total Images: 1,500
├── Training: 1,050 (70%)
├── Validation: 300 (20%)
└── Testing: 150 (10%)

Classes Distribution:
├── Rust: 400 images (26.7%)
├── Cercospora: 350 images (23.3%)
├── Leaf Miner: 400 images (26.7%)
└── Healthy: 350 images (23.3%)

Average Annotations per Image: 2.5 leaves
Total Annotations: ~3,750
```

## Legal and Ethical Considerations

### Image Rights

- Ensure you have rights to use all images
- Credit original photographers if required
- Respect any usage restrictions
- Don't use copyrighted images without permission

### Privacy

- Don't include identifiable information
- Remove GPS metadata from images
- Get permission if photographing private property

### Sharing Dataset

If sharing your dataset:
- Include LICENSE file
- Document data sources
- Specify usage restrictions
- Consider using [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Troubleshooting

### "Dataset not found" Error

Check:
1. Folder exists in correct location
2. Folder name matches exactly (case-sensitive)
3. data.yaml has correct paths
4. Images have correct extensions (.jpg, .png)

### Poor Training Results

Try:
1. Balance class distribution
2. Add more diverse images
3. Check annotation quality
4. Increase dataset size
5. Apply data augmentation

### Loading Errors

Common issues:
- Corrupted images → Remove or replace
- Wrong file format → Convert to JPG
- Missing annotations → Create or remove images
- Path issues → Use absolute paths or fix relative paths

## Need Help?

- Check [Roboflow Documentation](https://docs.roboflow.com/)
- See [YOLO Format Guide](https://docs.ultralytics.com/datasets/detect/)
- Read [TensorFlow Datasets Guide](https://www.tensorflow.org/tutorials/load_data/images)
- Open an issue on GitHub

---

Good luck building your dataset! 🌿📊
