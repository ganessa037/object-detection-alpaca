# 🦙 Alpaca Detection with YOLO Models

A comprehensive computer vision project for detecting alpacas using various YOLO model versions. This project compares the performance of YOLOv8n, YOLOv9t, YOLOv10n, and YOLOv11n models on a custom alpaca dataset.

## 📊 Project Overview

- **Objective**: Train and compare different YOLO models for alpaca detection
- **Dataset**: Custom annotated alpaca images
- **Models Tested**: YOLOv8n, YOLOv9t, YOLOv10n, YOLOv11n
- **Training Images**: 300 manually annotated images
- **Validation Images**: 52 images
- **Classes**: 1 (Alpaca)

## 🏆 Model Performance Comparison

### Final Model Results

| Model    | Final mAP@0.5 | Final Precision | Final Recall | Final mAP@0.5:0.95 |
|----------|---------------|-----------------|--------------|-------------------|
| YOLOv9t  | 0.669         | 0.707          | 0.638        | 0.454            |
| YOLOv10n | 0.607         | 0.683          | 0.545        | 0.394            |
| YOLOv11n | 0.678         | 0.743          | 0.615        | 0.477            |
| **YOLOv8n** | **0.733** | **0.792**      | **0.672**    | **0.476**        |

### 🥇 Winner: YOLOv8n
- **Best mAP@0.5**: 0.733 (73.3%)
- **Highest Precision**: 0.792 (79.2%)
- **Best Recall**: 0.672 (67.2%)
- **Training Epochs**: 40

## 📈 Key Insights

1. **YOLOv8n** consistently outperformed all other models across all metrics
2. **YOLOv11n** showed competitive performance with the highest precision among alternatives
3. **YOLOv10n** had the lowest performance, particularly in recall
4. All models required low confidence threshold (0.1) for optimal detection

## 🎯 Dataset Information

- **Total Training Images**: 300 images
- **Validation Images**: 52 images
- **Total Dataset**: 352 manually annotated images
- **Annotation Format**: YOLO format (normalized coordinates)
- **Image Sources**: Custom alpaca dataset
- **Labeling**: Manual annotation for precise bounding boxes
- **Train/Val Split**: 300/52 (85%/15% split)

## 🖼️ Results Showcase

### Image Detection Results
![Alpaca Detection Result](predicted_result.jpg)
*Sample alpaca detection with bounding boxes and confidence scores*

### Video Detection Results

#### Video 1: Alpaca Detection Demo
**File**: `alpaca.mp4_out`

*Original alpaca video with real-time detection overlay*

#### Video 2: Multiple Alpaca Detection  
**File**: `alpaca2.mp4_out`

*Secondary video demonstration showing model performance on different scenarios*

**Video Features:**
- Real-time bounding box detection
- Confidence score display
- Multiple alpaca tracking
- Smooth video processing at original FPS

## 🚀 Model Usage

### Best Model Recommendation
```python
from ultralytics import YOLO

# Load the best performing model
model = YOLO("runs/detect/train40/weights/best.pt")  # YOLOv8n

# Predict with low confidence threshold
results = model("your_image.jpg", conf=0.1)
results[0].show()  # Display results
```

### Performance Notes
- Use **confidence threshold of 0.1** for optimal detection
- Model works best on clear alpaca images
- Handles multiple alpacas in single image
- Real-time video processing capable

## 📁 Project Structure

```
object-detection-alpaca/
├── datasets/
│   ├── images/train/     # 300 training images
│   ├── images/val/       # 52 validation images
│   └── labels/train/     # YOLO format annotations
├── runs/detect/
│   ├── train/           # YOLOv9t results
│   ├── train2/          # YOLOv10n results  
│   ├── train35/         # YOLOv11n results
│   └── train40/         # YOLOv8n results (BEST)
├── videos/              # Test videos
├── config.yaml          # Dataset configuration
├── main.py              # Training script
├── predict_pic.py       # Image prediction
├── predict_vid.py       # Video prediction
├── compare_models.py    # Model comparison analysis
└── predicted_result.jpg # Sample output
```

## 🔧 Training Configuration

- **Input Resolution**: 640x640
- **Batch Size**: Optimized for 8GB GPU memory
- **Optimizer**: AdamW
- **Data Augmentation**: Standard YOLO augmentations
- **Epochs**: 40 (optimal performance achieved)

## 📊 Technical Specifications

### Hardware Requirements
- **GPU**: 8GB VRAM minimum
- **RAM**: 16GB recommended
- **Storage**: 5GB for dataset and models

### Software Dependencies
- Python 3.8+
- Ultralytics YOLO
- OpenCV
- PyTorch
- CUDA (for GPU acceleration)

## 🎯 Model Selection Criteria

The **YOLOv8n** model was selected as the final model based on:
1. **Highest mAP@0.5** (0.733) - Best overall detection accuracy
2. **Superior precision** (0.792) - Fewer false positives
3. **Best recall** (0.672) - Better at finding all alpacas
4. **Consistent performance** across different test scenarios
5. **Optimal speed/accuracy balance** for real-time applications


## 📝 Usage Instructions

1. **Training**: Run `python main.py` with proper config.yaml
2. **Image Prediction**: Use `python predict_pic.py` 
3. **Video Processing**: Execute `python predict_vid.py`
4. **Model Comparison**: Analyze with `python compare_models.py`

