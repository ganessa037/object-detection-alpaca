# Alpaca Detection with YOLO Models

A comprehensive computer vision project for detecting alpacas using various YOLO model versions. This project compares the performance of YOLOv8n, YOLOv9t, YOLOv10n, and YOLOv11n models on a custom alpaca dataset.

## Project Overview

- **Objective**: Train and compare different YOLO models for alpaca detection
- **Dataset**: Custom annotated alpaca images
- **Models Tested**: YOLOv8n, YOLOv9t, YOLOv10n, YOLOv11n
- **Training Images**: 300 manually annotated images
- **Validation Images**: 52 images
- **Classes**: 1 (Alpaca)

## Model Performance Comparison

### Final Model Results

| Model    | Final mAP@0.5 | Final Precision | Final Recall | Final mAP@0.5:0.95 |
|----------|---------------|-----------------|--------------|-------------------|
| YOLOv9t  | 0.669         | 0.707          | 0.638        | 0.454            |
| YOLOv10n | 0.607         | 0.683          | 0.545        | 0.394            |
| YOLOv11n | 0.678         | 0.743          | 0.615        | 0.477            |
| **YOLOv8n** | **0.733** | **0.792**      | **0.672**    | **0.476**        |

### Best Model: YOLOv8n
- **Best mAP@0.5**: 0.733 (73.3%)
- **Highest Precision**: 0.792 (79.2%)
- **Best Recall**: 0.672 (67.2%)
- **Training Epochs**: 40

## Key Insights

1. **YOLOv8n** consistently outperformed all other models across all metrics
2. **YOLOv11n** showed competitive performance with the highest precision among alternatives
3. **YOLOv10n** had the lowest performance, particularly in recall
4. All models required low confidence threshold (0.1) for optimal detection

## Dataset Information

- **Total Training Images**: 300 images
- **Validation Images**: 52 images
- **Total Dataset**: 352 manually annotated images
- **Annotation Format**: YOLO format (normalized coordinates)
- **Image Sources**: Custom alpaca dataset
- **Labeling**: Manual annotation for precise bounding boxes
- **Train/Val Split**: 300/52 (85%/15% split)

## Results Showcase

### Image Detection Results
![Alpaca Detection Result](predicted_result.jpg)
*Sample alpaca detection with bounding boxes and confidence scores*

### Video Detection Results

#### Video 1: Alpaca Detection Demo
https://github.com/ganessa037/object-detection-alpaca/raw/main/alpaca.mp4_out

*Original alpaca video with real-time detection overlay*

#### Video 2: Multiple Alpaca Detection  
https://github.com/ganessa037/object-detection-alpaca/raw/main/alpaca2.mp4_out

*Secondary video demonstration showing model performance on different scenarios*

```
## Project Structure

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
├── predicted_result.jpg # Sample output
├── alpaca.mp4_out       # Video result 1
└── alpaca2.mp4_out      # Video result 2
```






### Software Dependencies
- Python 3.8+
- Ultralytics YOLO
- OpenCV
- PyTorch
- CUDA (for GPU acceleration)

## Model Selection Criteria

The **YOLOv8n** model was selected as the final model based on:
1. **Highest mAP@0.5** (0.733) - Best overall detection accuracy
2. **Superior precision** (0.792) - Fewer false positives
3. **Best recall** (0.672) - Better at finding all alpacas
4. **Consistent performance** across different test scenarios
5. **Optimal speed/accuracy balance** for real-time applications


