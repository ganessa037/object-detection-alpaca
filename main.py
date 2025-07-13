from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Load a model - use Ultralytics supported versions:
    
    # YOLOv7 is NOT supported by Ultralytics library
    # Use these instead:
    
    # model = YOLO("yolov8n.pt")      # YOLOv8 Nano - smallest, fastest
    # model = YOLO("yolov9t.pt")    # YOLOv9 Tiny
    model = YOLO("yolov10n.pt")   # YOLOv10 Nano  
    # model = YOLO("yolo11n.pt")    # YOLOv11 Nano - latest


    # Train the model
    results = model.train(data='config.yaml', epochs=40)