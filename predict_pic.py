from ultralytics import YOLO
import glob
import os

if __name__ == '__main__':
    # Load your trained model
    model = YOLO("runs/detect/train2/weights/best.pt")
    
    # Get all images from pred folder
    image_files = glob.glob("datasets/images/pred/*.jpg")
    
    if not image_files:
        print("No images found in datasets/images/pred/")
        print("Please add test images to that folder")
        exit()
    
    print(f"Testing {len(image_files)} images...")
    
    # Run batched inference on all images
    results = model(image_files, conf=0.1)  # Low confidence for your model
    
    # Process results
    for i, result in enumerate(results):
        image_name = os.path.basename(image_files[i])
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            print(f"{image_name}: Found {len(boxes)} alpaca(s)")
            for box in boxes:
                conf = box.conf[0].item()
                print(f"  Confidence: {conf:.3f}")
        else:
            print(f"{image_name}: No alpacas detected")
        
        # Save result with bounding boxes
        result.save(filename=f"prediction_results/result_{image_name}")
    
    print(f"\nResults saved to prediction_results/ folder")