from ultralytics import YOLO

if __name__ == '__main__':
    # Load your trained model
    model = YOLO("runs/detect/train2/weights/best.pt")
    
    # Predict on one image
    results = model("datasets/images/pred/fcbfc93768bed01a.jpg", conf=0.1)
    
    # Show raw results
    print(results[0].boxes)
    
    # Show prediction on image
    results[0].show()
    
    # Save prediction image
    results[0].save("predicted_result.jpg")