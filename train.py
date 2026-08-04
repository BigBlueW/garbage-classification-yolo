import os
from ultralytics import YOLO

def main():
    model_path = "yolo11x.pt"
        
    model = YOLO(model_path) 

    # Train the model
    results = model.train(
        model=model_path,
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=32,          # Optimized for RTX 5090 (32GB VRAM)
        workers=16,        # Optimized for Ryzen 9950X (32 Threads)
        cache=True,        # Leverages 128GB RAM to load all images into memory for blazing fast training
        project="garbage_classification_runs",
        name="yolo11x_garbage_model",
        device="0"
    )
    print("Training finished. Weights saved to garbage_classification_runs/yolo11x_garbage_model/weights/")

if __name__ == '__main__':
    main()
