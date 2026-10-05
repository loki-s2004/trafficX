from ultralytics import YOLO

if __name__ == "__main__":

    # Load YOLOv11 Small model
    model = YOLO("yolo11s.pt")

    # Final training using ABO-optimized hyperparameters
    model.train(
        data="dataset/data.yaml",
        epochs=150,                 # full training
        imgsz=768,
        batch=8,                    # from ABO
        lr0=0.00557,                # from ABO
        conf=0.30,                  # from ABO
        iou=0.41,                   # from ABO
        device=0,                   # GPU (use "cpu" if no GPU)
        patience=40,
        workers=0,                  # Windows-safe
        cos_lr=True,
        verbose=True
    )

    # Validate the trained model
    model.val()
