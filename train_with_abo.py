from ultralytics import YOLO
from abo import AfricanBuffaloOptimization

def evaluate_yolo(params):
    model = YOLO("yolo11s.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=20,          # short runs for ABO
        imgsz=640,
        batch=params["batch"],
        lr0=params["lr"],
        conf=params["conf"],
        iou=params["iou"],
        device=0,
        workers=0,
        verbose=False
    )

    metrics = {
        "map": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr
    }

    return metrics


if __name__ == "__main__":
    abo = AfricanBuffaloOptimization(
        num_buffalos=5,
        max_iter=3
    )

    best_params = abo.optimize(evaluate_yolo)

    print("\n✅ BEST HYPERPARAMETERS FOUND BY ABO")
    for k, v in best_params.items():
        print(f"{k}: {v}")
