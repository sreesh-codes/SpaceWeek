"""Train the YOLO SAR Ship Detector on the OSSDD Sentinel-1 Dataset.

Fine-tunes YOLO11n on radar amplitude chips (VV/VH bands converted to RGB
via robust-z scoring) to detect ships in maritime anchorages.
Outputs bounding boxes and calculates vessel pixel dimensions for length matching.
"""
import os
from ultralytics import YOLO

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(repo_root, "data", "ossdd_yolo", "data.yaml")
    output_dir = os.path.join(repo_root, "models")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading pretrained YOLO11n model...")
    model = YOLO("yolo11n.pt")

    print(f"Training SAR ship detector on {data_yaml}...")
    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=16,
        workers=4,
        project=output_dir,
        name="sar_ship_yolo",
        exist_ok=True,
        verbose=True,
        save=True,
        val=True,
    )
    print("\nTraining completed successfully!")
    print(f"Best model weights saved at: {os.path.join(output_dir, 'sar_ship_yolo', 'weights', 'best.pt')}")

    # Validate best model
    val_metrics = model.val()
    print(f"Validation mAP50: {val_metrics.box.map50:.4f}, mAP50-95: {val_metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
