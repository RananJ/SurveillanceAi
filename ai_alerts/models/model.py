from ultralytics import YOLO
import os
import cv2
import torch


def main():
    print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"DEVICE: {torch.cuda.get_device_name(0)}")
        print("CUDA VRAM:", torch.cuda.get_device_properties(
            0).total_memory / 1e9, "GB")
    else:
        print("CUDA is not available. Using CPU.")

    if not os.path.exists("data.yaml"):
        print("ERROR: data.yaml file not found!")
        return
    model = YOLO("runs/detect/yolov11_custom/weights/best.pt")

    results = model.train(
        data="data.yaml",
        epochs=50,
        patience=10,
        imgsz=640,
        workers=5,
        device=0,
        pretrained=True,
        batch=12,
        amp=True,
        name="yolov11_custom",
        verbose=True,
        cache=True,
    )

    print("--- Starting Validation ---")
    best_model = "runs/detect/yolov11_custom/weights/best.pt"
    trained_model = YOLO(best_model)
    eval_results = trained_model.val()
    print("Evaluation Results:", eval_results)

    print("--- Starting Prediction on Test Set ---")
    test_results = trained_model.predict(source="dataset/images/test",
                                         conf=0.5, iou=0.7, save=True)
    print("Prediction complete. Results saved in the 'runs/detect' directory.")


if __name__ == "__main__":
    main()
