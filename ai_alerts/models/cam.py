
import os

import cv2
from ultralytics import YOLO


bestmodel = "runs/detect/yolov11_custom2/weights/best.pt"
trained_model = YOLO(bestmodel)

cap = cv2.VideoCapture("C:\\Users\\r125v\\Downloads\\input.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = trained_model(frame)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
