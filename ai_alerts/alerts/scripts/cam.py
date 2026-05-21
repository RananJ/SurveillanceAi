import os
import time
import cv2
import django
import threading
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from django.core.files import File

# --- THREADED FUNCTION: This runs in the background ---


def save_alert_and_process(all_frames, fps, frame_size, video_dir, detected_violations):
    """
    This function runs entirely in the background. It takes the accumulated frames,
    saves the clip directly using OpenCV, and updates the database.
    """
    # 1. Define filenames and save the clip using the 'avc1' (H.264) codec
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violation_{timestamp_str}.mp4"
    filepath = os.path.join(video_dir, filename)

    print(f"[THREAD] Saving {len(all_frames)} frames to {filename}...")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
    for frame in all_frames:
        out.write(frame)
    out.release()
    print(f"[THREAD] Video saved: {filename}")

    # 2. Save to database and generate transcript
    from alerts.models import Alert
    from alerts.transcript import generate_transcript
    with open(filepath, "rb") as f:
        alert = Alert.objects.create(
            violations=", ".join(detected_violations),
            camera_id="Camera_1",
            video=File(f, name=filename)
        )
    print(f"[THREAD] Alert saved to database: {filename}")
    generate_transcript(alert, alert.video.path)


def run():
    # --- Django setup ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VIDEO_DIR = os.path.join(BASE_DIR, "violations")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "alertsite.settings")
    django.setup()

    # --- YOLO model setup ---
    MODEL_PATH = r"C:\Users\r125v\SurveillanceAi\runs\detect\yolov11_custom2\weights\best.pt"
    model = YOLO(MODEL_PATH)

    # --- Video capture ---
    video_source = r"C:\Users\r125v\Downloads\constructiondemo.mp4"
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    os.makedirs(VIDEO_DIR, exist_ok=True)

    # --- Frame Buffer: Continuously store the last 2 seconds of video ---
    pre_roll_seconds = 2
    frame_buffer = deque(maxlen=int(fps * pre_roll_seconds))

    # --- Alert cooldown and recording state ---
    last_alert_time = 0
    cooldown = 15
    
    recording_buffer = None
    recording_frames_remaining = 0
    recording_violations = None

    print("[RUNSCRIPT] run() started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())

        # If currently capturing post-violation frames
        if recording_frames_remaining > 0:
            recording_buffer.append(frame.copy())
            recording_frames_remaining -= 1
            if recording_frames_remaining == 0:
                # Finished capturing frames; start background thread to write/save/transcribe
                frame_size = (frame.shape[1], frame.shape[0])
                save_thread = threading.Thread(
                    target=save_alert_and_process,
                    args=(
                        recording_buffer, fps, frame_size, VIDEO_DIR, recording_violations
                    )
                )
                save_thread.daemon = True
                save_thread.start()
                recording_buffer = None
                recording_violations = None

        results = model(frame)
        detected_violations = set()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                detected_violations.add(label)

        is_recording = (recording_frames_remaining > 0)

        if detected_violations and not is_recording:
            now = time.time()
            if now - last_alert_time > cooldown:
                last_alert_time = now
                print(
                    f"[ALERT] Violations detected. Buffering post-violation frames in main loop...")

                # Initialize recording buffer with pre-roll frames
                recording_buffer = list(frame_buffer)
                recording_frames_remaining = int(fps * 8)
                recording_violations = detected_violations

        # This live preview will now run smoothly without any pausing
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
