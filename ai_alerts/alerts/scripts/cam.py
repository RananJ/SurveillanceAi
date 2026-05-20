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


def record_and_process_alert(pre_violation_frames, cap_source, current_frame_pos, fps, frame_size, video_dir, detected_violations):
    """
    This function runs entirely in the background. It captures the post-violation frames,
    combines them with the pre-violation frames, saves the clip directly using OpenCV,
    and updates the database.
    """
    # 1. Capture the 8 seconds of video AFTER the violation was detected.
    post_violation_frames = []
    frames_to_capture_after = int(fps * 8)  # Capture 8 seconds worth of frames

    # Create a new, independent video capture object for the thread
    cap = cv2.VideoCapture(cap_source)
    # Jump to the exact frame where the main loop was
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)

    for _ in range(frames_to_capture_after):
        ret, frame = cap.read()
        if not ret:
            break
        post_violation_frames.append(frame)
    cap.release()

    # Combine the frames from before and after the event
    all_frames = list(pre_violation_frames) + post_violation_frames

    # 2. Define filenames and save the clip using the 'avc1' (H.264) codec
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violation_{timestamp_str}.mp4"
    filepath = os.path.join(video_dir, filename)

    print(f"[THREAD] Recording {len(all_frames)} frames to {filename}...")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
    for frame in all_frames:
        out.write(frame)
    out.release()
    print(f"[THREAD] Video saved: {filename}")

    # 3. Save to database and generate transcript
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
    recording_thread = None

    print("[RUNSCRIPT] run() started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())

        results = model(frame)
        detected_violations = set()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                detected_violations.add(label)

        is_recording = recording_thread is not None and recording_thread.is_alive()

        if detected_violations and not is_recording:
            now = time.time()
            if now - last_alert_time > cooldown:
                last_alert_time = now
                print(
                    f"[ALERT] Violations detected. Starting background recording thread...")

                frame_size = (frame.shape[1], frame.shape[0])
                current_frame_position = cap.get(cv2.CAP_PROP_POS_FRAMES)

                # --- The main thread now ONLY starts the thread and continues ---
                recording_thread = threading.Thread(
                    target=record_and_process_alert,
                    args=(
                        frame_buffer.copy(), video_source, current_frame_position,
                        fps, frame_size, VIDEO_DIR, detected_violations
                    )
                )
                recording_thread.daemon = True
                recording_thread.start()

        # This live preview will now run smoothly without any pausing
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
