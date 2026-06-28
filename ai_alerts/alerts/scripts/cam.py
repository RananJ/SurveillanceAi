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
    # Use MSMF backend on Windows to write H.264 (avc1) natively and bypass FFMPEG Cisco DLL lookup
    out = cv2.VideoWriter(filepath, cv2.CAP_MSMF, fourcc, fps, frame_size)
    for frame in all_frames:
        out.write(frame)
    out.release()
    print(f"[THREAD] Video saved: {filename}")

    # 2. Save to database and generate transcript
    from alerts.models import Alert
    from alerts.scripts.transcript import generate_transcript
    with open(filepath, "rb") as f:
        alert = Alert.objects.create(
            violations=", ".join(detected_violations),
            camera_id="Camera_1",
            video=File(f, name=filename)
        )
    print(f"[THREAD] Alert saved to database: {filename}")
    generate_transcript(alert, all_frames)

    # 3. Clean up the temporary file written by OpenCV to avoid duplication
    try:
        os.remove(filepath)
        print(f"[THREAD] Cleaned up temporary file: {filepath}")
    except Exception as e:
        print(f"[THREAD] Could not remove temporary file {filepath}: {e}")


def run():
    # --- Django setup ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    alerts_dir = os.path.dirname(current_script_dir)  # alerts directory
    django_root = os.path.dirname(alerts_dir)  # ai_alerts directory
    project_root = os.path.dirname(django_root)  # SurveillanceAi directory

    VIDEO_DIR = os.path.join(alerts_dir, "violations")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "alertsite.settings")
    django.setup()

    # --- YOLO model setup (Relative lookup only to avoid absolute path exposure) ---
    MODEL_PATH = os.path.join(project_root, "ai_alerts", "models", "best.pt")

    print(f"[RUNSCRIPT] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # --- Video capture (Check existence or use local fallback) ---
    #video_source = os.path.join(project_root,"ai_alerts","demo_video","constructiondemo.mp4")
    video_source=0

    print(f"[RUNSCRIPT] Opening video source: {video_source}")
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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append(frame.copy())

            # If currently capturing post-violation frames
            if recording_frames_remaining > 0:
                recording_buffer.append(frame.copy())
                recording_frames_remaining -= 1
                
                # Print buffering progress to guide the user
                if recording_frames_remaining % 15 == 0 or recording_frames_remaining == 0:
                    print(f"[RECORDING] Capturing safety violation: {recording_frames_remaining} frames remaining in buffer...")
                    
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

                    # Preload the VLM in the background while capturing
                    from alerts.scripts.transcript import preload_vlm_async
                    preload_vlm_async()

                    # Initialize recording buffer with pre-roll frames
                    recording_buffer = list(frame_buffer)
                    recording_frames_remaining = int(fps * 8)
                    recording_violations = detected_violations

            # This live preview will now run smoothly without any pausing
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv11 Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("[RUNSCRIPT] KeyboardInterrupt received. Exiting...")
    finally:
        print("[RUNSCRIPT] Releasing camera resources...")
        cap.release()
        cv2.destroyAllWindows()

        # Check if there is an unfinished recording in progress
        if recording_buffer is not None and len(recording_buffer) > 0:
            print(f"[SHUTDOWN] Finalizing active recording of {len(recording_buffer)} frames...")
            frame_size = (recording_buffer[0].shape[1], recording_buffer[0].shape[0])
            # Call synchronously so the script does not terminate before saving is complete
            save_alert_and_process(recording_buffer, fps, frame_size, VIDEO_DIR, recording_violations)


if __name__ == "__main__":
    run()
